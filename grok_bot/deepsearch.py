"""DeepSearch: plan queries -> parallel web search -> read top sources -> cited answer."""

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, ThinkSplitter
from .llm import LLMClient
from .tools import ToolRegistry

PLAN_PROMPT = """You plan web research. Given the user's question, reply with ONLY a JSON array \
of {n} short, diverse web search queries that together cover it. No prose."""

ANSWER_PROMPT = """Answer the question below using the numbered sources. Do not call any tools. \
Cite claims inline using ASCII square brackets as [1], [2], ... matching the source numbers. \
If the sources disagree or don't cover something, say so. End with a "Sources:" list of \
`[n] https://...` for each source you cited.

Question: {question}

{sources}"""

_URL_LINE = re.compile(r"^(https?://\S+)$", re.M)


def parse_queries(text: str, fallback: str, limit: int) -> list[str]:
    match = re.search(r"\[.*\]", text, re.S)
    if match:
        try:
            items = json.loads(match.group(0))
            queries = [str(q).strip() for q in items if str(q).strip()]
            if queries:
                return queries[:limit]
        except json.JSONDecodeError:
            pass
    lines = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", ln).strip(' "') for ln in text.splitlines()]
    queries = [ln for ln in lines if ln]
    return queries[:limit] or [fallback]


def pick_urls(search_outputs: list[str], limit: int) -> list[str]:
    """Round-robin the result URLs across queries so every query contributes, deduplicated."""
    per_query = [_URL_LINE.findall(out) for out in search_outputs]
    urls: list[str] = []
    for rank in range(max((len(u) for u in per_query), default=0)):
        for found in per_query:
            if rank < len(found) and found[rank] not in urls:
                urls.append(found[rank])
                if len(urls) == limit:
                    return urls
    return urls


def _status(step: str, detail: str) -> dict[str, Any]:
    return {"type": "status", "step": step, "detail": detail, "content": detail}


def _call_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


@dataclass
class DeepSearch:
    llm: LLMClient
    tools: ToolRegistry
    model: str
    think: bool = False
    max_queries: int = 4
    max_sources: int = 5
    source_chars: int = 1200
    new_messages: list[dict[str, Any]] = field(default_factory=list)

    async def _complete(self, messages: list[dict[str, Any]]) -> str:
        splitter, out = ThinkSplitter(), ""
        async for delta in self.llm.stream_chat(messages, [], self.model):
            for kind, text in splitter.feed(delta.get("content") or ""):
                out += text if kind == "text" else ""
        return out + "".join(t for k, t in splitter.flush() if k == "text")

    async def _batch(
        self, messages: list[dict[str, Any]], tool: str, args_list: list[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one tool over many argument sets in parallel; record as one assistant turn."""
        ids = [_call_id() for _ in args_list]
        calls = [
            {"id": i, "type": "function", "function": {"name": tool, "arguments": json.dumps(a)}}
            for i, a in zip(ids, args_list, strict=True)
        ]
        self._append(messages, {"role": "assistant", "content": "", "tool_calls": calls})
        for i, a in zip(ids, args_list, strict=True):
            yield {"type": "tool_call", "id": i, "name": tool, "arguments": a}
        results = await asyncio.gather(*(self.tools.call(tool, a) for a in args_list))
        for i, r in zip(ids, results, strict=True):
            yield {
                "type": "tool_result",
                "id": i,
                "name": tool,
                "output": r.output,
                "is_error": r.is_error,
            }
            content = r.output[: self.source_chars]
            self._append(
                messages, {"role": "tool", "tool_call_id": i, "name": tool, "content": content}
            )

    async def run(self, messages: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        prior = list(messages[:-1])
        question = str(messages[-1].get("content") or "")
        yield _status("plan", "Planning searches")
        plan = await self._complete(
            [
                {"role": "system", "content": PLAN_PROMPT.format(n=self.max_queries)},
                {"role": "user", "content": question},
            ]
        )
        queries = parse_queries(plan, question, self.max_queries)

        yield _status("search", "Searching: " + " | ".join(queries))
        searches: list[str] = []
        async for event in self._batch(messages, "web_search", [{"query": q} for q in queries]):
            if event["type"] == "tool_result" and not event["is_error"]:
                searches.append(event["output"])
            yield event

        urls = pick_urls(searches, self.max_sources)
        sources: list[tuple[str, str]] = []
        if urls:
            yield _status("read", f"Reading {len(urls)} sources")
            by_id: dict[str, str] = {}
            async for event in self._batch(messages, "fetch_url", [{"url": u} for u in urls]):
                if event["type"] == "tool_call":
                    by_id[event["id"]] = event["arguments"]["url"]
                elif not event["is_error"]:
                    sources.append((by_id[event["id"]], event["output"][: self.source_chars]))
                yield event
        if not sources:
            sources = [("web search results", s) for s in searches]

        yield _status("answer", "Writing answer")
        numbered = (
            "\n\n".join(f"[{n}] {url}\n{text}" for n, (url, text) in enumerate(sources, 1))
            or "(no sources found)"
        )
        writer = Agent(self.llm, ToolRegistry([]), self.model, max_steps=1, think=self.think)
        prompt = ANSWER_PROMPT.format(question=question, sources=numbered)
        async for event in writer.run([*prior, {"role": "user", "content": prompt}]):
            yield event
        self.new_messages.extend(writer.new_messages)
        messages.extend(writer.new_messages)

    def _append(self, messages: list[dict[str, Any]], message: dict[str, Any]) -> None:
        messages.append(message)
        self.new_messages.append(message)
