"""The agent loop: stream from the model, run tool calls, repeat until a final answer."""

import json
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient
from .tools import ToolRegistry

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_TAG_OPEN = "<tool_call>"
TOOL_TAG_CLOSE = "</tool_call>"
_TOOL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


@dataclass
class ParsedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""

    def as_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments or json.dumps(self.arguments),
            },
        }


def _new_id() -> str:
    return "call_" + uuid.uuid4().hex[:12]


def _parse_args(raw: str | dict | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return val if isinstance(val, dict) else {"_raw": val}


def parse_text_tool_calls(text: str) -> tuple[str, list[ParsedToolCall]]:
    """Extract `<tool_call>{"name":..., "arguments":{...}}</tool_call>` blocks from text."""
    calls = []
    for m in _TOOL_TAG_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        calls.append(ParsedToolCall(_new_id(), str(obj["name"]), _parse_args(args)))
    return _TOOL_TAG_RE.sub("", text).strip(), calls


class TagHoldback:
    """Streams text through, but withholds anything that is (or may become) a <tool_call> block."""

    def __init__(self):
        self.buf = ""
        self.in_tag = False

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out = ""
        while self.buf:
            if self.in_tag:
                end = self.buf.find(TOOL_TAG_CLOSE)
                if end == -1:
                    return out
                self.buf = self.buf[end + len(TOOL_TAG_CLOSE) :]
                self.in_tag = False
                continue
            start = self.buf.find(TOOL_TAG_OPEN)
            if start != -1:
                out += self.buf[:start]
                self.buf = self.buf[start + len(TOOL_TAG_OPEN) :]
                self.in_tag = True
                continue
            keep = _partial_suffix(self.buf, TOOL_TAG_OPEN)
            out += self.buf[: len(self.buf) - keep]
            self.buf = self.buf[len(self.buf) - keep :]
            return out
        return out

    def flush(self) -> str:
        rest = "" if self.in_tag else self.buf
        self.buf = ""
        return rest


def _partial_suffix(buf: str, tag: str) -> int:
    """Length of the longest suffix of `buf` that is a proper prefix of `tag`."""
    for k in range(min(len(tag) - 1, len(buf)), 0, -1):
        if tag.startswith(buf[-k:]):
            return k
    return 0


class ThinkSplitter:
    """Splits streamed content into ("think", text) and ("text", text) parts on <think> tags."""

    def __init__(self):
        self.buf = ""
        self.in_think = False

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        self.buf += chunk
        parts: list[tuple[str, str]] = []
        while self.buf:
            tag = THINK_CLOSE if self.in_think else THINK_OPEN
            kind = "think" if self.in_think else "text"
            idx = self.buf.find(tag)
            if idx != -1:
                if idx:
                    parts.append((kind, self.buf[:idx]))
                self.buf = self.buf[idx + len(tag) :]
                self.in_think = not self.in_think
                continue
            keep = _partial_suffix(self.buf, tag)
            if len(self.buf) > keep:
                parts.append((kind, self.buf[: len(self.buf) - keep]))
            self.buf = self.buf[len(self.buf) - keep :]
            break
        return parts

    def flush(self) -> list[tuple[str, str]]:
        rest, self.buf = self.buf, ""
        return [("think" if self.in_think else "text", rest)] if rest else []


@dataclass
class _StreamedCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class Agent:
    llm: LLMClient
    tools: ToolRegistry
    model: str
    max_steps: int = 8
    think: bool = False
    new_messages: list[dict[str, Any]] = field(default_factory=list)

    async def run(self, messages: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        """Yield UI events. Every message the agent produces is appended to `messages`
        and to `self.new_messages` (for persistence)."""
        schemas = self.tools.schemas()
        for step in range(self.max_steps):
            use_tools = step < self.max_steps - 1
            content = ""
            reasoning = ""
            streamed: dict[int, _StreamedCall] = {}
            holdback = TagHoldback()
            splitter = ThinkSplitter()
            async for delta in self.llm.stream_chat(
                messages, schemas if use_tools else [], self.model
            ):
                native = delta.get("reasoning") or delta.get("reasoning_content")
                parts = [("think", native)] if native else []
                if delta.get("content"):
                    parts += splitter.feed(delta["content"])
                for kind, text in parts:
                    if kind == "think":
                        reasoning += text
                        if self.think:
                            yield {"type": "think", "content": text}
                        continue
                    content += text
                    visible = holdback.feed(text)
                    if visible:
                        yield {"type": "token", "content": visible}
                for tc in delta.get("tool_calls") or []:
                    sc = streamed.setdefault(tc.get("index", len(streamed)), _StreamedCall())
                    sc.id = tc.get("id") or sc.id
                    fn = tc.get("function") or {}
                    sc.name += fn.get("name") or ""
                    args = fn.get("arguments")
                    sc.arguments += json.dumps(args) if isinstance(args, dict) else (args or "")
            for kind, text in splitter.flush():
                if kind == "think":
                    reasoning += text
                    if self.think:
                        yield {"type": "think", "content": text}
                else:
                    content += text
                    visible = holdback.feed(text)
                    if visible:
                        yield {"type": "token", "content": visible}
            tail = holdback.flush()
            if tail:
                yield {"type": "token", "content": tail}

            calls = [
                ParsedToolCall(sc.id or _new_id(), sc.name, _parse_args(sc.arguments), sc.arguments)
                for _, sc in sorted(streamed.items())
                if sc.name
            ]
            clean_content = content
            if not calls and TOOL_TAG_OPEN in content:
                clean_content, calls = parse_text_tool_calls(content)

            assistant: dict[str, Any] = {"role": "assistant", "content": clean_content.strip()}
            if reasoning and self.think:
                assistant["reasoning"] = reasoning
            if calls:
                assistant["tool_calls"] = [c.as_openai() for c in calls]
            self._append(messages, assistant)
            if not calls:
                return

            for call in calls:
                yield {
                    "type": "tool_call",
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                result = await self.tools.call(call.name, call.arguments)
                yield {
                    "type": "tool_result",
                    "id": call.id,
                    "name": call.name,
                    "output": result.output,
                    "is_error": result.is_error,
                }
                self._append(
                    messages,
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result.output,
                    },
                )

    def _append(self, messages: list[dict[str, Any]], message: dict[str, Any]) -> None:
        messages.append(message)
        self.new_messages.append(message)
