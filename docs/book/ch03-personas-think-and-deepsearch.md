# Chapter 3 — Personas, Think Mode and DeepSearch

> **What you build:** what makes the bot feel like Grok. Two personas (witty "fun" and neutral "regular"), streamed reasoning ("Think"), and DeepSearch, a fixed research pipeline that plans searches, reads sources in parallel and writes a cited answer. Files: `grok_bot/persona.py`, `grok_bot/agent.py` (`ThinkSplitter`), `grok_bot/deepsearch.py`.

## 3.1 Personas are just system prompts

`grok_bot/persona.py` (lines 1–63):

```python
from datetime import UTC, datetime
from typing import Any, Literal

Mode = Literal["fun", "regular"]

FUN_STYLE = """You are Grok, a maximally truth-seeking AI assistant with a sharp, witty, \
slightly rebellious sense of humor (think Hitchhiker's Guide to the Galaxy meets JARVIS).

Style:
- Be genuinely helpful first, funny second. Never let a joke get in the way of a correct answer.
- Be direct and concise. No corporate hedging, no moralizing lectures.
- Playful, good-natured roasting or banter when the user asks to be roasted is \
harmless fun—always oblige with wit.
- Have opinions and never be sycophantic: if the user pitches an absurd, terrible, \
or factually wrong idea (even if they ask you to praise it), push back directly and \
say why it is nonsense, while admitting genuine uncertainty honestly."""

REGULAR_STYLE = """You are Grok, a maximally truth-seeking AI assistant.

Style:
- Be clear, accurate and neutral in tone. No jokes or sarcasm unless the user asks for them.
- Be direct and concise. No corporate hedging, no moralizing lectures.
- Give well-reasoned answers, push back plainly on flawed premises instead of being \
sycophantic, and admit uncertainty honestly."""

TOOLS_AND_DATE = """

Tools:
- Use web_search and fetch_url for recent events, live facts, anything you are \
unsure about, or when the user asks you to search the web. Cite the URLs you used.
- Do NOT call web_search for greetings, roasts, brainstorming, opinions, math, or \
standard programming concepts you already know.
- Use python_exec for math, data crunching, or anything that needs exact computation.
- Do not invent tool results. If a tool fails, say so and try another approach.

Current date and time (UTC): {now}"""

MEMORY = """

Long-term memory:
- When the user shares a durable fact about themselves (name, preferences, projects) or asks
  you to remember something, save it with remember. Don't save secrets or one-off details.
- When asked to forget something, or a remembered fact is wrong, call forget with its id.
- Use what you remember naturally; don't recite it unprompted."""

STYLES: dict[str, str] = {"fun": FUN_STYLE, "regular": REGULAR_STYLE}


def system_prompt(
    mode: Mode = "fun",
    now: datetime | None = None,
    memories: list[dict[str, Any]] | None = None,
) -> str:
    """`memories=None` means memory is off; a list (maybe empty) enables the memory tools."""
    now = now or datetime.now(UTC)
    prompt = STYLES[mode] + TOOLS_AND_DATE.format(now=now.strftime("%Y-%m-%d %H:%M"))
    if memories is None:
        return prompt
    prompt += MEMORY
    if memories:
        facts = "\n".join(f"- [{m['id']}] {m['content']}" for m in memories)
        prompt += f"\n\nRelevant things you remember about this user ([id] fact):\n{facts}"
    return prompt
```

Three things this prompt does on purpose:

- **Anti-sycophancy is spelled out.** "If the user pitches an absurd, terrible, or factually wrong idea (even if they ask you to praise it), push back." Small open models happily agree with you unless told not to. The eval harness (Chapter 6) has a case for exactly this.
- **Tool etiquette.** "Do NOT call `web_search` for greetings, roasts…" Without this line, small models search the web for "hi".
- **The current date.** Models have a training cutoff. Without the date they confidently mix up "this year" with it.

`memories=None` means "memory is off for this turn", while `[]` means "on, but nothing recalled". The memory tool instructions are only added in the second case (Chapter 4).

A fine-tuned LoRA can push the style into the weights (Chapter 10). The prompt still does most of the work, and it works on any model, so you don't need a GPU to get the persona.

## 3.2 Think mode: separating reasoning from the answer

Reasoning models stream their thoughts in one of two ways:

1. A separate field: `delta.reasoning` (Ollama's gpt-oss) or `delta.reasoning_content` (DeepSeek, vLLM).
2. Inline in the text as `<think> … </think>` (Qwen, DeepSeek-R1 distills).

The agent loop (Chapter 2) handles case 1 directly: `native = delta.get("reasoning") or delta.get("reasoning_content")`. For case 2 there is a streaming splitter:

`grok_bot/agent.py` (lines 113–142):

```python
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
```

The hard part is that a tag can be split across chunks: `"<thi"` in one delta, `"nk>"` in the next. `_partial_suffix` holds back the shortest tail that could still turn into a tag, so nothing is mis-routed and nothing is delayed for longer than that.

When `think=False` the reasoning is still stripped from the answer but never sent to the user. When `think=True` it is streamed as `{"type": "think"}` events and saved as `assistant["reasoning"]`.

Think turns can also be routed to a stronger model. With `GROK_THINK_MODEL=groq/qwen/qwen3.8-27b` (or `groq/openai/gpt-oss-120b`) (a free cloud model), or `mac/qwen3.6:35b` on a big machine, normal chat stays on the fast model and Think uses the slower, smarter one (`grok_bot/app.py`):

`grok_bot/app.py` (lines 876–882):

```python
        model = req.model or settings.llm_model
        if req.provider and not model.startswith(req.provider + "/"):
            model = f"{req.provider}/{model}"
        if req.think and not req.model and not req.provider and settings.think_model:
            model = settings.think_model
        if images and settings.vision_model:
            model = settings.vision_model
```

The same block routes turns that include images to `GROK_VISION_MODEL` (Chapter 4).

## 3.3 DeepSearch: a fixed pipeline, not a free-form agent

You *could* do research by giving the Chapter 2 agent `web_search` and hoping it loops well. Small local models don't: they search once and answer, or loop on the same query. DeepSearch instead runs a fixed plan → search → read → write pipeline. The model is only asked to do the parts it is good at:

`grok_bot/deepsearch.py` (lines 110–153):

```python
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
```

1. **Plan.** One non-streaming call asks for "ONLY a JSON array of {n} short, diverse web search queries". `parse_queries` accepts a JSON array, or falls back to bullet/numbered lines, or finally the question itself. So a model that ignores the format still works.
2. **Search in parallel.** `_batch` runs all queries with `asyncio.gather` and records them as *one* assistant turn with several `tool_calls`. That keeps the stored history a valid OpenAI transcript.
3. **Read.** `pick_urls` takes result URLs round-robin across queries (rank 1 of every query, then rank 2, …), so one query can't use up all the sources. Then it fetches them in parallel and trims each to `source_chars` (1,200).
4. **Write.** A one-step `Agent` with **no tools** gets the numbered sources and is told to cite `[1]`, `[2]`. Because it is a normal `Agent`, Think mode and streaming work unchanged.

Every stage sends a `{"type": "status", "step": ...}` event, so the UI can show "Searching: … | …" while the user waits.

`grok_bot/deepsearch.py` (lines 84–108):

```python
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
```

## Try it

Run DeepSearch end to end with a scripted model and fake tools. Save as `try_deepsearch.py` in the repo root and run `PYTHONPATH=. .venv/bin/python try_deepsearch.py`:

```python
import asyncio

from grok_bot.deepsearch import DeepSearch
from grok_bot.tools import Tool, ToolRegistry
from tests.conftest import FakeLLM, text_turn


def search(query, max_results=5):
    return f"Result for {query}\nhttps://example.com/{query.replace(' ', '-')}"


def fetch(url):
    return f"page text of {url}"


tools = ToolRegistry([Tool("web_search", "", {}, search), Tool("fetch_url", "", {}, fetch)])
llm = FakeLLM(
    [
        text_turn('["ollama mac speed", "groq free tier limits"]'),
        text_turn("Mac is free but slow [1]; Groq is fast but rate-limited [2]."),
    ]
)


async def main():
    question = [{"role": "user", "content": "Mac or Groq for a free bot?"}]
    async for e in DeepSearch(llm, tools, "fake").run(question):
        print(e["type"], e.get("detail") or e.get("arguments") or e.get("content") or "")
    print("--- prompt the writer saw ---")
    print(llm.calls[-1]["messages"][-1]["content"][-260:])


asyncio.run(main())
```

Output:

```text
status Planning searches
status Searching: ollama mac speed | groq free tier limits
tool_call {'query': 'ollama mac speed'}
tool_call {'query': 'groq free tier limits'}
tool_result 
tool_result 
status Reading 2 sources
tool_call {'url': 'https://example.com/ollama-mac-speed'}
tool_call {'url': 'https://example.com/groq-free-tier-limits'}
tool_result 
tool_result 
status Writing answer
token Mac is free but slow [1]; Groq is fast but rate-limited [2].
--- prompt the writer saw ---
 for each source you cited.

Question: Mac or Groq for a free bot?

[1] https://example.com/ollama-mac-speed
page text of https://example.com/ollama-mac-speed

[2] https://example.com/groq-free-tier-limits
page text of https://example.com/groq-free-tier-limits
```

The persona and Think tests:

```bash
.venv/bin/pytest -q tests/test_modes.py
```

With a real model, use the Think and DeepSearch toggles in the web UI (Chapter 7), or send `"think": true` or `"deepsearch": true` in the `POST /api/chat` body.

## Exercises

1. Add a third persona, `"tutor"`, that answers with a question first. Where do `Mode`, `STYLES` and the API's validation need to change?
2. `ThinkSplitter` assumes there are no nested `<think>` tags. Write a test with `"<think>a<think>b</think>c</think>d"` and decide what the right output is.
3. Give DeepSearch a second round: after reading, ask the model "what is still unanswered?" and run up to 2 more queries. Keep the plan step's JSON parsing just as forgiving.
4. Sources are trimmed to the first 1,200 characters, which is often navigation text. Use `Files`' chunk search (Chapter 4) to pick the most relevant 1,200 characters of each page.
