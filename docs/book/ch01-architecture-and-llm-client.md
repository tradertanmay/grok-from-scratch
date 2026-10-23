# Chapter 1 — Architecture, Streaming and Multi-Provider Routing

> **What you build:** the bottom layer of Grok Bot. This is a client that streams from any OpenAI-compatible model server, plus a router that picks a provider per request. When a provider is down or full, the router moves on to the next one. Everything else in the book runs on top of these ~240 lines (`grok_bot/llm.py`, `grok_bot/providers.py`).

## 1.1 The whole system on one page

```
browser / CLI / OpenAI SDK
        │  POST /api/chat  or  POST /v1/chat/completions
        ▼
FastAPI app (grok_bot/app.py) ── auth, quotas, sessions, files, memory recall
        │  turn_events(): builds the system prompt and picks the model
        ▼
Agent / DeepSearch (grok_bot/agent.py, deepsearch.py) ── the loop: stream → tool calls → repeat
        │  stream_chat(messages, tools, model)
        ▼
ProviderRouter (grok_bot/providers.py) ── provider/model routing, spillover, failover
        │
        ▼
OpenAICompatClient (grok_bot/llm.py) ── HTTP + SSE to Ollama on a Mac, Groq, OpenRouter, vLLM on an H100 …
```

Each layer only knows about the one below it. Events flow back up as small JSON dicts (`{"type": "token", ...}`, `{"type": "tool_call", ...}`), and the web layer turns them into Server-Sent Events.

**Design decision: speak only the OpenAI wire protocol.** Ollama, vLLM, llama.cpp, LM Studio, Groq, OpenRouter, Together and OpenAI all expose `POST /v1/chat/completions` with `stream: true`. If we code against that one protocol, the "which model and where" question becomes configuration, not code. That is how the project answers "what if we don't have an H100?": the same code runs against a free cloud tier on any laptop, a small local model in Ollama, a big workstation or GPU box, or several of these at once.

## 1.2 The client interface

The rest of the code depends on a `Protocol`, not on a concrete class:

`grok_bot/llm.py` (lines 10–19):

```python
class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield OpenAI-style `delta` dicts (content / tool_calls)."""
        ...
```

A "delta" is the `choices[0].delta` object from a streaming chunk. It holds a piece of `content`, a piece of a `tool_calls` entry, or (on reasoning models) a piece of `reasoning`/`reasoning_content`. Keeping that raw shape and not inventing our own types means the agent loop in Chapter 2 can consume any provider unchanged.

Because it is a `Protocol`, tests can swap in a scripted fake with no network at all. This is `tests/conftest.py`, which almost every test in the repo uses:

`tests/conftest.py` (lines 7–25):

```python
class FakeLLM:
    """Replays scripted turns; each turn is a list of OpenAI-style deltas."""

    def __init__(self, turns: list[list[dict[str, Any]]]):
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    async def stream_chat(self, messages, tools, model) -> AsyncIterator[dict[str, Any]]:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools, "model": model})
        turn = self.turns.pop(0) if self.turns else [{"content": "(no more turns)"}]
        for delta in turn:
            yield delta

    async def list_models(self) -> list[str]:
        return ["fake-model"]


def text_turn(*pieces: str) -> list[dict[str, Any]]:
    return [{"content": p} for p in pieces]
```

## 1.3 Streaming from an OpenAI-compatible server

`grok_bot/llm.py` (lines 22–57):

```python
class OpenAICompatClient:
    def __init__(
        self, base_url: str, api_key: str, timeout: float = 300.0, connect_timeout: float = 5.0
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # A short connect timeout lets an offline provider (e.g. a sleeping Mac) fail over fast.
        self.timeout = httpx.Timeout(timeout, connect=connect_timeout)

    async def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str
    ) -> AsyncIterator[dict[str, Any]]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")[:500]
                    raise LLMError(f"LLM HTTP {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise LLMError(str(chunk["error"]))
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        if delta:
                            yield delta
```

Things worth noticing:

- **Two timeouts.** The read timeout is long (300 s), because a 35B model on a laptop can think for a while. The *connect* timeout is short (5 s). A sleeping Mac or a dead tunnel then fails in seconds, not minutes, which is what makes failover (1.5) fast. This change came out of our 20-user load test (Chapter 11).
- **Errors can arrive mid-stream.** Some servers return HTTP 200 and then send a chunk like `{"error": ...}`. The client turns that into an `LLMError` instead of silently stopping.
- **Ignore everything that isn't `data:`.** SSE allows comments and keep-alives, and `[DONE]` ends the stream.

The same class also implements `embed()` (used for memory and RAG in Chapter 4) and `list_models()` (used by the model picker).

## 1.4 Providers and model ids

A deployment can have several providers, listed in `providers.json` (`GROK_PROVIDERS_FILE`). Here is `providers.example.json`:

`providers.example.json` (lines 1–14):

```json
[
  {"name": "mac", "base_url": "http://localhost:11434/v1",
   "models": ["gpt-oss:latest", "qwen3.6:35b", "gemma4:12b"], "max_concurrent": 4},
  {"name": "mlx", "base_url": "http://127.0.0.1:8088/v1",
   "models": ["grok-lora"], "max_concurrent": 1},
  {"name": "h100", "base_url": "http://H100_HOST:8000/v1",
   "models": ["openai/gpt-oss-120b"]},
  {"name": "groq", "base_url": "https://api.groq.com/openai/v1", "api_key": "$GROQ_API_KEY",
   "models": []},
  {"name": "openrouter", "base_url": "https://openrouter.ai/api/v1",
   "api_key": "$OPENROUTER_API_KEY", "models": []},
  {"name": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
   "api_key": "$GEMINI_API_KEY", "models": []}
]
```

Model ids are written `provider/model`, for example `mac/gpt-oss:latest`. `resolve()` maps an id to a provider:

`grok_bot/providers.py` (lines 104–113):

```python
    def resolve(self, model: str) -> tuple[Provider, str]:
        prefix, sep, rest = model.partition("/")
        if sep:
            for p in self.providers:
                if p.name == prefix:
                    return p, rest
        for p in self.providers:
            if model in p.models:
                return p, model
        return self.providers[0], model
```

API keys can be written as `"$GROQ_API_KEY"`, so secrets stay in the environment and out of the file (`_resolve_key`). With no `providers.json`, the router falls back to a single `default` provider built from `LLM_BASE_URL`/`LLM_MODEL`, so a one-model setup needs no config file at all.

## 1.5 The router: fallback, cooldown and spillover

`grok_bot/providers.py` (lines 74–143):

```python
class ProviderRouter:
    def __init__(
        self,
        providers: list[Provider],
        fallback_models: list[str] | None = None,
        client_factory: Callable[[Provider], LLMClient] | None = None,
        connect_timeout: float = 5.0,
    ):
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self.fallback_models = fallback_models or []
        self._factory = client_factory or (
            lambda p: OpenAICompatClient(p.base_url, p.api_key, connect_timeout=connect_timeout)
        )
        self._clients: dict[str, LLMClient] = {}
        self.inflight: dict[str, int] = {}
        self.down_until: dict[str, float] = {}

    def available(self, provider: Provider) -> bool:
        if self.down_until.get(provider.name, 0) > time.monotonic():
            return False
        cap = provider.max_concurrent
        return not cap or self.inflight.get(provider.name, 0) < cap

    def _client(self, provider: Provider) -> LLMClient:
        if provider.name not in self._clients:
            self._clients[provider.name] = self._factory(provider)
        return self._clients[provider.name]

    def resolve(self, model: str) -> tuple[Provider, str]:
        prefix, sep, rest = model.partition("/")
        if sep:
            for p in self.providers:
                if p.name == prefix:
                    return p, rest
        for p in self.providers:
            if model in p.models:
                return p, model
        return self.providers[0], model

    async def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str
    ) -> AsyncIterator[dict[str, Any]]:
        chain = [self.resolve(m) for m in [model, *(m for m in self.fallback_models if m != model)]]
        ready = [c for c in chain if self.available(c[0])]
        if ready and ready[0] is not chain[0]:
            log.info("spilling over from %s to %s", chain[0][0].name, ready[0][0].name)
        errors = []
        for provider, name in ready + [c for c in chain if c not in ready]:
            started = False
            self.inflight[provider.name] = self.inflight.get(provider.name, 0) + 1
            try:
                async for delta in self._client(provider).stream_chat(messages, tools, name):
                    if not started:
                        started = True
                        self.down_until.pop(provider.name, None)
                        if (served := served_model.get()) is not None:
                            served["model"] = f"{provider.name}/{name}"
                    yield delta
                return
            except (httpx.TransportError, LLMError) as e:
                if started:
                    raise
                if isinstance(e, httpx.ConnectError | httpx.ConnectTimeout):
                    self.down_until[provider.name] = time.monotonic() + DOWN_SECONDS
                errors.append(f"{provider.name}/{name}: {type(e).__name__}: {e}")
            finally:
                self.inflight[provider.name] -= 1
        raise LLMError("all providers failed: " + "; ".join(errors))
```

Read `stream_chat` as three rules:

1. **Build the chain.** First the requested model, then `GROK_FALLBACK_MODELS` in order.
2. **Skip busy or down providers first.** `available()` is false when a provider failed to *connect* in the last `DOWN_SECONDS` (30 s), or when it already has `max_concurrent` streams open. Those providers move to the back of the chain. They aren't removed, so if everything is busy we still try them instead of failing.
3. **Never switch providers after the first token.** Once `started` is true, an error is re-raised. Switching mid-answer would splice two different models' text together, so the turn fails visibly instead.

`inflight` is decremented in `finally`, so a client that disconnects mid-stream still frees its slot.

### Why spillover matters: the numbers

We load-tested a local workstation running Ollama with 20 concurrent users (`python -m grok_bot.loadtest`, Chapter 11):

| setup | p50 time to first token | throughput | source |
|---|---|---|---|
| Mac only | 60.45 s | 8.8 turns/min | Queued in 5 waves of 4 |
| Mac `max_concurrent: 4` + cloud fallback | 0.23 s | 62.8–80.2 turns/min | 4 local + 16 cloud burst |

A laptop can only serve a few streams at a time. Everyone above that limit waited in a queue. Spillover sends them to a cloud fallback straight away.

### Which provider actually answered?

Usage stats and `/metrics` should record the model that *answered*, not the one that was requested. The router writes it into a per-turn `ContextVar` holder as soon as a provider starts streaming (`served_model`, lines 29–30 and 125–129 above). After that change, the same 20-user test reported the split correctly:

```text
grok_chat_turns_total{model="cloud/llama-3.3-70b-versatile",outcome="ok"} 16
grok_chat_turns_total{model="mac/gpt-oss:latest",outcome="ok"} 4
```

The holder is a mutable `dict` rather than a plain value because a `ContextVar.set()` inside a child task isn't visible to the parent. Mutating a shared dict is.

## Try it

Run the provider tests (no model server needed):

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q tests/test_providers.py
```

See failover and spillover with fake providers. Save this as `try_router.py` and run `.venv/bin/python try_router.py`:

```python
import asyncio

import httpx

from grok_bot.providers import Provider, ProviderRouter, served_model


class Down:
    async def stream_chat(self, messages, tools, model):
        raise httpx.ConnectError("mac is asleep")
        yield


class Echo:
    def __init__(self, name):
        self.name = name

    async def stream_chat(self, messages, tools, model):
        yield {"content": f"hello from {self.name}/{model}"}


async def main():
    providers = [Provider("mac", "http://mac/v1"), Provider("cloud", "http://cloud/v1")]
    clients = {"mac": Down(), "cloud": Echo("cloud")}
    router = ProviderRouter(providers, ["cloud/llama-3.3-70b"], lambda p: clients[p.name])
    for attempt in range(2):
        served = {}
        served_model.set(served)
        text = "".join([d["content"] async for d in router.stream_chat([], [], "mac/gpt-oss")])
        print(attempt, text, served, "mac down for 30s:", "mac" in router.down_until)


asyncio.run(main())
```

Expected output:

```text
0 hello from cloud/llama-3.3-70b {'model': 'cloud/llama-3.3-70b'} mac down for 30s: True
1 hello from cloud/llama-3.3-70b {'model': 'cloud/llama-3.3-70b'} mac down for 30s: True
```

On the second attempt the Mac isn't even tried first, because it is in its 30-second cooldown.

With a real model, pick one that fits your machine (Chapter 12 has the full setups):

- **Any laptop, no GPU:** get a free Groq key and set `LLM_BASE_URL=https://api.groq.com/openai/v1`, `LLM_API_KEY=gsk_...`, `LLM_MODEL=openai/gpt-oss-120b` (or `llama-3.3-70b-versatile`).
- **8–16 GB laptop:** install [Ollama](https://ollama.com), run `ollama pull qwen3:4b` (`2.5 GB`) and set `LLM_MODEL=qwen3:4b`.
- **32 GB+ machine:** `ollama pull gpt-oss` and keep the defaults (`LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=gpt-oss:latest`).

Then run `python -m grok_bot` and open http://127.0.0.1:8000.

## Exercises

1. Add a `priority` field to `Provider` and sort the fallback chain by it. Which test in `tests/test_providers.py` do you need to update?
2. A provider that returns HTTP 429 (rate-limited) is currently retried on every request. Make `LLMError` carry the status code, and put 429s into cooldown too.
3. Why does the router decrement `inflight` in `finally` and not after the `async for` loop? Write a test where the consumer stops reading after the first delta.
4. `served_model` records the *last* provider in a multi-step tool turn. Change it to record every provider used, and decide how `/metrics` should count a turn served by two providers.
