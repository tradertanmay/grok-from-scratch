"""Streaming client for any OpenAI-compatible /chat/completions endpoint."""

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol

import httpx


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    def stream_chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], model: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield OpenAI-style `delta` dicts (content / tool_calls)."""
        ...


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

    async def list_models(self) -> list[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/models", headers=headers)
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                json={"model": model, "input": texts},
                headers=headers,
            )
            if resp.status_code >= 400:
                raise LLMError(f"embeddings HTTP {resp.status_code}: {resp.text[:300]}")
            data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in data]
