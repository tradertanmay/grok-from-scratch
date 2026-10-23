"""Named LLM providers with per-request model routing and automatic fallback.

Model ids are `provider/model` (e.g. `mac/gpt-oss:latest`); a bare model name goes to the
first provider that lists it, else the first provider.

Fallback order per request: the chosen model, then GROK_FALLBACK_MODELS. Providers that are
at their `max_concurrent` limit (spillover) or failed to connect in the last DOWN_SECONDS are
moved to the back of that order instead of being waited on.
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .llm import LLMClient, LLMError, OpenAICompatClient

log = logging.getLogger("grok_bot.providers")
DOWN_SECONDS = 30.0
# Set by the caller to a dict per turn; the router writes the `provider/model` that answered.
served_model: ContextVar[dict[str, str] | None] = ContextVar("served_model", default=None)


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key: str = "not-needed"
    models: tuple[str, ...] = ()
    max_concurrent: int = 0  # in-flight streams before spilling over; 0 = unlimited


def _resolve_key(value: str) -> str:
    return os.environ.get(value[1:], "") if value.startswith("$") else value


def load_providers(settings: Settings) -> list[Provider]:
    path = Path(settings.providers_file) if settings.providers_file else None
    if path and path.is_file():
        data = json.loads(path.read_text())
        providers = [
            Provider(
                name=p["name"],
                base_url=p["base_url"].rstrip("/"),
                api_key=_resolve_key(p.get("api_key", "not-needed")) or "not-needed",
                models=tuple(p.get("models", [])),
                max_concurrent=int(p.get("max_concurrent", 0)),
            )
            for p in data
        ]
    else:
        providers = []
    if not providers:
        providers = [
            Provider("default", settings.llm_base_url, settings.llm_api_key, (settings.llm_model,))
        ]
    first = providers[0]
    if settings.max_concurrent_primary and not first.max_concurrent:
        providers[0] = Provider(
            first.name, first.base_url, first.api_key, first.models, settings.max_concurrent_primary
        )
    return providers


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

    async def embed(self, texts: list[str], model: str) -> list[list[float]]:
        provider, name = self.resolve(model)
        return await self._client(provider).embed(texts, name)

    async def list_providers(self) -> list[dict[str, Any]]:
        async def probe(p: Provider) -> dict[str, Any]:
            try:
                names, healthy = await self._client(p).list_models(), True
            except Exception:
                names, healthy = list(p.models), False
            return {"name": p.name, "models": [f"{p.name}/{n}" for n in names], "healthy": healthy}

        return list(await asyncio.gather(*(probe(p) for p in self.providers)))

    async def list_models(self) -> list[str]:
        return [m for p in await self.list_providers() for m in p["models"]]
