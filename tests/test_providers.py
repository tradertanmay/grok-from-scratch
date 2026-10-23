import asyncio
import json

import httpx
import pytest

from grok_bot.config import Settings
from grok_bot.llm import LLMError
from grok_bot.providers import Provider, ProviderRouter, load_providers
from tests.conftest import FakeLLM, text_turn


class DownLLM:
    async def stream_chat(self, messages, tools, model):
        raise httpx.ConnectError("refused")
        yield  # pragma: no cover

    async def list_models(self):
        raise httpx.ConnectError("refused")


class MidStreamFailLLM:
    async def stream_chat(self, messages, tools, model):
        yield {"content": "partial"}
        raise LLMError("boom")


PROVIDERS = [
    Provider("mac", "http://mac/v1", models=("gpt-oss:latest",)),
    Provider("h100", "http://h100/v1", models=("openai/gpt-oss-120b",)),
]


def router(clients, fallback=None):
    return ProviderRouter(PROVIDERS, fallback, client_factory=lambda p: clients[p.name])


async def drain(r, model):
    return [d async for d in r.stream_chat([], [], model)]


def test_resolve():
    r = router({})
    assert r.resolve("h100/openai/gpt-oss-120b") == (PROVIDERS[1], "openai/gpt-oss-120b")
    assert r.resolve("openai/gpt-oss-120b") == (PROVIDERS[1], "openai/gpt-oss-120b")
    assert r.resolve("gpt-oss:latest") == (PROVIDERS[0], "gpt-oss:latest")
    assert r.resolve("unknown") == (PROVIDERS[0], "unknown")


@pytest.mark.anyio
async def test_fallback_when_provider_down():
    ok = FakeLLM([text_turn("hi")])
    r = router({"mac": DownLLM(), "h100": ok}, fallback=["h100/openai/gpt-oss-120b"])
    assert await drain(r, "mac/gpt-oss:latest") == [{"content": "hi"}]
    assert ok.calls[0]["model"] == "openai/gpt-oss-120b"


@pytest.mark.anyio
async def test_all_down_raises_with_details():
    r = router({"mac": DownLLM(), "h100": DownLLM()}, fallback=["h100/x"])
    with pytest.raises(LLMError, match="mac/gpt-oss:latest.*h100/x"):
        await drain(r, "mac/gpt-oss:latest")


@pytest.mark.anyio
async def test_no_fallback_after_stream_started():
    ok = FakeLLM([text_turn("never")])
    r = router({"mac": MidStreamFailLLM(), "h100": ok}, fallback=["h100/x"])
    with pytest.raises(LLMError, match="boom"):
        await drain(r, "mac/gpt-oss:latest")
    assert ok.calls == []


@pytest.mark.anyio
async def test_list_models_prefixes_and_survives_down_provider():
    r = router({"mac": FakeLLM([]), "h100": DownLLM()})
    assert await r.list_models() == ["mac/fake-model", "h100/openai/gpt-oss-120b"]


def test_load_providers_file_and_env_keys(tmp_path, monkeypatch):
    f = tmp_path / "providers.json"
    f.write_text(
        json.dumps(
            [{"name": "groq", "base_url": "https://x/v1/", "api_key": "$MY_KEY", "models": ["m"]}]
        )
    )
    monkeypatch.setenv("MY_KEY", "secret")
    [p] = load_providers(Settings(providers_file=str(f)))
    assert p == Provider("groq", "https://x/v1", "secret", ("m",))


def test_load_providers_defaults_to_llm_env(tmp_path):
    s = Settings(providers_file=str(tmp_path / "missing.json"), llm_model="m")
    assert load_providers(s) == [Provider("default", s.llm_base_url, s.llm_api_key, ("m",))]


class HeldLLM:
    """Streams one delta, then waits until released."""

    def __init__(self):
        self.release = asyncio.Event()
        self.calls = 0

    async def stream_chat(self, messages, tools, model):
        self.calls += 1
        yield {"content": "held"}
        await self.release.wait()


def capped(cap=1):
    return [Provider("mac", "http://mac/v1", models=("m",), max_concurrent=cap), PROVIDERS[1]]


@pytest.mark.anyio
async def test_spillover_when_primary_busy():
    mac, cloud = HeldLLM(), FakeLLM([text_turn("cloud"), text_turn("x")])
    r = ProviderRouter(
        capped(), ["h100/big"], client_factory=lambda p: {"mac": mac, "h100": cloud}[p.name]
    )
    first = r.stream_chat([], [], "mac/m")
    assert await first.__anext__() == {"content": "held"}
    assert r.inflight["mac"] == 1
    assert await drain(r, "mac/m") == [{"content": "cloud"}]
    mac.release.set()
    assert [d async for d in first] == []
    assert r.inflight["mac"] == 0
    assert await drain(r, "mac/m") == [{"content": "held"}]
    assert mac.calls == 2 and len(cloud.calls) == 1


@pytest.mark.anyio
async def test_busy_primary_still_used_when_fallback_fails():
    mac = HeldLLM()
    r = ProviderRouter(
        capped(), ["h100/big"], client_factory=lambda p: {"mac": mac, "h100": DownLLM()}[p.name]
    )
    first = r.stream_chat([], [], "mac/m")
    await first.__anext__()
    second = r.stream_chat([], [], "mac/m")
    assert await second.__anext__() == {"content": "held"}
    assert r.inflight == {"mac": 2, "h100": 0}
    mac.release.set()
    await first.aclose()
    await second.aclose()
    assert r.inflight["mac"] == 0


@pytest.mark.anyio
async def test_down_provider_skipped_until_cooldown(monkeypatch):
    class CountingDown(DownLLM):
        calls = 0

        async def stream_chat(self, messages, tools, model):
            CountingDown.calls += 1
            raise httpx.ConnectTimeout("timed out")
            yield  # pragma: no cover

    ok = FakeLLM([text_turn("a"), text_turn("b"), text_turn("c")])
    r = router({"mac": CountingDown(), "h100": ok}, fallback=["h100/big"])
    assert await drain(r, "mac/gpt-oss:latest") == [{"content": "a"}]
    assert await drain(r, "mac/gpt-oss:latest") == [{"content": "b"}]
    assert CountingDown.calls == 1
    monkeypatch.setattr("grok_bot.providers.time.monotonic", lambda: 1e12)
    assert await drain(r, "mac/gpt-oss:latest") == [{"content": "c"}]
    assert CountingDown.calls == 2


def test_max_concurrent_config(tmp_path, monkeypatch):
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps(
            [
                {"name": "a", "base_url": "http://a/v1"},
                {"name": "b", "base_url": "http://b/v1", "max_concurrent": 2},
            ]
        )
    )
    ps = load_providers(Settings(providers_file=str(path), max_concurrent_primary=4))
    assert [p.max_concurrent for p in ps] == [4, 2]
    [default] = load_providers(Settings(providers_file="", max_concurrent_primary=3))
    assert default.max_concurrent == 3
    monkeypatch.setenv("GROK_MAX_CONCURRENT_PRIMARY", "4")
    monkeypatch.setenv("GROK_CONNECT_TIMEOUT", "2.5")
    s = Settings.from_env()
    assert s.max_concurrent_primary == 4 and s.connect_timeout == 2.5


def test_connect_timeout_is_short():
    from grok_bot.llm import OpenAICompatClient

    c = OpenAICompatClient("http://x/v1", "k", timeout=120, connect_timeout=3)
    assert c.timeout.connect == 3 and c.timeout.read == 120


@pytest.mark.anyio
async def test_served_model_reports_fallback():
    from grok_bot.providers import served_model

    ok = FakeLLM([text_turn("hi")])
    r = router({"mac": DownLLM(), "h100": ok}, fallback=["h100/big"])
    served: dict[str, str] = {}
    served_model.set(served)
    await drain(r, "mac/gpt-oss:latest")
    assert served == {"model": "h100/big"}
