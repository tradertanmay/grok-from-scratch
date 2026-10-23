"""Tests for the concurrent load-test harness (grok_bot.loadtest)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.llm import LLMError
from grok_bot.loadtest import format_summary, load_keys_file, run_load_test
from grok_bot.store import Store
from grok_bot.tools import default_registry
from tests.conftest import FakeLLM, text_turn


class AlwaysFailingLLM:
    async def stream_chat(self, messages, tools, model):
        raise LLMError("upstream exploded")
        yield  # pragma: no cover


def test_loadtest_20_users_with_default_auth_rate_limit(tmp_path, monkeypatch):
    """Verifies 20 concurrent users succeed & clean up even with default auth_rate_limit=10."""
    from grok_bot.auth import RateLimiter

    clock = [1000.0]
    monkeypatch.setattr(RateLimiter.__init__, "__defaults__", (60.0, lambda: clock[0]))

    async def fake_sleep(seconds: float) -> None:
        clock[0] += seconds + 0.1

    llm = FakeLLM([text_turn(f"Reply {i}") for i in range(25)])
    # Use default auth_rate_limit=10!
    settings = Settings(
        llm_model="fake-model",
        db_path=str(tmp_path / "loadtest.db"),
        chat_rate_limit=30,
        auth_rate_limit=10,
    )
    store = Store(settings.db_path)
    app = create_app(settings=settings, store=store, llm=llm, tools=default_registry())

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/auth/signup",
                json={"username": "admin_owner", "password": "password123"},
            )
            return await run_load_test(
                users=20,
                requests_per_user=1,
                model="grok-regular",
                client=client,
                cleanup=True,
                sleep_fn=fake_sleep,
            )

    summary = asyncio.run(_run())
    assert summary["users"] == 20
    assert summary["succeeded"] == 20
    assert summary["failed"] == 0
    assert summary["undeleted_users"] == []
    text = format_summary(summary)
    assert "CONCURRENT_USERS: 20 | REQUESTS: 20/20 (failed=0, empty=0)" in text
    assert len(store.list_users()) == 1


def test_loadtest_midstream_errors_counted_as_failed(tmp_path):
    """Verifies mid-stream `data: {"error": ...}` chunks are counted as failed, not succeeded."""
    settings = Settings(
        llm_model="fake-model",
        db_path=str(tmp_path / "err.db"),
        auth_rate_limit=0,
    )
    store = Store(settings.db_path)
    app = create_app(
        settings=settings,
        store=store,
        llm=AlwaysFailingLLM(),
        tools=default_registry(),
    )

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/auth/signup",
                json={"username": "admin_owner", "password": "password123"},
            )
            return await run_load_test(
                users=3,
                model="grok-regular",
                client=client,
                cleanup=True,
            )

    summary = asyncio.run(_run())
    assert summary["succeeded"] == 0
    assert summary["failed"] == 3
    assert "upstream exploded" in summary["errors"][0]["error"]
    assert len(store.list_users()) == 1


def test_loadtest_aborts_on_empty_db_when_first_user_becomes_admin(tmp_path):
    """Verifies run_load_test aborts immediately if run against an empty DB with no admin."""
    settings = Settings(
        llm_model="fake-model",
        db_path=str(tmp_path / "empty.db"),
        auth_rate_limit=0,
    )
    store = Store(settings.db_path)
    app = create_app(
        settings=settings,
        store=store,
        llm=FakeLLM([text_turn("hi")]),
        tools=default_registry(),
    )

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_load_test(users=3, client=client)

    with pytest.raises(RuntimeError, match="sole admin"):
        asyncio.run(_run())


def test_loadtest_keys_file_works_when_signup_closed(tmp_path):
    """Verifies --keys-file works on GROK_SIGNUP=closed without signing up or deleting accounts."""
    llm = FakeLLM([text_turn("Key 1 reply"), text_turn("Key 2 reply")])
    settings = Settings(
        llm_model="fake-model",
        db_path=str(tmp_path / "closed.db"),
        signup="closed",
        admin_username="boss",
        admin_password="password123",
    )
    store = Store(settings.db_path)
    app = create_app(settings=settings, store=store, llm=llm, tools=default_registry())

    boss = store.list_users()[0]
    k1 = store.create_api_key(boss["id"], "k1")["key"]
    k2 = store.create_api_key(boss["id"], "k2")["key"]
    keys_path = tmp_path / "keys.txt"
    keys_path.write_text(f"# test keys\n{k1}\n{k2}\n", encoding="utf-8")
    assert load_keys_file(keys_path) == [k1, k2]

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await run_load_test(
                users=2,
                keys_file=keys_path,
                client=client,
            )

    summary = asyncio.run(_run())
    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    assert len(store.list_users()) == 1
