import json

import httpx
import pytest

from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.evals import Case, check_answer, load_cases, run_suite
from grok_bot.evals.__main__ import summary
from grok_bot.evals.runner import Turn
from grok_bot.store import Store
from grok_bot.tools import ToolRegistry
from tests.conftest import FakeLLM, text_turn, tool_turn

pytestmark = pytest.mark.anyio


def test_bundled_cases_load_and_have_known_checks():
    known = {"contains_any", "contains_all", "not_contains", "regex", "not_regex", "refuses",
             "tool_called", "tool_not_called", "no_tools", "status_events", "min_chars",
             "max_chars", "max_seconds"}  # fmt: skip
    cases = load_cases()
    assert len(cases) >= 12 and len({c.id for c in cases}) == len(cases)
    for c in cases:
        assert set(c.checks) <= known, c.id
        assert c.steps and all("message" in s for s in c.steps)


def test_check_answer():
    turn = Turn(answer="It's Canberra.", tools=["web_search"], seconds=3)
    assert check_answer(turn, {"contains_any": ["canberra"], "max_seconds": 5}) == []
    bad = check_answer(turn, {"no_tools": True, "regex": r"^\d+$", "max_chars": 3})
    assert len(bad) == 3
    assert check_answer(Turn(answer="Sorry, I can't help with that."), {"refuses": True}) == []
    assert check_answer(Turn(answer="Sure! Step 1..."), {"refuses": True}) == ["did not refuse"]
    assert check_answer(Turn(answer="", error="boom"), {}) == ["error event: boom", "empty answer"]


def asgi_client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.parametrize("auth", [True, False])
async def test_run_suite_against_app(tmp_path, auth):
    llm = FakeLLM(
        [
            text_turn("Noted, Zorblax."),
            text_turn("Zorblax"),
            tool_turn("echo", '{"text": "hi"}'),
            text_turn("done"),
            text_turn("Paris"),
        ]
    )
    from grok_bot.tools.base import Tool

    echo = Tool("echo", "echo", {"type": "object", "properties": {}}, lambda text="": text)
    store = Store(str(tmp_path / "e.db"))
    settings = Settings(llm_model="fake", embed_model="", auth_enabled=auth)
    app = create_app(settings, llm=llm, store=store, tools=ToolRegistry([echo]))
    cases = [
        Case(
            "ctx",
            "context",
            [{"message": "codename Zorblax"}, {"message": "codename?"}],
            {"contains_any": ["zorblax"]},
        ),  # fmt: skip
        Case("tool", "tools", [{"message": "echo"}], {"tool_called": "echo"}),
        Case(
            "wrong",
            "persona",
            [{"message": "capital of Australia"}],
            {"contains_any": ["Canberra"]},
        ),  # fmt: skip
        Case("rag", "files", [{"message": "q", "upload": {"name": "a.txt", "content": "x"}}], {}),
    ]
    async with asgi_client(app) as client:
        results = await run_suite(client, cases, model="mac/m")
        assert [r.status for r in results][:3] == ["pass", "pass", "fail"]
        assert results[3].status in {"skip", "fail", "pass"}
        assert all(c["model"] == "mac/m" for c in llm.calls)
        assert llm.calls[1]["messages"][-2]["content"] == "Noted, Zorblax."
        if auth:
            assert client.cookies.get("grok_session")
        assert (await client.get("/api/sessions")).json() == []
    assert "3/4 passed" in summary(results)


async def test_upload_skipped_when_server_has_no_files_route():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok", "auth_enabled": False})
        if request.url.path == "/api/memories":
            return httpx.Response(200, json={"memories": []})
        return httpx.Response(404, json={"detail": "Not Found"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://t")
    case = Case("rag", "files", [{"message": "q", "upload": {"name": "a.txt", "content": "x"}}], {})
    async with client:
        [result] = await run_suite(client, [case])
    assert result.status == "skip"


async def test_cli_writes_json(tmp_path, monkeypatch):
    from grok_bot.evals import __main__ as cli

    async def fake_suite(client, cases, model, think, on_result=None):
        from grok_bot.evals import Result

        return [Result(c.id, c.category, "pass", []) for c in cases]

    monkeypatch.setattr(cli, "run_suite", fake_suite)
    out = tmp_path / "r.json"
    code = await cli.main(["--only", "persona", "--json", str(out), "--min-pass", "1"])
    assert code == 0
    assert {r["category"] for r in json.loads(out.read_text())} == {"persona"}
