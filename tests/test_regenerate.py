import json

import pytest
from fastapi.testclient import TestClient

from grok_bot.app import ChatRequest, create_app
from grok_bot.config import Settings
from grok_bot.store import Store
from grok_bot.tools import ToolRegistry
from tests.conftest import FakeLLM, text_turn, tool_turn
from tests.test_api import sse_events
from tests.test_auth import signup


def make(tmp_path, turns, **settings):
    llm = FakeLLM(turns)
    store = Store(str(tmp_path / "t.db"))
    app = create_app(
        Settings(llm_model="fake", embed_model="", **settings),
        llm=llm,
        store=store,
        tools=ToolRegistry([]),
    )
    return app, llm, store


def roles(store, sid):
    return [(m["role"], m["content"]) for m in store.get_messages(sid)]


def test_regenerate_replaces_last_answer(tmp_path):
    app, llm, store = make(
        tmp_path, [text_turn("one"), text_turn("two"), text_turn("three")], auth_enabled=False
    )
    client = TestClient(app)
    sid = sse_events(client.post("/api/chat", json={"message": "hi"}))[0]["session_id"]
    events = sse_events(client.post(f"/api/sessions/{sid}/regenerate"))
    assert [e["type"] for e in events] == ["session", "token", "done"]
    assert events[1]["content"] == "two"
    assert roles(store, sid) == [("user", "hi"), ("assistant", "two")]
    assert [m["role"] for m in llm.calls[1]["messages"]] == ["system", "user"]

    body = {"model": "mac/other", "mode": "regular", "think": False}
    sse_events(client.post(f"/api/sessions/{sid}/regenerate", json=body))
    assert llm.calls[2]["model"] == "mac/other"
    assert roles(store, sid) == [("user", "hi"), ("assistant", "three")]


def test_regenerate_checks_owner_and_empty(tmp_path):
    app, _, store = make(tmp_path, [text_turn("a")])
    alice, bob = TestClient(app), TestClient(app)
    signup(alice, "alice")
    signup(bob, "bob")
    sid = sse_events(alice.post("/api/chat", json={"message": "hi"}))[0]["session_id"]
    assert bob.post(f"/api/sessions/{sid}/regenerate").status_code == 404
    empty = store.create_session(user_id=alice.get("/api/auth/me").json()["user"]["id"])
    assert alice.post(f"/api/sessions/{empty}/regenerate").status_code == 409


def chat_endpoint(app):
    return next(r.endpoint for r in app.routes if getattr(r, "path", "") == "/api/chat")


async def stop_after(app, store, message, n_events):
    resp = await chat_endpoint(app)(ChatRequest(message=message), store.get_user("local"))
    it = resp.body_iterator
    events = [json.loads((await it.__anext__())[5:]) for _ in range(n_events)]
    await it.aclose()  # what Starlette does when the client disconnects
    return events


@pytest.mark.anyio
async def test_stop_keeps_partial_answer(tmp_path):
    app, _, store = make(tmp_path, [text_turn("Hello", " wor", "ld")], auth_enabled=False)
    events = await stop_after(app, store, "hi", 3)
    sid = events[0]["session_id"]
    assert [e["type"] for e in events] == ["session", "token", "token"]
    assert roles(store, sid) == [("user", "hi"), ("assistant", "Hello wor")]
    [row] = store.usage_summary("2000-01-01")
    assert row["errors"] == 0 and row["requests"] == 1


@pytest.mark.anyio
async def test_stop_during_tool_call_keeps_history_valid(tmp_path):
    app, llm, store = make(
        tmp_path, [tool_turn("missing_tool", '{"x": 1}'), text_turn("ok")], auth_enabled=False
    )
    events = await stop_after(app, store, "do it", 2)
    sid = events[0]["session_id"]
    assert events[1]["type"] == "tool_call"
    msgs = store.get_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[2]["tool_call_id"] == msgs[1]["tool_calls"][0]["id"]
    assert msgs[2]["content"] == "(stopped by the user)"

    client = TestClient(app)
    sse_events(client.post("/api/chat", json={"message": "again", "session_id": sid}))
    sent = [m["role"] for m in llm.calls[-1]["messages"]]
    assert sent == ["system", "user", "assistant", "tool", "user"]
