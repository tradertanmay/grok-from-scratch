import json

from fastapi.testclient import TestClient

from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.store import Store
from grok_bot.tools import Tool, ToolRegistry
from tests.conftest import FakeLLM, text_turn, tool_turn


def make_client(turns, tmp_path):
    llm = FakeLLM(turns)
    tools = ToolRegistry([Tool("add", "Add", {"type": "object"}, lambda a, b: str(a + b))])
    app = create_app(
        Settings(llm_model="fake", auth_enabled=False),
        llm=llm,
        store=Store(str(tmp_path / "t.db")),
        tools=tools,
    )
    return TestClient(app), llm


def sse_events(resp):
    return [json.loads(line[5:]) for line in resp.text.splitlines() if line.startswith("data:")]


def test_health_and_tools(tmp_path):
    client, _ = make_client([], tmp_path)
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/tools").json()[0]["name"] == "add"


def test_chat_stream_and_session_persistence(tmp_path):
    client, llm = make_client(
        [tool_turn("add", '{"a": 2, "b": 3}'), text_turn("It's ", "5."), text_turn("Again!")],
        tmp_path,
    )
    resp = client.post("/api/chat", json={"message": "what is 2+3?"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp)
    assert [e["type"] for e in events] == [
        "session",
        "tool_call",
        "tool_result",
        "token",
        "token",
        "done",
    ]
    sid = events[0]["session_id"]
    assert events[2]["output"] == "5"

    sessions = client.get("/api/sessions").json()
    assert sessions[0]["id"] == sid and sessions[0]["title"] == "what is 2+3?"
    msgs = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[-1]["content"] == "It's 5."

    client.post("/api/chat", json={"session_id": sid, "message": "again"})
    sent = llm.calls[-1]["messages"]
    assert sent[0]["role"] == "system"
    assert [m["role"] for m in sent[1:]] == ["user", "assistant", "tool", "assistant", "user"]
    assert all("created_at" not in m for m in sent)

    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_chat_unknown_session_404(tmp_path):
    client, _ = make_client([], tmp_path)
    assert client.post("/api/chat", json={"session_id": "x", "message": "hi"}).status_code == 404


def test_llm_error_becomes_error_event(tmp_path):
    class Boom(FakeLLM):
        async def stream_chat(self, messages, tools, model):
            raise RuntimeError("model down")
            yield  # pragma: no cover

    app = create_app(
        Settings(auth_enabled=False),
        llm=Boom([]),
        store=Store(str(tmp_path / "t.db")),
        tools=ToolRegistry(),
    )
    events = sse_events(TestClient(app).post("/api/chat", json={"message": "hi"}))
    assert events[-1] == {"type": "error", "message": "RuntimeError: model down"}
