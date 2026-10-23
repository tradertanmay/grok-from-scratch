import pytest
from fastapi.testclient import TestClient

from grok_bot.agent import Agent, ThinkSplitter
from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.deepsearch import DeepSearch, parse_queries, pick_urls
from grok_bot.persona import system_prompt
from grok_bot.store import Store
from grok_bot.tools import Tool, ToolRegistry
from tests.conftest import FakeLLM, text_turn
from tests.test_api import sse_events


async def collect(runner, messages):
    return [e async for e in runner.run(messages)]


def joined(events, kind):
    return "".join(e["content"] for e in events if e["type"] == kind)


def test_think_splitter_handles_split_tags():
    s = ThinkSplitter()
    parts = []
    for chunk in ["<thi", "nk>plan ", "it</th", "ink>\n\nAns", "wer <", "b>"]:
        parts += s.feed(chunk)
    parts += s.flush()
    assert "".join(t for k, t in parts if k == "think") == "plan it"
    assert "".join(t for k, t in parts if k == "text") == "\n\nAnswer <b>"


@pytest.mark.anyio
@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
async def test_native_reasoning_forwarded_when_think_on(field):
    llm = FakeLLM([[{field: "hmm "}, {field: "ok"}, {"content": "42"}]])
    agent = Agent(llm, ToolRegistry([]), "m", think=True)
    msgs = [{"role": "user", "content": "q"}]
    events = await collect(agent, msgs)
    assert joined(events, "think") == "hmm ok"
    assert joined(events, "token") == "42"
    assert msgs[-1] == {"role": "assistant", "content": "42", "reasoning": "hmm ok"}


@pytest.mark.anyio
async def test_think_tags_hidden_and_dropped_when_think_off():
    llm = FakeLLM([text_turn("<think>secret", "</think>\n\nHi")])
    agent = Agent(llm, ToolRegistry([]), "m")
    msgs = [{"role": "user", "content": "q"}]
    events = await collect(agent, msgs)
    assert [e["type"] for e in events if e["type"] != "token"] == []
    assert "secret" not in joined(events, "token")
    assert msgs[-1] == {"role": "assistant", "content": "Hi"}


def test_persona_modes():
    assert "humor" in system_prompt("fun")
    regular = system_prompt("regular")
    assert "No jokes" in regular and "humor" not in regular
    assert "web_search" in regular


def test_parse_queries_json_lines_and_fallback():
    assert parse_queries('Sure: ["a", "b", "c"]', "q", 2) == ["a", "b"]
    assert parse_queries("1. first\n- second\n", "q", 4) == ["first", "second"]
    assert parse_queries("", "q", 4) == ["q"]


def test_pick_urls_round_robin_dedup():
    a = "[1] A\nhttps://a.com\nx\n\n[2] B\nhttps://b.com\ny"
    b = "[1] C\nhttps://a.com\nz\n\n[2] D\nhttps://d.com\nw"
    assert pick_urls([a, b], 3) == ["https://a.com", "https://b.com", "https://d.com"]


def research_tools(fetched: list[str]) -> ToolRegistry:
    def web_search(query: str) -> str:
        return f"[1] {query}\nhttps://{query.replace(' ', '-')}.com\nsnippet"

    def fetch_url(url: str) -> str:
        fetched.append(url)
        return f"URL: {url}\n\ncontent of {url}"

    obj = {"type": "object"}
    return ToolRegistry(
        [Tool("web_search", "s", obj, web_search), Tool("fetch_url", "f", obj, fetch_url)]
    )


@pytest.mark.anyio
async def test_deepsearch_pipeline():
    llm = FakeLLM([text_turn('["alpha q", "beta q"]'), text_turn("Answer [1][2]")])
    fetched: list[str] = []
    ds = DeepSearch(llm, research_tools(fetched), "m")
    msgs = [{"role": "user", "content": "compare alpha and beta"}]
    events = await collect(ds, msgs)

    statuses = [e for e in events if e["type"] == "status"]
    assert [e["step"] for e in statuses] == ["plan", "search", "read", "answer"]
    assert statuses[0]["detail"] == statuses[0]["content"] == "Planning searches"
    calls = [(e["name"], e["arguments"]) for e in events if e["type"] == "tool_call"]
    assert calls[:2] == [("web_search", {"query": "alpha q"}), ("web_search", {"query": "beta q"})]
    assert sorted(fetched) == ["https://alpha-q.com", "https://beta-q.com"]
    assert joined(events, "token") == "Answer [1][2]"

    final_prompt = llm.calls[1]["messages"][-1]["content"]
    assert (
        "[1] https://alpha-q.com" in final_prompt
        and "content of https://beta-q.com" in final_prompt
    )
    assert llm.calls[1]["tools"] == []
    roles = [m["role"] for m in ds.new_messages]
    assert roles == ["assistant", "tool", "tool", "assistant", "tool", "tool", "assistant"]
    assert ds.new_messages[-1]["content"] == "Answer [1][2]"


def test_chat_api_modes_and_deepsearch(tmp_path):
    llm = FakeLLM(
        [
            [{"reasoning": "thinking"}, {"content": "plain"}],
            text_turn('["q1"]'),
            text_turn("cited [1]"),
        ]
    )
    app = create_app(
        Settings(llm_model="fake", auth_enabled=False),
        llm=llm,
        store=Store(str(tmp_path / "t.db")),
        tools=research_tools([]),
    )
    client = TestClient(app)

    r = client.post("/api/chat", json={"message": "hi", "persona": "regular"})
    events = sse_events(r)
    assert joined(events, "think") == "thinking"
    assert "No jokes" in llm.calls[0]["messages"][0]["content"]
    sid = events[0]["session_id"]
    saved = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert saved[-1]["reasoning"] == "thinking"
    assert "reasoning" not in llm.calls[0]["messages"][-1]

    r = client.post(
        "/api/chat",
        json={"session_id": sid, "message": "research", "deepsearch": True, "think": False},
    )
    events = sse_events(r)
    assert events[-1]["type"] == "done"
    assert joined(events, "token") == "cited [1]"
    assert "reasoning" not in str(llm.calls[1]["messages"])

    assert client.post("/api/chat", json={"message": "x", "mode": "evil"}).status_code == 422
