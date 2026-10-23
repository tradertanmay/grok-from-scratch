import json
import sqlite3

import httpx
import pytest
from fastapi.testclient import TestClient

from grok_bot import llm as llm_mod
from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.embeddings import LLMEmbedder, SafeEmbedder, cosine, pack, unpack
from grok_bot.llm import OpenAICompatClient
from grok_bot.memory import Memory
from grok_bot.persona import system_prompt
from grok_bot.providers import Provider, ProviderRouter
from grok_bot.store import Store, fts_query
from grok_bot.tools import ToolRegistry
from tests.conftest import FakeLLM, text_turn, tool_turn
from tests.test_api import sse_events

pytestmark = pytest.mark.anyio

CONCEPTS = {
    "pizza": 0,
    "food": 0,
    "eat": 0,
    "python": 1,
    "programming": 1,
    "code": 1,
    "dog": 2,
    "pet": 2,
}


class ConceptEmbedder:
    """Maps words to a few 'concept' axes so recall can be semantic, not lexical."""

    model = "fake-embed"

    def __init__(self):
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = [0.0, 0.0, 0.0, 0.01]
            for w in t.lower().replace("?", " ").split():
                if w in CONCEPTS:
                    v[CONCEPTS[w]] += 1
            out.append(v)
        return out


class BrokenEmbedder:
    model = "broken"

    def __init__(self):
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        raise httpx.ConnectError("no embeddings here")


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "m.db"))
    for uid in ("u1", "u2"):
        s._conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, '!', 'x')",
            (uid, uid),
        )
    s._conn.commit()
    return s


def test_vector_helpers_and_fts_query():
    v = [0.5, -1.0, 2.0]
    assert unpack(pack(v)) == v
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == 0 and cosine([0, 0], [1, 1]) == 0
    assert fts_query('My "dog" is OR NEAR a dog') == '"dog" "near"'.replace(" ", " OR ", 1)
    assert fts_query("a ? !") == ""


async def test_store_crud_isolation_and_fts(store):
    mem = Memory(store)
    a = await mem.add("u1", "  Favourite   language is Python ")
    assert a["content"] == "Favourite language is Python"
    assert (await mem.add("u1", "favourite language is python"))["id"] == a["id"]
    await mem.add("u1", "Has a dog called Rex")
    await mem.add("u2", "Also likes Python")
    assert [m["content"] for m in mem.all("u1")] == [
        "Has a dog called Rex",
        "Favourite language is Python",
    ]
    hits = await mem.recall("u1", "which language do I use? python!")
    assert [h["id"] for h in hits] == [a["id"]]
    assert await mem.recall("u1", "zebra") == []
    assert not mem.delete("u2", a["id"])
    assert mem.delete("u1", a["id"]) and not mem.delete("u1", a["id"])
    assert await mem.recall("u1", "python") == []
    assert [m["content"] for m in await mem.recall("u2", "python")] == ["Also likes Python"]
    with pytest.raises(ValueError):
        await mem.add("u1", "   ")


async def test_store_without_fts5_falls_back_to_word_match(store):
    store.has_fts = False
    mem = Memory(store)
    await mem.add("u1", "Has a dog called Rex")
    await mem.add("u1", "Lives in Pune")
    assert [m["content"] for m in await mem.recall("u1", "my dog")] == ["Has a dog called Rex"]


async def test_embedding_recall_is_semantic(store):
    emb = ConceptEmbedder()
    mem = Memory(store, SafeEmbedder(emb))
    await mem.add("u1", "Loves pizza")
    await mem.add("u1", "Writes Python code all day")
    await mem.add("u1", "Has a dog")
    hits = await mem.recall("u1", "what food should I eat")
    assert [h["content"] for h in hits] == ["Loves pizza"]
    assert await mem.recall("u1", "tell me about the weather") == []


async def test_embedding_failure_falls_back_and_backs_off(store):
    now = [0.0]
    emb = BrokenEmbedder()
    mem = Memory(store, SafeEmbedder(emb, clock=lambda: now[0]))
    m = await mem.add("u1", "Has a dog called Rex")
    assert emb.calls == 1
    assert store.memory_vectors("u1", "broken") == []
    assert [h["id"] for h in await mem.recall("u1", "dog")] == [m["id"]]
    await mem.add("u1", "Lives in Pune")
    assert emb.calls == 1
    now[0] = 1000
    await mem.add("u1", "Drinks chai")
    assert emb.calls == 2


async def test_vectors_from_another_embed_model_are_ignored(store):
    old = Memory(store, SafeEmbedder(ConceptEmbedder()))
    await old.add("u1", "Loves pizza")
    new_emb = ConceptEmbedder()
    new_emb.model = "other-model"
    mem = Memory(store, SafeEmbedder(new_emb))
    assert [h["content"] for h in await mem.recall("u1", "pizza")] == ["Loves pizza"]
    assert new_emb.calls == 0


async def test_remember_and_forget_tools(store):
    reg = ToolRegistry(Memory(store).tools("u1"))
    saved = await reg.call("remember", {"fact": "Name is Alex"})
    assert not saved.is_error and "Name is Alex" in saved.output
    mid = store.list_memories("u1")[0]["id"]
    assert store.list_memories("u2") == []
    assert (await reg.call("forget", {"memory_id": "nope"})).is_error
    assert not (await reg.call("forget", {"memory_id": mid})).is_error
    assert store.list_memories("u1") == []


async def test_router_and_openai_client_embed(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        data = [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
        return httpx.Response(200, json={"data": data})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        llm_mod.httpx,
        "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    router = ProviderRouter(
        [Provider("mac", "http://mac/v1", models=("nomic-embed-text",))],
        client_factory=lambda p: OpenAICompatClient(p.base_url, p.api_key),
    )
    vecs = await LLMEmbedder(router, "mac/nomic-embed-text").embed(["a", "b"])
    assert vecs == [[1.0, 0.0], [0.0, 1.0]]
    assert seen["url"] == "http://mac/v1/embeddings"
    assert seen["body"] == {"model": "nomic-embed-text", "input": ["a", "b"]}


def test_system_prompt_memory_section():
    assert "remember" not in system_prompt("fun")
    assert "remember" in system_prompt("fun", memories=[])
    p = system_prompt("regular", memories=[{"id": "abc", "content": "Name is Alex"}])
    assert "- [abc] Name is Alex" in p


def make_app(tmp_path, turns, **settings):
    llm = FakeLLM(turns)
    app = create_app(
        Settings(llm_model="fake", **settings),
        llm=llm,
        store=Store(str(tmp_path / "t.db")),
        tools=ToolRegistry([]),
    )
    return TestClient(app), llm


def test_memory_api_is_per_user(tmp_path):
    client, _ = make_app(tmp_path, [])
    pw = "correct horse"
    client.post("/api/auth/signup", json={"username": "alice", "password": pw})
    m = client.post("/api/memories", json={"content": "Likes tea"}).json()
    assert set(m) == {"id", "content", "created_at"}
    assert client.get("/api/memories").json() == {"memories": [m]}
    assert client.post("/api/memories", json={"content": ""}).status_code == 422
    assert client.post("/api/memories", json={"content": "   "}).status_code == 422

    bob = TestClient(client.app)
    bob.post("/api/auth/signup", json={"username": "bob", "password": pw})
    assert bob.get("/api/memories").json() == {"memories": []}
    assert bob.delete(f"/api/memories/{m['id']}").status_code == 404
    assert TestClient(client.app).get("/api/memories").status_code == 401

    assert client.delete(f"/api/memories/{m['id']}").json() == {"deleted": m["id"]}
    assert client.get("/api/memories").json() == {"memories": []}


def test_memory_disabled(tmp_path):
    client, llm = make_app(tmp_path, [text_turn("hi")], auth_enabled=False, memory_enabled=False)
    assert client.get("/api/memories").status_code == 404
    client.post("/api/chat", json={"message": "hello"})
    assert "remember" not in llm.calls[0]["messages"][0]["content"]
    assert llm.calls[0]["tools"] == []


def test_chat_recalls_and_remembers(tmp_path):
    turns = [
        tool_turn("remember", json.dumps({"fact": "Name is Alex"})),
        text_turn("Nice to meet you, Alex!"),
        text_turn("You're Alex."),
        text_turn("incognito"),
    ]
    client, llm = make_app(tmp_path, turns, auth_enabled=False)
    events = sse_events(client.post("/api/chat", json={"message": "hi, I'm Alex"}))
    results = [e for e in events if e["type"] == "tool_result"]
    assert results and not results[0]["is_error"]
    assert {t["function"]["name"] for t in llm.calls[0]["tools"]} == {"remember", "forget"}
    (mem,) = client.get("/api/memories").json()["memories"]
    assert mem["content"] == "Name is Alex"

    client.post("/api/chat", json={"message": "what is my name, Alex?"})
    assert f"[{mem['id']}] Name is Alex" in llm.calls[2]["messages"][0]["content"]

    client.post("/api/chat", json={"message": "what is my name, Alex?", "memory": False})
    assert "Name is Alex" not in llm.calls[3]["messages"][0]["content"]
    assert llm.calls[3]["tools"] == []


def test_existing_db_gets_memory_tables(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL,"
        " created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()
    s = Store(str(path))
    m = s.add_memory("local", "x fact", None, None)
    assert s.list_memories("local") == [m]
