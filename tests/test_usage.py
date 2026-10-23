from fastapi.testclient import TestClient

from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.llm import LLMError
from grok_bot.store import Store
from grok_bot.tools import ToolRegistry
from tests.conftest import FakeLLM, text_turn
from tests.test_auth import signup


class BrokenLLM(FakeLLM):
    async def stream_chat(self, messages, tools, model):
        raise LLMError("provider down")
        yield {}


def make(tmp_path, n_turns=10, broken=False, **settings):
    llm = (BrokenLLM if broken else FakeLLM)([text_turn("hello there") for _ in range(n_turns)])
    app = create_app(
        Settings(llm_model="fake", embed_model="", **settings),
        llm=llm,
        store=Store(str(tmp_path / "t.db")),
        tools=ToolRegistry([]),
    )
    admin = TestClient(app)
    signup(admin, "boss")
    bob = TestClient(app)
    signup(bob, "bob")
    return admin, bob


def chat(client, text="hi"):
    return client.post("/api/chat", json={"message": text, "think": False})


def test_daily_quota_blocks_users_not_admins(tmp_path):
    admin, bob = make(tmp_path, daily_messages=2)
    assert chat(bob).status_code == 200 and chat(bob).status_code == 200
    r = chat(bob)
    assert r.status_code == 429 and "daily limit of 2" in r.json()["detail"]
    me = bob.get("/api/usage").json()
    assert me["today"] == {"messages": 2, "limit": 2, "remaining": 0}
    for _ in range(3):
        assert chat(admin).status_code == 200
    assert admin.get("/api/usage").json()["today"]["limit"] is None


def test_usage_accounting(tmp_path):
    admin, bob = make(tmp_path)
    chat(bob, "what's up")
    chat(admin)
    mine = bob.get("/api/usage").json()
    assert mine["today"] == {"messages": 1, "limit": None, "remaining": None}
    [row] = mine["days"]
    assert row["username"] == "bob" and row["model"] == "fake" and row["requests"] == 1
    assert row["completion_chars"] == len("hello there") and row["errors"] == 0
    assert row["prompt_chars"] > len("what's up") and row["est_tokens"] > 0

    assert bob.get("/api/admin/usage").status_code == 403
    report = admin.get("/api/admin/usage?days=1").json()
    assert report["users"]["bob"]["requests"] == 1 and report["users"]["boss"]["requests"] == 1
    assert report["models"]["fake"]["requests"] == 2
    assert len(report["rows"]) == 2 and report["daily_limit"] is None
    assert admin.get("/api/admin/usage?days=0").status_code == 422


def test_failed_turn_is_counted_as_error(tmp_path):
    admin, bob = make(tmp_path, broken=True)
    body = chat(bob).text
    assert '"type": "error"' in body
    [row] = bob.get("/api/usage").json()["days"]
    assert row["requests"] == 1 and row["errors"] == 1


def test_usage_records_the_provider_that_answered(tmp_path):
    from grok_bot.providers import Provider, ProviderRouter
    from tests.test_providers import DownLLM

    cloud = FakeLLM([text_turn("from cloud")])
    router = ProviderRouter(
        [Provider("mac", "http://mac/v1"), Provider("cloud", "http://cloud/v1")],
        ["cloud/big"],
        client_factory=lambda p: {"mac": DownLLM(), "cloud": cloud}[p.name],
    )
    app = create_app(
        Settings(llm_model="mac/gpt-oss", embed_model="", metrics_token="t"),
        llm=router,
        store=Store(str(tmp_path / "t.db")),
        tools=ToolRegistry([]),
    )
    c = TestClient(app)
    signup(c, "boss")
    assert "from cloud" in chat(c).text
    [row] = c.get("/api/usage").json()["days"]
    assert row["model"] == "cloud/big"
    assert (
        'grok_chat_turns_total{model="cloud/big",outcome="ok"} 1'
        in c.get("/metrics", headers={"Authorization": "Bearer t"}).text
    )
