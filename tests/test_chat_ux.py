import pytest
from fastapi.testclient import TestClient

from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.providers import Provider, ProviderRouter
from grok_bot.store import Store
from grok_bot.tools import ToolRegistry
from tests.conftest import FakeLLM, text_turn
from tests.test_api import make_client, sse_events


def chat(client, message, session_id=None):
    events = sse_events(
        client.post("/api/chat", json={"message": message, "session_id": session_id})
    )
    return events[0]["session_id"]


def test_rename_search_truncate(tmp_path):
    client, llm = make_client(
        [text_turn("Paris."), text_turn("Rome."), text_turn("Bern.")], tmp_path
    )
    a = chat(client, "capital of france")
    chat(client, "and italy", a)
    b = chat(client, "swiss capital_50%")

    assert (
        client.patch(f"/api/sessions/{a}", json={"title": "Capitals"}).json()["title"] == "Capitals"
    )
    assert client.patch(f"/api/sessions/{a}", json={"title": ""}).status_code == 422
    assert client.patch("/api/sessions/nope", json={"title": "x"}).status_code == 404

    ids = lambda q: [s["id"] for s in client.get("/api/sessions", params={"q": q}).json()]  # noqa: E731
    assert ids("capitals") == [a]
    assert ids("rome") == [a]
    assert ids("_50%") == [b]
    assert ids("50_") == []
    assert len(client.get("/api/sessions").json()) == 2

    assert client.post(f"/api/sessions/{a}/truncate", json={"from_index": 2}).json() == {
        "deleted": 2
    }
    msgs = client.get(f"/api/sessions/{a}").json()["messages"]
    assert [m["content"] for m in msgs] == ["capital of france", "Paris."]
    assert client.post(f"/api/sessions/{a}/truncate", json={"from_index": -1}).status_code == 422


def test_share_is_stable_and_read_only(tmp_path):
    client, _ = make_client([text_turn("hi there")], tmp_path)
    sid = chat(client, "hello")
    share = client.post(f"/api/sessions/{sid}/share").json()
    assert share["url"] == f"/?share={share['share_id']}"
    assert client.post(f"/api/sessions/{sid}/share").json() == share
    shared = client.get(f"/api/shared/{share['share_id']}").json()
    assert shared["title"] == "hello" and "id" not in shared
    assert [m["content"] for m in shared["messages"]] == ["hello", "hi there"]
    assert client.get("/api/shared/bogus").status_code == 404
    client.delete(f"/api/sessions/{sid}")
    assert client.get(f"/api/shared/{share['share_id']}").status_code == 404


class Listing(FakeLLM):
    def __init__(self, models, fail=False):
        super().__init__([])
        self.models, self.fail = models, fail

    async def list_models(self):
        if self.fail:
            raise OSError("down")
        return self.models


@pytest.mark.anyio
async def test_router_lists_providers_with_health():
    clients = {"mac": Listing(["gpt-oss:latest"]), "h100": Listing([], fail=True)}
    router = ProviderRouter(
        [Provider("mac", "http://m"), Provider("h100", "http://h", models=("big",))],
        client_factory=lambda p: clients[p.name],
    )
    assert await router.list_providers() == [
        {"name": "mac", "models": ["mac/gpt-oss:latest"], "healthy": True},
        {"name": "h100", "models": ["h100/big"], "healthy": False},
    ]
    assert await router.list_models() == ["mac/gpt-oss:latest", "h100/big"]


def test_models_endpoint_groups_and_chat_provider_field(tmp_path):
    mac = Listing(["gpt-oss:latest"])
    mac.turns = [text_turn("ok")]
    router = ProviderRouter([Provider("mac", "http://m")], client_factory=lambda p: mac)
    app = create_app(
        Settings(llm_model="gpt-oss:latest", auth_enabled=False),
        llm=router,
        store=Store(str(tmp_path / "t.db")),
        tools=ToolRegistry([]),
    )
    client = TestClient(app)
    body = client.get("/api/models").json()
    assert body["providers"][0]["healthy"] is True
    assert body["models"] == ["mac/gpt-oss:latest"]
    client.post("/api/chat", json={"message": "x", "provider": "mac"})
    assert mac.calls[0]["model"] == "gpt-oss:latest"
