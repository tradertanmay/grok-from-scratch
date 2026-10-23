import json

from fastapi.testclient import TestClient

from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.openai_compat import resolve_model, split_messages
from grok_bot.store import Store
from grok_bot.tools import ToolRegistry
from tests.conftest import FakeLLM, text_turn
from tests.test_auth import signup


def setup(tmp_path, turns=(), **settings):
    llm = FakeLLM([text_turn(*t) if isinstance(t, tuple) else text_turn(t) for t in turns])
    store = Store(str(tmp_path / "t.db"))
    app = create_app(
        Settings(llm_model="fake", **settings), llm=llm, store=store, tools=ToolRegistry([])
    )
    signup(TestClient(app), "boss")  # first user is admin (quota-exempt)
    browser = TestClient(app)
    signup(browser, "ann")
    key = browser.post("/api/keys", json={"name": "cursor"}).json()
    return app, llm, store, browser, key


def api(app, key):
    return TestClient(app, headers={"Authorization": f"Bearer {key['key']}"})


def test_key_lifecycle(tmp_path):
    app, _, _, browser, key = setup(tmp_path)
    assert key["key"].startswith("gk_") and key["prefix"] == key["key"][:10]
    [listed] = browser.get("/api/keys").json()
    assert listed["name"] == "cursor" and "key" not in listed and "key_hash" not in listed
    client = api(app, key)
    assert client.get("/api/auth/me").json()["user"]["username"] == "ann"
    assert client.get("/api/keys").status_code == 403
    assert client.post("/api/keys").status_code == 403
    assert browser.get("/api/keys").json()[0]["last_used_at"]
    bad = TestClient(app, headers={"Authorization": "Bearer gk_nope"})
    assert bad.get("/v1/models").status_code == 401
    assert browser.delete(f"/api/keys/{key['id']}").json() == {"ok": True}
    assert client.get("/api/auth/me").status_code == 401
    assert browser.delete(f"/api/keys/{key['id']}").status_code == 404


def test_disabled_user_key_rejected(tmp_path):
    app, _, store, _, key = setup(tmp_path)
    uid = next(u["id"] for u in store.list_users() if u["username"] == "ann")
    store.update_user(uid, disabled=True)
    assert api(app, key).get("/v1/models").status_code == 401


def test_completion_non_stream(tmp_path):
    app, llm, store, _, key = setup(tmp_path, turns=[("Hel", "lo!")])
    r = api(app, key).post(
        "/v1/chat/completions",
        json={
            "model": "grok-regular",
            "messages": [
                {"role": "system", "content": "Answer in French."},
                {"role": "user", "content": "earlier q"},
                {"role": "assistant", "content": "earlier a"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
            "temperature": 0.2,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion" and body["model"] == "grok-regular"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello!"}
    assert body["usage"]["total_tokens"] > 0
    sent = llm.calls[0]["messages"]
    assert "Answer in French." in sent[0]["content"]
    assert [m["role"] for m in sent[1:]] == ["user", "assistant", "user"]
    sid = r.headers["x-grok-session"]
    roles = [m["role"] for m in store.get_messages(sid)]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_completion_stream(tmp_path):
    app, _, _, _, key = setup(tmp_path, turns=[("a", "b", "c")])
    r = api(app, key).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    lines = [ln[6:] for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(x) for x in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "abc"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] == 1


def test_completion_errors_and_quota(tmp_path):
    app, _, _, _, key = setup(tmp_path, turns=["one"], daily_messages=1)
    client = api(app, key)
    msg = {"messages": [{"role": "user", "content": "hi"}]}
    bad = {"messages": [{"role": "assistant", "content": "hi"}]}
    assert client.post("/v1/chat/completions", json=bad).status_code == 400
    assert client.post("/v1/chat/completions", json=msg).status_code == 200
    assert client.post("/v1/chat/completions", json=msg).status_code == 429


def test_models_and_auth_off(tmp_path):
    llm = FakeLLM([text_turn("ok")])
    app = create_app(
        Settings(llm_model="fake", auth_enabled=False),
        llm=llm,
        store=Store(str(tmp_path / "t.db")),
        tools=ToolRegistry([]),
    )
    client = TestClient(app)
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids[:2] == ["grok", "grok-regular"] and "fake" in ids
    r = client.post(
        "/v1/chat/completions",
        json={"model": "fake", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.json()["choices"][0]["message"]["content"] == "ok"
    assert llm.calls[0]["model"] == "fake"


def test_helpers():
    assert resolve_model("grok-think")["think"] is True
    assert resolve_model("mac/qwen")["model"] == "mac/qwen"
    sys, hist, text, imgs = split_messages(
        [
            {"role": "developer", "content": "be brief"},
            {"role": "tool", "content": "ignored"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                ],
            },
        ]
    )
    assert (sys, hist, text, imgs) == ("be brief", [], "what is this", ["data:image/png;base64,AA"])
