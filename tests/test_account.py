import json

from fastapi.testclient import TestClient

from tests.test_api import sse_events
from tests.test_auth import PW, make_app, signup


def setup(tmp_path):
    app = make_app(tmp_path, turns=["Hello Ann"])
    boss, ann = TestClient(app), TestClient(app)
    signup(boss, "boss")
    signup(ann, "ann")
    sse_events(ann.post("/api/chat", json={"message": "hi, I'm Ann", "think": False}))
    ann.post("/api/memories", json={"content": "Ann likes tea"})
    return boss, ann


def test_export_json_and_markdown(tmp_path):
    boss, ann = setup(tmp_path)
    r = ann.get("/api/account/export")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    data = json.loads(r.text)
    assert data["user"]["username"] == "ann"
    [sess] = data["sessions"]
    assert [m["content"] for m in sess["messages"]] == ["hi, I'm Ann", "Hello Ann"]
    assert [m["content"] for m in data["memories"]] == ["Ann likes tea"]
    assert data["usage"][0]["requests"] == 1
    assert boss.get("/api/account/export").json()["sessions"] == []

    md = ann.get("/api/account/export?format=md")
    assert md.headers["content-type"].startswith("text/markdown")
    assert "**You:** hi, I'm Ann" in md.text and "**Grok:** Hello Ann" in md.text
    assert "- Ann likes tea" in md.text
    assert ann.get("/api/account/export?format=xml").status_code == 422


def test_delete_account(tmp_path):
    boss, ann = setup(tmp_path)
    assert ann.request("DELETE", "/api/account", json={"password": "nope"}).status_code == 403
    r = ann.request("DELETE", "/api/account", json={"password": PW})
    assert r.json() == {"ok": True}
    assert ann.get("/api/auth/me").status_code == 401
    login = {"username": "ann", "password": PW}
    assert ann.post("/api/auth/login", json=login).status_code == 401
    assert [u["username"] for u in boss.get("/api/admin/users").json()] == ["boss"]

    r = boss.request("DELETE", "/api/account", json={"password": PW})
    assert r.status_code == 409


def test_delete_account_needs_auth_on(tmp_path):
    client = TestClient(make_app(tmp_path, auth_enabled=False))
    assert client.request("DELETE", "/api/account", json={"password": PW}).status_code == 400
    assert client.get("/api/account/export").json()["user"]["id"] == "local"
