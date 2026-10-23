import pytest
from fastapi.testclient import TestClient

from grok_bot.config import Settings
from grok_bot.store import Store
from tests.test_auth import PW, make_app, signup
from tests.test_files import make_client


def admin_and_client(tmp_path, **settings):
    client = TestClient(make_app(tmp_path, **settings))
    assert signup(client, "boss").json()["user"]["is_admin"] is True
    return client


def test_signup_mode_from_env(monkeypatch):
    monkeypatch.delenv("GROK_SIGNUP", raising=False)
    monkeypatch.delenv("GROK_ALLOW_SIGNUP", raising=False)
    assert Settings.from_env().signup == "open"
    monkeypatch.setenv("GROK_ALLOW_SIGNUP", "off")
    assert Settings.from_env().signup == "closed"
    monkeypatch.setenv("GROK_SIGNUP", "Invite")
    assert Settings.from_env().signup == "invite"
    monkeypatch.setenv("GROK_SIGNUP", "maybe")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_closed_signup_still_allows_first_user(tmp_path):
    client = admin_and_client(tmp_path, signup="closed")
    assert client.get("/api/health").json()["signup"] == "closed"
    assert client.get("/api/auth/me").json()["signup"] == "closed"
    other = TestClient(client.app)
    assert signup(other, "eve").status_code == 403


def test_invite_codes(tmp_path):
    admin = admin_and_client(tmp_path, signup="invite")
    guest = TestClient(admin.app)
    assert signup(guest, "eve").status_code == 403

    inv = admin.post("/api/admin/invites", json={"uses": 1}).json()
    assert len(inv["code"]) >= 16
    listed = admin.get("/api/admin/invites").json()
    assert [i["id"] for i in listed] == [inv["id"]] and "code" not in listed[0]

    def join(name, code):
        body = {"username": name, "password": PW, "invite": code}
        return TestClient(admin.app).post("/api/auth/signup", json=body)

    assert join("eve", "wrong").status_code == 403
    assert join("boss", inv["code"]).status_code == 409  # taken name doesn't burn the code
    ok = TestClient(admin.app).post(
        "/api/auth/signup", json={"username": "eve", "password": PW, "invite_code": inv["code"]}
    )
    assert ok.status_code == 200
    assert join("mallory", inv["code"]).status_code == 403
    assert admin.get("/api/admin/invites").json() == []

    inv2 = admin.post("/api/admin/invites", json={"uses": 5, "days": 1}).json()
    assert admin.delete(f"/api/admin/invites/{inv2['id']}").json() == {"ok": True}
    assert join("trent", inv2["code"]).status_code == 403
    assert admin.delete(f"/api/admin/invites/{inv2['id']}").status_code == 404


def test_admin_routes_require_admin(tmp_path):
    admin = admin_and_client(tmp_path)
    bob = TestClient(admin.app)
    signup(bob, "bob")
    bob_id = bob.get("/api/auth/me").json()["user"]["id"]
    for method, url in [
        ("get", "/api/admin/users"),
        ("patch", f"/api/admin/users/{bob_id}"),
        ("delete", f"/api/admin/users/{bob_id}"),
        ("post", "/api/admin/invites"),
        ("get", "/api/admin/invites"),
    ]:
        kwargs = {} if method in {"get", "delete"} else {"json": {}}
        assert getattr(bob, method)(url, **kwargs).status_code == 403


def test_disable_promote_reset_delete(tmp_path):
    admin = admin_and_client(tmp_path)
    me = admin.get("/api/auth/me").json()["user"]["id"]
    bob = TestClient(admin.app)
    signup(bob, "bob")
    bob_id = bob.get("/api/auth/me").json()["user"]["id"]
    sid = Store(str(tmp_path / "t.db")).create_session(user_id=bob_id)

    assert admin.patch(f"/api/admin/users/{me}", json={"disabled": True}).status_code == 400
    assert admin.delete(f"/api/admin/users/{me}").status_code == 400
    assert admin.patch("/api/admin/users/nope", json={"disabled": True}).status_code == 404
    assert admin.patch("/api/admin/users/local", json={"disabled": True}).status_code == 404

    r = admin.patch(f"/api/admin/users/{bob_id}", json={"disabled": True}).json()
    assert r["disabled"] is True
    assert bob.get("/api/auth/me").status_code == 401  # tokens revoked
    login = {"username": "bob", "password": PW}
    assert TestClient(admin.app).post("/api/auth/login", json=login).status_code == 401
    admin.patch(f"/api/admin/users/{bob_id}", json={"disabled": False, "is_admin": True})
    assert bob.post("/api/auth/login", json=login).json()["user"]["is_admin"] is True

    url = f"/api/admin/users/{bob_id}/password"
    assert admin.post(url, json={"password": "short"}).status_code == 422
    assert admin.post(url, json={"password": "new password!"}).json() == {"ok": True}
    assert bob.get("/api/auth/me").status_code == 401
    assert bob.post("/api/auth/login", json=login).status_code == 401
    new = {"username": "bob", "password": "new password!"}
    assert bob.post("/api/auth/login", json=new).status_code == 200

    assert admin.delete(f"/api/admin/users/{bob_id}").json() == {"ok": True}
    assert bob.get("/api/auth/me").status_code == 401
    assert [u["username"] for u in admin.get("/api/admin/users").json()] == ["boss"]
    assert Store(str(tmp_path / "t.db")).get_session(sid) is None
    assert signup(TestClient(admin.app), "bob").status_code == 200


def test_think_model_routing(tmp_path, monkeypatch):
    client, llm = make_client(
        tmp_path, ["a", "b", "c", "d"], auth_enabled=False, think_model="mac/qwen3.6:35b"
    )
    client.post("/api/chat", json={"message": "hi", "think": True})
    client.post("/api/chat", json={"message": "hi", "think": False})
    client.post("/api/chat", json={"message": "hi", "think": True, "model": "mac/gpt-oss"})
    client.post("/api/chat", json={"message": "hi", "think": True, "provider": "groq"})
    assert [c["model"] for c in llm.calls] == [
        "mac/qwen3.6:35b",
        "fake",
        "mac/gpt-oss",
        "groq/fake",
    ]
    monkeypatch.setenv("GROK_THINK_MODEL", " mac/q ")
    assert Settings.from_env().think_model == "mac/q"
