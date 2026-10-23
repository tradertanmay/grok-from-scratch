import sqlite3

from fastapi.testclient import TestClient

from grok_bot.app import create_app
from grok_bot.auth import RateLimiter, hash_password, verify_password
from grok_bot.config import Settings
from grok_bot.store import Store
from grok_bot.tools import ToolRegistry
from tests.conftest import FakeLLM, text_turn
from tests.test_api import sse_events

PW = "correct horse"


def make_app(tmp_path, turns=(), **settings):
    llm = FakeLLM([text_turn(t) for t in turns])
    return create_app(
        Settings(llm_model="fake", **settings),
        llm=llm,
        store=Store(str(tmp_path / "t.db")),
        tools=ToolRegistry([]),
    )


def signup(client, username, password=PW):
    return client.post("/api/auth/signup", json={"username": username, "password": password})


def test_password_hashing():
    h = hash_password(PW)
    assert h.startswith("scrypt$") and PW not in h
    assert verify_password(PW, h)
    assert not verify_password("wrong", h)
    assert not verify_password(PW, "!") and not verify_password(PW, "md5$1$2$3$aa$bb")
    assert hash_password(PW) != h


def test_rate_limiter_window():
    now = [0.0]
    rl = RateLimiter(2, window=60, clock=lambda: now[0])
    assert rl.hit("a") == 0 and rl.hit("a") == 0
    assert rl.hit("a") == 60 and rl.hit("b") == 0
    now[0] = 61
    assert rl.hit("a") == 0
    assert RateLimiter(0).hit("x") == 0


def test_signup_login_logout_and_admin(tmp_path):
    client = TestClient(make_app(tmp_path))
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/health").json()["auth_enabled"] is True

    r = signup(client, "alice")
    assert r.status_code == 200 and r.json()["user"]["is_admin"] is True
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert client.get("/api/auth/me").json()["user"]["username"] == "alice"

    bob = TestClient(client.app)
    assert signup(bob, "bob").json()["user"]["is_admin"] is False
    assert signup(TestClient(client.app), "ALICE").status_code == 409
    assert signup(TestClient(client.app), "a b").status_code == 422
    assert signup(TestClient(client.app), "carol", "short").status_code == 422
    assert bob.get("/api/admin/users").status_code == 403
    assert [u["username"] for u in client.get("/api/admin/users").json()] == ["alice", "bob"]

    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401
    bad = client.post("/api/auth/login", json={"username": "alice", "password": "nope nope"})
    assert bad.status_code == 401
    ghost = client.post("/api/auth/login", json={"username": "ghost", "password": PW})
    assert ghost.status_code == 401 and ghost.json() == bad.json()
    assert (
        client.post("/api/auth/login", json={"username": "Alice", "password": PW}).status_code
        == 200
    )
    assert client.get("/api/auth/me").status_code == 200


def test_chats_are_private_per_user(tmp_path):
    app = make_app(tmp_path, turns=["hi alice"])
    alice, bob, anon = TestClient(app), TestClient(app), TestClient(app)
    signup(alice, "alice")
    signup(bob, "bob")

    assert anon.get("/api/sessions").status_code == 401
    assert anon.post("/api/chat", json={"message": "x"}).status_code == 401

    sid = sse_events(alice.post("/api/chat", json={"message": "secret plans"}))[0]["session_id"]
    assert [s["id"] for s in alice.get("/api/sessions").json()] == [sid]
    assert bob.get("/api/sessions").json() == []
    assert bob.get("/api/sessions", params={"q": "secret"}).json() == []
    for call in (
        lambda: bob.get(f"/api/sessions/{sid}"),
        lambda: bob.delete(f"/api/sessions/{sid}"),
        lambda: bob.patch(f"/api/sessions/{sid}", json={"title": "mine"}),
        lambda: bob.post(f"/api/sessions/{sid}/truncate", json={"from_index": 0}),
        lambda: bob.post(f"/api/sessions/{sid}/share"),
        lambda: bob.post("/api/chat", json={"session_id": sid, "message": "x"}),
    ):
        assert call().status_code == 404
    share = alice.post(f"/api/sessions/{sid}/share").json()["share_id"]
    assert anon.get(f"/api/shared/{share}").status_code == 200
    assert len(alice.get(f"/api/sessions/{sid}").json()["messages"]) == 2


def test_admin_from_env_and_signup_disabled(tmp_path):
    app = make_app(tmp_path, admin_username="boss", admin_password=PW, allow_signup=False)
    client = TestClient(app)
    assert signup(client, "eve").status_code == 403
    r = client.post("/api/auth/login", json={"username": "boss", "password": PW})
    assert r.json()["user"]["is_admin"] is True


def test_rate_limits(tmp_path):
    app = make_app(tmp_path, turns=["a", "b"], chat_rate_limit=1, auth_rate_limit=2)
    client = TestClient(app)
    assert signup(client, "alice").status_code == 200
    assert client.post("/api/chat", json={"message": "1"}).status_code == 200
    r = client.post("/api/chat", json={"message": "2"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    client.post("/api/auth/login", json={"username": "alice", "password": "x"})
    assert (
        client.post("/api/auth/login", json={"username": "alice", "password": PW}).status_code
        == 429
    )


def test_auth_off_uses_local_user(tmp_path):
    client = TestClient(make_app(tmp_path, turns=["ok"], auth_enabled=False))
    assert client.get("/api/auth/me").json()["user"]["id"] == "local"
    assert signup(client, "alice").status_code == 400
    sse_events(client.post("/api/chat", json={"message": "hey"}))
    assert len(client.get("/api/sessions").json()) == 1


def test_migrates_pre_accounts_db(tmp_path):
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        """CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT, tool_calls TEXT, tool_call_id TEXT, name TEXT,
            created_at TEXT NOT NULL);
        INSERT INTO sessions VALUES ('s1', 'old chat', '2026-01-01', '2026-01-01');
        INSERT INTO messages (session_id, role, content, created_at)
            VALUES ('s1', 'user', 'hi', '2026-01-01');"""
    )
    conn.commit()
    conn.close()
    store = Store(path)
    assert [s["id"] for s in store.list_sessions("local")] == ["s1"]
    assert store.get_messages("s1")[0]["content"] == "hi"
