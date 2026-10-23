import asyncio
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from grok_bot.tasks import next_run
from tests.test_api import sse_events
from tests.test_auth import make_app, signup

MON_9 = datetime(2026, 10, 19, 9, 0, tzinfo=UTC)  # a Monday


def test_next_run_math():
    assert next_run("daily", MON_9, time="08:00") == MON_9 + timedelta(hours=23)
    assert next_run("daily", MON_9, time="10:30") == MON_9 + timedelta(hours=1, minutes=30)
    assert next_run("daily", MON_9, time="09:00") == MON_9 + timedelta(days=1)
    assert next_run("weekly", MON_9, time="09:00", weekday=2) == MON_9 + timedelta(days=2)
    assert next_run("weekly", MON_9, time="08:00", weekday=0) == MON_9 + timedelta(days=7, hours=-1)
    assert next_run("once", MON_9, at="2026-10-19T08:00:00") is None
    assert next_run("once", MON_9, at="2026-10-19T12:00:00+02:00") == MON_9 + timedelta(hours=1)


def setup(tmp_path, turns=(), **settings):
    app = make_app(tmp_path, turns=turns, **settings)
    signup(TestClient(app), "boss")
    ann = TestClient(app)
    signup(ann, "ann")
    return app, ann


def test_crud_and_validation(tmp_path):
    app, ann = setup(tmp_path)
    t = ann.post("/api/tasks", json={"title": "News", "prompt": "AI news", "time": "07:30"}).json()
    assert t["schedule"] == "daily" and t["enabled"] is True and t["next_run_at"]
    assert ann.post("/api/tasks", json={"title": "x", "prompt": "y"}).status_code == 400
    past = {"title": "x", "prompt": "y", "schedule": "once", "at": "2020-01-01T00:00:00Z"}
    assert ann.post("/api/tasks", json=past).status_code == 400
    bad_time = {"title": "x", "prompt": "y", "time": "25:00"}
    assert ann.post("/api/tasks", json=bad_time).status_code == 422

    off = ann.patch(f"/api/tasks/{t['id']}", json={"enabled": False}).json()
    assert off["enabled"] is False and off["next_run_at"] is None
    on = ann.patch(f"/api/tasks/{t['id']}", json={"enabled": True, "time": "08:00"}).json()
    assert on["next_run_at"].endswith("08:00:00+00:00")

    boss = TestClient(app)
    boss.post("/api/auth/login", json={"username": "boss", "password": "correct horse"})
    assert boss.get("/api/tasks").json() == []
    assert boss.patch(f"/api/tasks/{t['id']}", json={"title": "z"}).status_code == 404
    assert boss.delete(f"/api/tasks/{t['id']}").status_code == 404
    assert ann.delete(f"/api/tasks/{t['id']}").json() == {"ok": True}
    assert ann.get("/api/tasks").json() == []


def test_runner_runs_due_tasks_once(tmp_path):
    app, ann = setup(tmp_path, turns=["daily digest", "one-off answer"])
    daily = ann.post("/api/tasks", json={"title": "Digest", "prompt": "digest", "time": "07:00"})
    soon = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    once = ann.post(
        "/api/tasks", json={"title": "Remind", "prompt": "remind", "schedule": "once", "at": soon}
    )
    later = datetime.now(UTC) + timedelta(days=1, hours=2)
    assert asyncio.run(app.state.run_due_tasks(later)) == 2
    assert asyncio.run(app.state.run_due_tasks(later)) == 0

    tasks = {t["title"]: t for t in ann.get("/api/tasks").json()}
    assert (
        tasks["Digest"]["last_status"] == "ok"
        and tasks["Digest"]["next_run_at"] > later.isoformat()
    )
    assert tasks["Remind"]["enabled"] is False and tasks["Remind"]["next_run_at"] is None
    sid = tasks["Digest"]["last_session_id"]
    msgs = ann.get(f"/api/sessions/{sid}").json()["messages"]
    assert msgs[0]["content"] == "digest"
    assert msgs[1]["role"] == "assistant" and msgs[1]["content"] in (
        "daily digest",
        "one-off answer",
    )
    titles = [s["title"] for s in ann.get("/api/sessions").json()]
    assert sorted(titles) == ["Task: Digest", "Task: Remind"]
    assert daily.status_code == once.status_code == 200


def test_run_now_and_quota_skip(tmp_path):
    app, ann = setup(tmp_path, turns=["now!"], daily_messages=1)
    t = ann.post("/api/tasks", json={"title": "T", "prompt": "p", "time": "07:00"}).json()
    events = sse_events(ann.post(f"/api/tasks/{t['id']}/run"))
    assert "".join(e.get("content", "") for e in events if e["type"] == "token") == "now!"
    assert ann.get("/api/tasks").json()[0]["last_status"] == "ok"
    assert ann.post(f"/api/tasks/{t['id']}/run").status_code == 429
    asyncio.run(app.state.run_due_tasks(datetime.now(UTC) + timedelta(days=1, hours=1)))
    assert ann.get("/api/tasks").json()[0]["last_status"] == "skipped: daily limit reached"
