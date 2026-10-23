import json
import logging

from fastapi.testclient import TestClient

from grok_bot.observability import JsonFormatter, Metrics
from tests.test_api import sse_events
from tests.test_auth import make_app


def test_metrics_requires_token(tmp_path):
    assert TestClient(make_app(tmp_path)).get("/metrics").status_code == 404
    client = TestClient(make_app(tmp_path, metrics_token="s3cret"))
    assert client.get("/metrics").status_code == 404
    assert client.get("/metrics", headers={"Authorization": "Bearer nope"}).status_code == 404
    assert client.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_metrics_count_requests_and_turns(tmp_path):
    client = TestClient(make_app(tmp_path, turns=["hey"], auth_enabled=False, metrics_token="t"))
    client.get("/api/health")
    sse_events(client.post("/api/chat", json={"message": "hi", "think": False}))
    client.get("/api/sessions/nope")
    text = client.get("/metrics", headers={"Authorization": "Bearer t"}).text
    assert 'grok_http_requests_total{method="GET",route="/api/health",status="200"} 1' in text
    assert 'route="/api/sessions/{session_id}",status="404"} 1' in text
    assert 'grok_chat_turns_total{model="fake",outcome="ok"} 1' in text
    assert "# TYPE grok_uptime_seconds gauge" in text and "grok_users 0" in text


def test_metrics_label_escaping():
    m = Metrics()
    m.inc("x_total", model='a"b\\c')
    assert 'x_total{model="a\\"b\\\\c"} 1' in m.render()


def test_json_formatter():
    rec = logging.LogRecord("grok_bot.access", logging.INFO, "", 0, "GET %s", ("/x",), None)
    rec.fields = {"status": 200, "ms": 1.5}
    out = json.loads(JsonFormatter().format(rec))
    assert out["msg"] == "GET /x" and out["status"] == 200 and out["level"] == "info"
