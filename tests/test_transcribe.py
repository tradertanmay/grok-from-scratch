import httpx
from fastapi.testclient import TestClient

from grok_bot.app import create_app
from grok_bot.config import Settings
from grok_bot.store import Store
from grok_bot.stt import STT
from grok_bot.tools import ToolRegistry
from tests.conftest import FakeLLM
from tests.test_auth import signup

WEBM = ("note.webm", b"\x1aE\xdf\xa3fake-audio", "audio/webm")


def setup(tmp_path, handler=None, **settings):
    tmp_path.mkdir(parents=True, exist_ok=True)
    seen = []

    def default(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "  hello grok  ", "language": "en"})

    stt = (
        STT(
            "http://stt.local/v1",
            "whisper-x",
            "sk-test",
            transport=httpx.MockTransport(handler or default),
        )
        if handler is not False
        else None
    )
    app = create_app(
        Settings(llm_model="fake", **settings),
        llm=FakeLLM([]),
        store=Store(str(tmp_path / "t.db")),
        tools=ToolRegistry([]),
        stt=stt,
    )
    client = TestClient(app)
    signup(client, "ann")
    return app, client, seen


def test_transcribe_forwards_to_openai_compatible_server(tmp_path):
    _, client, seen = setup(tmp_path)
    assert client.get("/api/health").json()["stt"] is True
    r = client.post("/api/transcribe", files={"file": WEBM}, data={"language": "en"})
    assert r.status_code == 200, r.text
    assert r.json() == {"text": "hello grok", "language": "en"}
    [req] = seen
    assert str(req.url) == "http://stt.local/v1/audio/transcriptions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = req.content
    assert b'name="model"\r\n\r\nwhisper-x' in body
    assert b'name="language"\r\n\r\nen' in body
    assert b"fake-audio" in body


def test_transcribe_requires_auth_and_config(tmp_path):
    app, _, _ = setup(tmp_path)
    assert TestClient(app).post("/api/transcribe", files={"file": WEBM}).status_code == 401
    _, client, _ = setup(tmp_path / "off", handler=False)
    assert client.get("/api/health").json()["stt"] is False
    assert client.post("/api/transcribe", files={"file": WEBM}).status_code == 503


def test_transcribe_validates_upload(tmp_path):
    _, client, seen = setup(tmp_path, max_upload_mb=1)
    pdf = ("x.pdf", b"%PDF", "application/pdf")
    assert client.post("/api/transcribe", files={"file": pdf}).status_code == 415
    empty = ("x.webm", b"", "audio/webm")
    assert client.post("/api/transcribe", files={"file": empty}).status_code == 422
    big = ("x.webm", b"0" * (1024 * 1024 + 1), "audio/webm;codecs=opus")
    assert client.post("/api/transcribe", files={"file": big}).status_code == 413
    assert seen == []


def test_transcribe_upstream_errors_are_502(tmp_path):
    def broken(request):
        return httpx.Response(500, text="model not loaded")

    _, client, _ = setup(tmp_path, handler=broken)
    r = client.post("/api/transcribe", files={"file": WEBM})
    assert r.status_code == 502 and "500" in r.json()["detail"]

    def down(request):
        raise httpx.ConnectError("refused")

    _, client, _ = setup(tmp_path / "down", handler=down)
    assert client.post("/api/transcribe", files={"file": WEBM}).status_code == 502


def test_v1_audio_transcriptions_with_api_key(tmp_path):
    app, client, _ = setup(tmp_path)
    key = client.post("/api/keys", json={"name": "sdk"}).json()["key"]
    api = TestClient(app, headers={"Authorization": f"Bearer {key}"})
    r = api.post("/v1/audio/transcriptions", files={"file": WEBM}, data={"model": "whisper-1"})
    assert r.status_code == 200 and r.json()["text"] == "hello grok"
    r = api.post("/v1/audio/transcriptions", files={"file": WEBM}, data={"response_format": "text"})
    assert r.status_code == 200 and r.text == "hello grok"
