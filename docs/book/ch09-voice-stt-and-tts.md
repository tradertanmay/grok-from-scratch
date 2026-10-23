# Chapter 9 — End-to-End Voice Input & Output (Cloud Whisper, Local Whisper & Browser Web Speech)

> **Source Files Covered:** [`grok_bot/stt.py`](../../grok_bot/stt.py) · [`grok_bot/mlx_stt.py`](../../grok_bot/mlx_stt.py) · [`grok_bot/app.py`](../../grok_bot/app.py) · [`frontend/app.js`](../../frontend/app.js) · [`tests/test_stt.py`](../../tests/test_stt.py)

Grok Bot includes full **Voice Input (Speech-to-Text)** and **Voice Output (Text-to-Speech)** to Grok Bot. Because most users do **not** have a dedicated GPU or high-end workstation, Grok Bot supports **three zero-H100 voice setups** that work on any 8 GB Windows, Linux, or macOS laptop:

| Setup | Hardware Needed | Configuration | How It Works |
| :--- | :--- | :--- | :--- |
| **1. Browser Web Speech (Default)** | Any laptop / phone (`0 MB` RAM) | Leave `GROK_STT_URL` unset | `frontend/app.js` uses `window.SpeechRecognition` for mic input and `window.speechSynthesis` for **🔊 Speak** output |
| **2. Free Cloud Whisper API** | Any 8 GB laptop (`0 MB` RAM) | `GROK_STT_URL=https://api.groq.com/openai/v1`<br>`GROK_STT_KEY=$GROQ_API_KEY`<br>`GROK_STT_MODEL=whisper-large-v3-turbo` | Browser `MediaRecorder` uploads audio to `POST /api/transcribe`, which proxies to Groq's free OpenAI-compatible Whisper API |
| **3. Self-Hosted Local Whisper** | Linux/Windows (`faster-whisper` / `whisper.cpp`) or macOS (`grok_bot.mlx_stt`) | `GROK_STT_URL=http://127.0.0.1:8089/v1` | `POST /api/transcribe` forwards audio to your local `/v1/audio/transcriptions` server |

---

## 1. End-to-End Voice Architecture

```mermaid
sequenceDiagram
    participant Mic as Browser Microphone
    participant UI as frontend/app.js
    participant API as grok_bot/app.py (/api/transcribe)
    participant STT as Any OpenAI Whisper Server (Groq Cloud, faster-whisper, or mlx_stt :8089)

    UI->>API: GET /api/health
    API-->>UI: {"status": "ok", "stt": true | false}
    alt stt == false (Zero-Config Browser Speech)
        Mic->>UI: window.SpeechRecognition (dictation directly into #chat-input)
    else stt == true (Server Whisper via MediaRecorder)
        Mic->>UI: Click 🎤 Voice (MediaRecorder audio/webm;codecs=opus)
        Mic->>UI: Click ⏹ Stop Mic
        UI->>API: POST /api/transcribe (multipart/form-data file=voice.webm)
        API->>STT: POST /v1/audio/transcriptions (Bearer GROK_STT_KEY)
        STT-->>API: {"text": "Hello Grok, summarize the latest SpaceX news."}
        API-->>UI: Populate #chat-input & focus composer
    end
```

---

## 2. Backend Proxy (`grok_bot/stt.py` & `POST /api/transcribe`)

[`grok_bot/stt.py`](../../grok_bot/stt.py) implements an async OpenAI-compatible `/audio/transcriptions` client (`STT`) that works identically with **Groq's free Whisper API**, **OpenAI**, **`faster-whisper-server` on Linux/Windows**, or **`grok_bot.mlx_stt`**:

`grok_bot/stt.py` (lines 36–89):

```python
class STT:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.url = base_url.rstrip("/") + "/audio/transcriptions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport

    async def transcribe(
        self,
        filename: str,
        content_type: str,
        data: bytes,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        form = {"model": self.model, "response_format": "json"}
        if language:
            form["language"] = language
        if prompt:
            form["prompt"] = prompt
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as c:
                r = await c.post(
                    self.url,
                    data=form,
                    files={"file": (filename or "audio.webm", data, content_type)},
                    headers=headers,
                )
        except httpx.HTTPError as e:
            raise STTError(f"speech-to-text server unreachable: {e}") from e
        if r.status_code >= 400:
            raise STTError(f"speech-to-text server returned {r.status_code}: {r.text[:200]}")
        try:
            body = r.json()
        except ValueError:
            body = {"text": r.text}
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            raise STTError("speech-to-text server returned no text")
        out: dict[str, Any] = {"text": text.strip()}
        for k in ("language", "duration"):
            if body.get(k) is not None:
                out[k] = body[k]
        return out
```

And in [`grok_bot/app.py`](../../grok_bot/app.py), `POST /api/transcribe` enforces authentication, rate limiting, and a streaming upload cap (`GROK_STT_MAX_MB`, default `25 MB`) before forwarding the audio bytes to `STT.transcribe()`.

---

## 3. Self-Hosted Local Whisper Server (`grok_bot/mlx_stt.py`)

If you want to run Whisper locally instead of calling Groq's cloud Whisper endpoint:
- **On Linux / Windows (CPU or NVIDIA GPU):** Run `faster-whisper-server` or `whisper.cpp --port 8089` and set `GROK_STT_URL=http://127.0.0.1:8089/v1`.
- **On macOS (`grok_bot/mlx_stt.py`):** We also ship a lightweight FastAPI server in [`grok_bot/mlx_stt.py`](../../grok_bot/mlx_stt.py) (`default_model = "mlx-community/whisper-tiny-mlx"`, only ~150 MB) that exposes the exact same `/v1/audio/transcriptions` contract:

`grok_bot/mlx_stt.py` (lines 23–81):

```python
def transcribe_file(
    audio_path: str | Path,
    model_repo: str = DEFAULT_MODEL,
    language: str | None = None,
    initial_prompt: str | None = None,
) -> dict[str, Any]:
    try:
        import mlx_whisper
    except ImportError as e:
        raise RuntimeError("mlx-whisper is not installed (`uv pip install mlx-whisper`)") from e

    kwargs: dict[str, Any] = {"path_or_hf_repo": model_repo}
    if language:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    res = mlx_whisper.transcribe(str(audio_path), **kwargs)
    return {
        "text": (res.get("text") or "").strip(),
        "language": res.get("language") or language or "en",
    }


def create_stt_app(default_model: str = DEFAULT_MODEL, transcriber=transcribe_file) -> FastAPI:
    app = FastAPI(title="Grok Bot Mac MLX-Whisper STT")

    @app.get("/health")
    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model": default_model}

    @app.post("/audio/transcriptions")
    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),  # noqa: B008
        model: str = Form(default=""),
        language: str | None = Form(default=None),
        prompt: str | None = Form(default=None),
        response_format: str = Form(default="json"),
    ):
        data = await file.read()
        if not data:
            raise HTTPException(422, "empty audio upload")
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        chosen_model = default_model if not model or model in ("whisper-1", "default") else model
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            result = transcriber(
                tmp.name,
                model_repo=chosen_model,
                language=language,
                initial_prompt=prompt,
            )
        if response_format == "text":
            return PlainTextResponse(result["text"])
        return result

    return app
```

---

## 4. Dual-Path Browser Mic Capture & `speechSynthesis` TTS (`frontend/app.js`)

In `frontend/app.js`, `toggleVoiceInput()` checks `state.sttEnabled` (populated from `GET /api/health` → `h.stt`). When `true`, it records microphone audio via `MediaRecorder` and uploads the blob to `POST /api/transcribe`; when `false`, it falls back transparently to `window.SpeechRecognition || window.webkitSpeechRecognition`.

For voice output, `speakAssistantReply(text, btnEl)` in `frontend/app.js` uses `window.speechSynthesis` so every assistant message has a zero-latency **🔊 Speak** / **⏹ Stop TTS** toggle on any laptop or mobile browser.

---

## 5. Try it

1. **Run the STT unit and proxy test suite (works on any laptop with zero external server):**
   ```bash
   .venv/bin/pytest tests/test_stt.py tests/test_frontend.py -k "stt" -v
   ```
   Real output:
   ```text
   tests/test_stt.py::test_health_reports_stt_flag PASSED                   [ 14%]
   tests/test_stt.py::test_transcribe_disabled_returns_501 PASSED           [ 28%]
   tests/test_stt.py::test_transcribe_proxies_multipart_to_upstream PASSED  [ 42%]
   tests/test_stt.py::test_transcribe_rejects_empty_and_oversized_files PASSED [ 57%]
   tests/test_stt.py::test_transcribe_upstream_errors_map_to_502 PASSED     [ 71%]
   tests/test_stt.py::test_transcribe_requires_auth_when_enabled PASSED     [ 85%]
   tests/test_frontend.py::test_mlx_stt_server_routes PASSED                [100%]
   ============================== 7 passed in 0.42s ===============================
   ```

2. **Point Grok Bot at Groq's free cloud Whisper API (any 8 GB laptop, zero local model):**
   ```bash
   GROK_STT_URL=https://api.groq.com/openai/v1 \
   GROK_STT_KEY=$GROQ_API_KEY \
   GROK_STT_MODEL=whisper-large-v3-turbo \
   .venv/bin/python -m grok_bot
   ```

---

## 6. Exercises

1. **Language Hint Selector:** Pass the browser's `navigator.language.slice(0, 2)` as a `language` form field from `frontend/app.js` through `POST /api/transcribe` to `STT.transcribe()` so non-English dictation skips Whisper's language-detection pass.
2. **Auto-Send Voice Mode:** Add a user preference toggle (`Auto-send after voice transcription`) in `frontend/app.js` that calls `sendChatMessage()` automatically as soon as `POST /api/transcribe` returns non-empty text.

---

[← Chapter 8: Admin Console, Scheduled Tasks & API Keys](ch08-admin-tasks-keys-and-portability.md) · [Book Index](README.md) · [Chapter 10: Persona Fine-Tuning with LoRA (Free Colab Unsloth & Local MLX) →](ch10-lora-finetuning-mlx-and-unsloth.md)
