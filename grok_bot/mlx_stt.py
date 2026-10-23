"""Apple Silicon MLX Whisper OpenAI-compatible Speech-to-Text server.

Run on Mac:
    python -m grok_bot.mlx_stt --port 8089 --model mlx-community/whisper-tiny-mlx

Then point Grok Bot at it:
    GROK_STT_URL=http://127.0.0.1:8089/v1
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

DEFAULT_MODEL = os.environ.get("GROK_MLX_WHISPER_MODEL", "mlx-community/whisper-tiny-mlx")


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grok_bot.mlx_stt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args(argv)
    uvicorn.run(create_stt_app(default_model=args.model), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
