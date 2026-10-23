"""Speech-to-text via any OpenAI-compatible /audio/transcriptions endpoint.

Works with a Mac mlx-whisper or whisper.cpp server, Groq's free whisper API,
or OpenAI itself: set GROK_STT_URL to the server's /v1 base URL.
"""

from typing import Any

import httpx

AUDIO_TYPES = {
    "audio/webm",
    "audio/ogg",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/aac",
    "audio/flac",
    "video/webm",
}


class STTError(Exception):
    pass


def is_audio(content_type: str) -> bool:
    return content_type.split(";")[0].strip().lower() in AUDIO_TYPES


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
