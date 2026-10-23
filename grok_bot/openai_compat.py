"""Translation between the OpenAI chat-completions wire format and Grok turns."""

import json
import time
import uuid
from typing import Any

# model id -> (persona, think, deepsearch); any other id is routed as a provider model.
PERSONA_MODELS: dict[str, tuple[str, bool, bool]] = {
    "grok": ("fun", False, False),
    "grok-regular": ("regular", False, False),
    "grok-think": ("fun", True, False),
    "grok-regular-think": ("regular", True, False),
    "grok-deepsearch": ("fun", False, True),
}


def resolve_model(model: str) -> dict[str, Any]:
    if model in PERSONA_MODELS:
        mode, think, deepsearch = PERSONA_MODELS[model]
        return {"model": None, "mode": mode, "think": think, "deepsearch": deepsearch}
    return {"model": model, "mode": "fun", "think": False, "deepsearch": False}


def split_content(content: Any) -> tuple[str, list[str]]:
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    texts, images = [], []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(str(part.get("text") or ""))
        elif part.get("type") == "image_url":
            url = part.get("image_url")
            url = url.get("url") if isinstance(url, dict) else url
            if isinstance(url, str):
                images.append(url)
    return "\n".join(texts), images


def split_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]], str, list[str]]:
    """-> (extra system text, prior user/assistant turns, last user text, last user images)."""
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("the last message must have role 'user'")
    system, history = [], []
    for m in messages[:-1]:
        text, _ = split_content(m.get("content"))
        if m.get("role") in ("system", "developer"):
            system.append(text)
        elif m.get("role") in ("user", "assistant") and text:
            history.append({"role": m["role"], "content": text})
    text, images = split_content(messages[-1].get("content"))
    return "\n\n".join(s for s in system if s), history, text, images


def estimate_tokens(chars: int) -> int:
    return (chars + 3) // 4


class Completion:
    def __init__(self, model: str):
        self.id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        self.created = int(time.time())
        self.model = model

    def _base(self, obj: str) -> dict[str, Any]:
        return {"id": self.id, "object": obj, "created": self.created, "model": self.model}

    def chunk(self, delta: dict[str, Any], finish: str | None = None) -> str:
        body = self._base("chat.completion.chunk")
        body["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    def usage_chunk(self, usage: dict[str, int]) -> str:
        body = {**self._base("chat.completion.chunk"), "choices": [], "usage": usage}
        return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

    def full(self, content: str, reasoning: str, usage: dict[str, int]) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        return {
            **self._base("chat.completion"),
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": usage,
        }


def usage(prompt_chars: int, completion_chars: int) -> dict[str, int]:
    p, c = estimate_tokens(prompt_chars), estimate_tokens(completion_chars)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
