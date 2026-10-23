"""Local Apple Silicon MLX + LoRA OpenAI-compatible LLM server (`grok_bot.mlx_llm`).

Serves `POST /v1/chat/completions` (streaming & non-streaming) and `GET /v1/models`
using `mlx-lm` on Apple Silicon Metal GPU, with optional `--adapter-path` for LoRA
adapters trained via `training/train_mac_mlx.py`. Automatically translates OpenAI
`tools` schemas and `role="tool"` turns into the `<tool_call>{...}</tool_call>`
text protocol understood by `grok_bot/agent.py`.

Usage:
    python -m grok_bot.mlx_llm --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
        --adapter-path adapters --host 127.0.0.1 --port 8088
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

DEFAULT_MLX_LLM_MODEL = os.environ.get(
    "GROK_MLX_LLM_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
)
DEFAULT_MLX_ADAPTER_PATH = os.environ.get("GROK_MLX_ADAPTER_PATH", "")
DEFAULT_MLX_MODEL_ALIAS = os.environ.get("GROK_MLX_MODEL_ALIAS", "grok-lora")

_mlx_lock = threading.Lock()
_loaded_state: dict[str, Any] = {}


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MLX_MODEL_ALIAS
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    stream: bool = False
    max_tokens: int = 512


def format_messages_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Normalize OpenAI messages & tools into system/user/assistant turns for chat templates."""
    tool_block = ""
    if tools:
        specs = []
        for t in tools:
            fn = t.get("function") if isinstance(t, dict) else None
            if isinstance(fn, dict) and fn.get("name"):
                specs.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    }
                )
        if specs:
            tool_block = (
                "\n\n## Tool Calling Protocol\n"
                "When you need to call a tool, respond ONLY with a single tag "
                "in this exact format:\n"
                '<tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>\n'
                "Available tools:\n" + json.dumps(specs, ensure_ascii=False)
            )

    normalized: list[dict[str, str]] = []
    injected_tools = False
    for m in messages:
        role = str(m.get("role") or "user")
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(
                str(p.get("text") or "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        text = str(content or "")

        if role in ("system", "developer"):
            if tool_block and not injected_tools:
                text = text + tool_block
                injected_tools = True
            normalized.append({"role": "system", "content": text})
        elif role == "tool":
            name = m.get("name") or "tool"
            normalized.append({"role": "user", "content": f"[Tool result ({name})]:\n{text}"})
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs and not text:
                tc_tags = []
                for tc in tcs:
                    fn = (tc.get("function") if isinstance(tc, dict) else None) or {}
                    raw_args = fn.get("arguments", "{}")
                    try:
                        parsed_args = (
                            json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        )
                    except json.JSONDecodeError:
                        parsed_args = {}
                    payload = json.dumps(
                        {"name": fn.get("name", ""), "arguments": parsed_args},
                        ensure_ascii=False,
                    )
                    tc_tags.append(f"<tool_call>{payload}</tool_call>")
                text = "\n".join(tc_tags)
            normalized.append({"role": "assistant", "content": text})
        else:
            normalized.append({"role": "user", "content": text})

    if tool_block and not injected_tools:
        normalized.insert(0, {"role": "system", "content": tool_block.strip()})
    return normalized


def _default_stream_generator(
    messages: list[dict[str, str]],
    *,
    model_repo: str,
    adapter_path: str,
    max_tokens: int,
) -> Iterable[str]:
    import mlx_lm

    key = f"{model_repo}::{adapter_path}"
    with _mlx_lock:
        if _loaded_state.get("key") != key:
            load_kwargs: dict[str, Any] = {}
            if adapter_path:
                load_kwargs["adapter_path"] = adapter_path
            model, tokenizer = mlx_lm.load(model_repo, **load_kwargs)
            _loaded_state["key"] = key
            _loaded_state["model"] = model
            _loaded_state["tokenizer"] = tokenizer
        else:
            model = _loaded_state["model"]
            tokenizer = _loaded_state["tokenizer"]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
            piece = getattr(resp, "text", "") if not isinstance(resp, str) else resp
            if piece:
                yield piece


def create_mlx_llm_app(
    model_repo: str = DEFAULT_MLX_LLM_MODEL,
    adapter_path: str = DEFAULT_MLX_ADAPTER_PATH,
    model_alias: str = DEFAULT_MLX_MODEL_ALIAS,
    stream_generator: Callable[..., Iterable[str]] | None = None,
) -> FastAPI:
    gen_fn = stream_generator or _default_stream_generator
    app = FastAPI(title="Grok Bot MLX-LM LoRA Server")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_repo": model_repo,
            "adapter_path": adapter_path or None,
            "model": model_alias,
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        ids = list(dict.fromkeys([model_alias, model_repo]))
        return {
            "object": "list",
            "data": [
                {"id": mid, "object": "model", "created": 0, "owned_by": "mlx-lm"} for mid in ids
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        if not body.messages:
            raise HTTPException(400, "messages cannot be empty")
        norm_msgs = format_messages_with_tools(body.messages, body.tools)
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not body.stream:
            pieces = await run_in_threadpool(
                lambda: list(
                    gen_fn(
                        norm_msgs,
                        model_repo=model_repo,
                        adapter_path=adapter_path,
                        max_tokens=body.max_tokens,
                    )
                )
            )
            full_text = "".join(pieces)
            return {
                "id": cmpl_id,
                "object": "chat.completion",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_text},
                        "finish_reason": "stop",
                    }
                ],
            }

        async def _sse():
            first = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(first)}\n\n"
            it = gen_fn(
                norm_msgs,
                model_repo=model_repo,
                adapter_path=adapter_path,
                max_tokens=body.max_tokens,
            )
            async for piece in iterate_in_threadpool(it):
                chunk = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": body.model,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            done_chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="python -m grok_bot.mlx_llm",
        description="Run local Apple Silicon MLX-LM + LoRA OpenAI-compatible server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8088, help="Bind port (default: 8088)")
    parser.add_argument("--model", default=DEFAULT_MLX_LLM_MODEL, help="MLX HuggingFace model repo")
    parser.add_argument(
        "--adapter-path",
        default=DEFAULT_MLX_ADAPTER_PATH,
        help="Optional directory containing MLX LoRA adapters (adapters.safetensors)",
    )
    parser.add_argument(
        "--alias",
        default=DEFAULT_MLX_MODEL_ALIAS,
        help="Model id exposed in /v1/models (default: grok-lora)",
    )
    args = parser.parse_args(argv)

    app = create_mlx_llm_app(
        model_repo=args.model,
        adapter_path=args.adapter_path,
        model_alias=args.alias,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
