from fastapi.testclient import TestClient

from grok_bot.mlx_llm import create_mlx_llm_app, format_messages_with_tools


def test_format_messages_with_tools_injects_protocol_and_normalizes_turns():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    messages = [
        {"role": "system", "content": "You are Grok."},
        {"role": "user", "content": [{"type": "text", "text": "Find SpaceX news"}]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "web_search", "arguments": '{"query": "SpaceX"}'},
                }
            ],
        },
        {"role": "tool", "name": "web_search", "content": "Starship flight 8"},
    ]
    norm = format_messages_with_tools(messages, tools)
    assert len(norm) == 4
    assert norm[0]["role"] == "system"
    assert "You are Grok." in norm[0]["content"]
    assert "## Tool Calling Protocol" in norm[0]["content"]
    assert '"name": "web_search"' in norm[0]["content"]
    assert norm[1] == {"role": "user", "content": "Find SpaceX news"}
    assert norm[2] == {
        "role": "assistant",
        "content": (
            '<tool_call>{"name": "web_search", "arguments": {"query": "SpaceX"}}</tool_call>'
        ),
    }
    assert norm[3] == {"role": "user", "content": "[Tool result (web_search)]:\nStarship flight 8"}


def test_format_messages_with_tools_inserts_system_when_absent():
    tools = [{"type": "function", "function": {"name": "python_exec", "description": "Run code"}}]
    norm = format_messages_with_tools([{"role": "user", "content": "2+2"}], tools)
    assert len(norm) == 2
    assert norm[0]["role"] == "system"
    assert "## Tool Calling Protocol" in norm[0]["content"]
    assert norm[1] == {"role": "user", "content": "2+2"}


def test_mlx_llm_endpoints_non_streaming_streaming_and_400():
    seen = {}

    def fake_gen(messages, *, model_repo="", adapter_path="", max_tokens=512):
        seen["messages"] = messages
        seen["model_repo"] = model_repo
        seen["adapter_path"] = adapter_path
        seen["max_tokens"] = max_tokens
        yield "Hello "
        yield "from MLX!"

    app = create_mlx_llm_app(
        model_repo="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        adapter_path="/tmp/adapters",
        model_alias="grok-lora",
        stream_generator=fake_gen,
    )
    client = TestClient(app)

    # /health and /v1/models
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model"] == "grok-lora"
    m = client.get("/v1/models").json()
    assert [x["id"] for x in m["data"]] == [
        "grok-lora",
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    ]

    # 400 on empty messages
    r_bad = client.post("/v1/chat/completions", json={"model": "grok-lora", "messages": []})
    assert r_bad.status_code == 400

    # Non-streaming completion
    r_sync = client.post(
        "/v1/chat/completions",
        json={
            "model": "grok-lora",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 64,
            "stream": False,
        },
    )
    assert r_sync.status_code == 200
    body = r_sync.json()
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello from MLX!"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert seen["max_tokens"] == 64

    # Streaming SSE completion ending in [DONE]
    r_sse = client.post(
        "/v1/chat/completions",
        json={
            "model": "grok-lora",
            "messages": [{"role": "user", "content": "Stream hi"}],
            "stream": True,
        },
    )
    assert r_sse.status_code == 200
    assert r_sse.headers["content-type"].startswith("text/event-stream")
    lines = [ln for ln in r_sse.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert any("Hello " in ln for ln in lines)
    assert any("from MLX!" in ln for ln in lines)


def test_book_excerpts_match_source_files_verbatim():
    import glob
    import pathlib
    import re

    pattern = re.compile(
        r"`([^`\n]+?)(?:\s*\(lines\s*(\d+)[–-](\d+)\))?`"
        r"(?:\s*\(lines\s*(\d+)[–-](\d+)\))?[^\n]*\n+```[a-zA-Z0-9_-]*\n(.*?)\n```",
        re.DOTALL,
    )
    checked = 0
    for path in sorted(glob.glob("docs/book/ch*.md")):
        text = pathlib.Path(path).read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            src = m.group(1).strip()
            start = m.group(2) or m.group(4)
            end = m.group(3) or m.group(5)
            if not start or not end:
                continue
            src_lines = pathlib.Path(src).read_text(encoding="utf-8").splitlines()
            expected = "\n".join(src_lines[int(start) - 1 : int(end)])
            assert m.group(6) == expected, f"Mismatch in {path}: {src} (lines {start}–{end})"
            checked += 1
    assert checked >= 60
