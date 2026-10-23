import asyncio

from fastapi.testclient import TestClient

from grok_bot.app import create_app
from grok_bot.cli import format_cli_event, run_single_turn
from grok_bot.config import Settings
from grok_bot.store import Store
from grok_bot.tools import default_registry

from .conftest import FakeLLM, text_turn


def test_frontend_static_assets_mounted(tmp_path):
    llm = FakeLLM([text_turn("Hello from Grok UI")])
    settings = Settings(llm_model="test-model", db_path=str(tmp_path / "ui.db"))
    client = TestClient(create_app(settings=settings, llm=llm))

    index_resp = client.get("/")
    assert index_resp.status_code == 200
    assert "Grok Bot" in index_resp.text
    assert 'id="chat-input"' in index_resp.text
    assert (
        'id="mode-think-btn" type="button" class="mode-pill" aria-pressed="false"'
        in index_resp.text
    )
    assert 'id="mode-deepsearch-btn"' in index_resp.text
    assert 'id="admin-modal"' in index_resp.text
    assert 'id="account-modal"' in index_resp.text
    assert 'id="tasks-modal"' in index_resp.text
    assert 'id="api-keys-modal"' in index_resp.text
    assert 'id="voice-input-btn"' in index_resp.text
    assert 'id="canvas-pane"' in index_resp.text
    assert 'id="canvas-iframe"' in index_resp.text
    assert 'id="auth-invite-field"' in index_resp.text
    assert 'id="quota-badge"' in index_resp.text

    for path, snippet in [
        ("/styles.css", "--accent-primary"),
        ("/styles.css", ".canvas-pane"),
        ("/app.js", "regenerateLastReply"),
        ("/app.js", "runTaskNow"),
        ("/app.js", "toggleVoiceInput"),
        ("/app.js", "openCanvas"),
        ("/manifest.json", "GrokBot"),
        ("/sw.js", "CACHE_NAME"),
        ("/icon.svg", "<svg"),
    ]:
        r = client.get(path)
        assert r.status_code == 200, f"Failed to serve {path}"
        assert snippet in r.text

    manifest = client.get("/manifest.json").json()
    assert manifest["display"] == "standalone"
    assert manifest["start_url"] == "/"
    assert any(i["src"] == "/icon.svg" for i in manifest["icons"])


def test_mlx_stt_server_routes():
    from grok_bot.mlx_stt import create_stt_app

    seen = {}

    def fake_transcriber(path, model_repo="", language=None, initial_prompt=None):
        seen["model_repo"] = model_repo
        seen["language"] = language
        return {"text": "Hello from MLX Whisper", "language": language or "en"}

    client = TestClient(
        create_stt_app(default_model="mlx-community/whisper-tiny-mlx", transcriber=fake_transcriber)
    )
    assert client.get("/health").json()["status"] == "ok"
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("voice.wav", b"RIFF....WAVEfmt ", "audio/wav")},
        data={"model": "whisper-1", "language": "en"},
    )
    assert r.status_code == 200
    assert r.json() == {"text": "Hello from MLX Whisper", "language": "en"}
    assert seen["model_repo"] == "mlx-community/whisper-tiny-mlx"

    r_txt = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("voice.wav", b"RIFF....WAVEfmt ", "audio/wav")},
        data={"response_format": "text"},
    )
    assert r_txt.status_code == 200 and r_txt.text == "Hello from MLX Whisper"


def test_mlx_llm_server_routes_and_tool_protocol():
    from grok_bot.mlx_llm import create_mlx_llm_app

    seen = {}

    def fake_gen(messages, *, model_repo="", adapter_path="", max_tokens=512):
        seen["messages"] = messages
        seen["model_repo"] = model_repo
        seen["adapter_path"] = adapter_path
        seen["max_tokens"] = max_tokens
        if any("Tool Calling Protocol" in m.get("content", "") for m in messages):
            yield '<tool_call>{"name": "web_search", "arguments": {"query": "MLX"}}</tool_call>'
        else:
            yield "Hello "
            yield "from MLX LoRA!"

    client = TestClient(
        create_mlx_llm_app(
            model_repo="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            adapter_path="/tmp/grok_mlx_adapters",
            model_alias="grok-lora",
            stream_generator=fake_gen,
        )
    )
    assert client.get("/health").json()["model"] == "grok-lora"
    models = client.get("/v1/models").json()["data"]
    assert models[0]["id"] == "grok-lora"

    # Non-streaming completion
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "grok-lora",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Hello from MLX LoRA!"

    # Streaming completion with tools passed (verifies <tool_call> protocol compatibility)
    r_stream = client.post(
        "/v1/chat/completions",
        json={
            "model": "grok-lora",
            "messages": [{"role": "user", "content": "Search MLX"}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
            "stream": True,
        },
    )
    assert r_stream.status_code == 200
    assert "<tool_call>" in r_stream.text
    assert "data: [DONE]" in r_stream.text


def test_cli_single_turn_and_formatting(tmp_path):
    llm = FakeLLM([text_turn("CLI reply works!")])
    store = Store(str(tmp_path / "cli.db"))
    tools = default_registry()
    sid = store.create_session("Test CLI")

    reply = asyncio.run(run_single_turn("Hi Grok", sid, store, llm, tools, "fake-model"))
    assert reply == "CLI reply works!"
    msgs = store.get_messages(sid)
    assert len(msgs) == 2
    assert msgs[1]["content"] == "CLI reply works!"

    assert "tool_call" in format_cli_event(
        {"type": "tool_call", "name": "web_search", "arguments": {"query": "x"}}
    )
    assert "OK" in format_cli_event({"type": "tool_result", "output": "found", "is_error": False})


def test_playwright_browser_e2e(tmp_path):
    import socket
    import threading
    import time

    import pytest
    import uvicorn

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    llm = FakeLLM(
        [
            text_turn("First reply from Grok E2E!"),
            text_turn("Regenerated reply from Grok E2E!"),
            text_turn(
                "Task digest reply from Grok E2E!\n\n```html\n<h1>Hello Grok Canvas</h1>\n```"
            ),
        ]
    )
    settings = Settings(
        llm_model="gpt-oss:latest",
        db_path=str(tmp_path / "pw.db"),
        auth_enabled=True,
        signup="open",
        daily_messages=10,
        embed_model="",
        tasks_enabled=False,
    )
    app = create_app(settings=settings, llm=llm, store=Store(settings.db_path))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)

    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="chrome", headless=True)
            except Exception:
                try:
                    browser = p.chromium.launch(headless=True)
                except Exception:
                    pytest.skip("No Chromium/Chrome binary available for Playwright")
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{port}/")

            # 1. Think button defaults to OFF
            assert page.get_attribute("#mode-think-btn", "aria-pressed") == "false"

            # 2. Sign up first user (becomes admin automatically)
            page.click("#auth-tab-signup")
            page.fill("#auth-username", "admin_user")
            page.fill("#auth-password", "supersecret123")
            page.click("#auth-submit-btn")
            page.wait_for_selector("#auth-modal.hidden", state="attached", timeout=5000)
            assert page.inner_text("#user-display-name") == "admin_user"

            # 3. Admin button is visible for admin user; open Admin Console & create invite
            assert not page.locator("#open-admin-btn").evaluate(
                "el => el.classList.contains('hidden')"
            )
            page.click("#open-admin-btn")
            page.wait_for_selector("#admin-modal:not(.hidden)")
            page.click("#admin-create-invite-btn")
            page.wait_for_selector("#admin-new-invite-box:not(.hidden)")
            invite_code = page.inner_text("#admin-new-invite-code").strip()
            assert len(invite_code) >= 6
            page.click("[data-close-modal='admin-modal']")

            # 4. Send a chat message & verify streaming + regenerate
            page.fill("#chat-input", "Hello E2E")
            page.click("#send-btn")
            page.wait_for_function(
                "() => document.body.innerText.includes('First reply from Grok E2E!')",
                timeout=5000,
            )

            # Click Regenerate
            page.click(".regen-btn")
            page.wait_for_function(
                "() => document.body.innerText.includes('Regenerated reply from Grok E2E!')",
                timeout=5000,
            )

            # 5. Open My Data & Account modal and verify export links
            page.click("#open-account-btn")
            page.wait_for_selector("#account-modal:not(.hidden)")
            assert (
                page.get_attribute("#export-account-json", "href")
                == "/api/account/export?format=json"
            )
            page.click("[data-close-modal='account-modal']")

            # 6. Open Personal API Keys (/v1) modal and create a gk_... key
            page.click("#open-api-keys-btn")
            page.wait_for_selector("#api-keys-modal:not(.hidden)")
            page.fill("#api-key-name-input", "E2E-Cursor")
            page.click("#api-key-create-btn")
            page.wait_for_selector("#new-api-key-box:not(.hidden)")
            api_key = page.inner_text("#new-api-key-value").strip()
            assert api_key.startswith("gk_")
            page.click("[data-close-modal='api-keys-modal']")

            # 7. Open Scheduled Tasks modal, create a daily task, and click Run Now
            page.click("#open-tasks-btn")
            page.wait_for_selector("#tasks-modal:not(.hidden)")
            page.fill("#task-title", "Morning AI Digest")
            page.fill("#task-prompt", "Summarize top AI news")
            page.click("#task-submit-btn")
            page.wait_for_selector("[data-run-task]")
            page.click("[data-run-task]")
            page.wait_for_function(
                "() => document.body.innerText.includes('Task digest reply from Grok E2E!')",
                timeout=5000,
            )

            # 8. Click "Preview in Canvas" on the HTML code block and verify sandboxed CSP iframe
            page.wait_for_selector(".canvas-open-btn")
            page.click(".canvas-open-btn")
            page.wait_for_selector("#canvas-pane:not(.hidden)")
            srcdoc = page.get_attribute("#canvas-iframe", "srcdoc") or ""
            assert "Content-Security-Policy" in srcdoc
            assert "<h1>Hello Grok Canvas</h1>" in srcdoc
            page.click("#canvas-tab-code")
            assert "<h1>Hello Grok Canvas</h1>" in page.inner_text("#canvas-code-view")
            page.click("#canvas-close-btn")
            page.wait_for_selector("#canvas-pane.hidden", state="attached")

            # 9. Verify Light default theme & Specialist Bot Roster switching
            assert page.get_attribute("html", "data-theme") == "light"
            page.click("[data-select-bot='code-architect']")
            assert page.inner_text("#active-bot-name") == "Code Architect"
            page.click("[data-select-bot='grok-main']")
            assert page.inner_text("#active-bot-name") == "T Bot"

            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=3)
