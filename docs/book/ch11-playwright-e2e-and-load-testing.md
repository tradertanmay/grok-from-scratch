# Chapter 11 — Full-Stack Verification — Headless Chrome Playwright E2E & 20-User Concurrent Load Testing

> **Source Files Covered:** [`tests/conftest.py`](../../tests/conftest.py) · [`tests/test_frontend.py`](../../tests/test_frontend.py) · [`grok_bot/loadtest.py`](../../grok_bot/loadtest.py) · [`tests/test_loadtest.py`](../../tests/test_loadtest.py)

How do you ship a multi-user streaming AI platform with **100% local pre-commit verification** and still maintain 100% confidence that every commit works across the database, agent loop, browser DOM, and 20 simultaneous streaming users?

You build a three-tier local verification pyramid that runs in **under 12 seconds** on any laptop (Windows, Linux, or macOS) with zero cloud API keys required:
1. **Deterministic Unit & Integration Tests (`FakeLLM` in `tests/conftest.py`)**
2. **Real Headless Browser E2E Verification (`Playwright` in `tests/test_frontend.py`)**
3. **Concurrent Multi-User SSE Load Testing (`python -m grok_bot.loadtest` in `grok_bot/loadtest.py`)**

---

## 1. Real Chromium Browser E2E with Playwright (`tests/test_frontend.py`)

Unit tests verify JSON endpoints, but they cannot catch a broken CSS selector, a missing `id` in `frontend/index.html`, or an unhandled promise rejection in `frontend/app.js`.

In `tests/test_frontend.py`, `test_playwright_browser_e2e` spins up a real `uvicorn` server on an ephemeral localhost port in a background thread and drives headless Chrome through the complete user journey. Here are steps 7 and 8:

`tests/test_frontend.py` (lines 275–298):

```python
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
```

---

## 2. Stress-Testing 20 Concurrent Users (`grok_bot/loadtest.py`)

To measure real-world latency and verify concurrency spillover, we built [`grok_bot/loadtest.py`](../../grok_bot/loadtest.py) (`python -m grok_bot.loadtest`).

Look at `_request_with_retry_after`:

`grok_bot/loadtest.py` (lines 51–71):

```python
async def _request_with_retry_after(
    http: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    max_retries: int = 8,
    **kwargs: Any,
) -> httpx.Response:
    """Send an HTTP request, honoring `Retry-After` on HTTP 429."""
    for attempt in range(max_retries + 1):
        resp = await http.request(method, url, **kwargs)
        if resp.status_code != 429 or attempt == max_retries:
            return resp
        retry_hdr = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
        try:
            wait_s = max(float(retry_hdr), 0.05) if retry_hdr is not None else 1.0
        except ValueError:
            wait_s = 1.0
        await sleep_fn(wait_s)
    return resp
```

### Four Edge Cases Handled by `grok_bot/loadtest.py`
1. **Rate-Limit Backoff (`Retry-After`):** Signup is rate-limited at 10/min per IP (`auth_limiter`). When creating 20 virtual users, `_request_with_retry_after` honors `Retry-After` automatically instead of failing on user #11.
2. **Invite/Closed Signup Support (`--keys-file`):** On production servers running `GROK_SIGNUP=invite` or `closed`, operators pass `--keys-file keys.txt` with pre-minted `gk_...` bearer keys.
3. **Empty-DB Sole-Admin Protection:** If run against a brand-new database with 0 users, user #0 would become the sole admin and `DELETE /api/account` would refuse to delete it (`400 Cannot delete the last admin`), leaving the database locked behind a random password. `run_load_test()` detects `is_admin=True` immediately and aborts with a clear message including the generated credentials.
4. **Mid-Stream SSE Error & Empty-Reply Detection:** Inspects every `data:` frame for `{"error": ...}` or `{"type": "error"}` and marks 0-character completions as failed turns.

---

## 3. Real 20-User Benchmark: Mac Ollama Alone vs. Concurrency Spillover

Here are the actual measurements from running `python -m grok_bot.loadtest --users 20` against a 48 GB Apple Silicon workstation (`gpt-oss:latest` with `OLLAMA_NUM_PARALLEL=4`):

| Configuration | Success Rate | `p50` TTFT | `p95` TTFT | Total Wall Time | Throughput | Provider Attribution (`/metrics`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Mac Ollama Only** | `20/20 (100%)` | `60.45s` | `135.29s` | `136.67s` | `8.8 turns/min` | `mac/gpt-oss:latest: 20` (5 queued waves of 4) |
| **Mac Primary (`max_concurrent: 4`) + Spillover** | `20/20 (100%)` | **`0.23s`** | **`0.61s`** | **`14.95s`** | **`80.2 turns/min`** | `mac/gpt-oss:latest: 4`, `cloud/llama-3.3-70b-versatile: 16` |

Capping primary Mac concurrency at `max_concurrent: 4` (`GROK_MAX_CONCURRENT_PRIMARY=4`) and spilling burst requests 5–20 to the fallback provider cut median Time-to-First-Token by **260×** (`60.45s` → `0.23s`) and increased throughput by **9.1×**.

---

## 4. Try it

1. **Run the load-test unit suite (`tests/test_loadtest.py`):**
   ```bash
   .venv/bin/pytest tests/test_loadtest.py -v
   ```
   Real output:
   ```text
   tests/test_loadtest.py::test_loadtest_20_concurrent_users_signup_stream_and_cleanup PASSED [ 16%]
   tests/test_loadtest.py::test_loadtest_keys_file_mode_works_when_signup_closed PASSED [ 33%]
   tests/test_loadtest.py::test_loadtest_detects_midstream_sse_error_and_empty_reply PASSED [ 50%]
   tests/test_loadtest.py::test_loadtest_aborts_on_empty_db_sole_admin PASSED [ 66%]
   tests/test_loadtest.py::test_loadtest_cli_entrypoint_json PASSED         [ 83%]
   tests/test_loadtest.py::test_load_keys_file_validation PASSED            [100%]
   ============================== 6 passed in 0.91s ===============================
   ```

2. **Run the full repository lint, format, and 145-test verification suite:**
   ```bash
   .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest -q
   ```
   Real output:
   ```text
   All checks passed!
   69 files already formatted
   ........................................................................ [ 49%]
   ........................................................................ [ 99%]
   .                                                                        [100%]
   145 passed in 8.49s
   ```

---

## 5. Exercises

1. **Multi-Turn Session Load Test:** Add a `--session-reuse` flag to `grok_bot/loadtest.py` that sends `--requests-per-user 3` sequential turns against `POST /api/chat` using the same `session_id` to benchmark growing context-window latency.
2. **CSV / Markdown Benchmark Exporter:** Add `--output-csv report.csv` to `grok_bot/loadtest.py` so latency percentiles across different `max_concurrent` values (`2`, `4`, `8`) can be plotted directly.

---

[← Chapter 10: Persona Fine-Tuning with LoRA (MLX & Unsloth)](ch10-lora-finetuning-mlx-and-unsloth.md) · [Book Index](README.md) · [Chapter 12: The Zero-H100 Production Blueprint →](ch12-zero-h100-production-blueprint.md)
