# Changelog

All notable changes to **Grok From Scratch** (`grok-bot`) are documented in this file.

## [v0.1.0](https://github.com/tradertanmay/grok-from-scratch/releases/tag/v0.1.0) — Initial Release

### Added
- **12-Chapter Engineering Book ([`docs/book/`](docs/book/README.md)):** Complete *"Building Grok Bot from Scratch"* book (`Ch 01`–`Ch 12`) with 65/65 mechanically verified source excerpts (`test_book_excerpts_match_source_files_verbatim`).
- **Any-Hardware 3-Tier Support & Universal QLoRA ([`training/train_unsloth.py`](training/train_unsloth.py), [`training/train_mac_mlx.py`](training/train_mac_mlx.py)):** Full support for any 8 GB–16 GB Windows/Linux/Mac laptop via free cloud APIs (Groq, Gemini, OpenRouter), SQLite `FTS5` zero-RAM RAG, compact 3B local Ollama models (`qwen2.5:3b`), and 4-bit QLoRA on a free Google Colab T4 GPU or Apple Silicon.
- **Multi-Provider Router & OpenAI Wire Protocol Client ([`grok_bot/llm.py`](grok_bot/llm.py), [`grok_bot/providers.py`](grok_bot/providers.py)):** Priority + health-cooldown routing and concurrency spillover across local and cloud providers.
- **ReAct Agent, Sandboxed Python & MCP Client ([`grok_bot/agent.py`](grok_bot/agent.py), [`grok_bot/mcp_client.py`](grok_bot/mcp_client.py)):** Parallel tool execution, `<tool_call>` fallback parser, SSRF-guarded web tools, Docker/local Python execution, and Model Context Protocol (`stdio` and Streamable HTTP) integration.
- **Personas, `<think>` Stream Parser & DeepSearch ([`grok_bot/persona.py`](grok_bot/persona.py), [`grok_bot/deepsearch.py`](grok_bot/deepsearch.py)):** Regular, Fun, and Unhinged personas, streaming `<think>` extraction, reasoning-model routing (`GROK_THINK_MODEL`), and multi-step citation-verified web research.
- **Hybrid RAG, Long-Term Memory & Vision ([`grok_bot/memory.py`](grok_bot/memory.py), [`grok_bot/files.py`](grok_bot/files.py)):** Cosine vector search + SQLite `FTS5` BM25 fallback, per-user memory tools, PDF/text chunking, and multimodal vision routing (`GROK_VISION_MODEL`).
- **Zero-Build Dark-Mode Web UI & Sandboxed Grok Studio Canvas ([`frontend/`](frontend/)):** Streaming SSE interface, collapsible `<think>` and tool cards, and interactive HTML/SVG/JS preview inside `<iframe sandbox="allow-scripts">` with strict inline `Content-Security-Policy`.
- **Voice Mode (STT & TTS) ([`grok_bot/stt.py`](grok_bot/stt.py), [`grok_bot/mlx_stt.py`](grok_bot/mlx_stt.py)):** `POST /api/transcribe` proxy for Groq Whisper / local Whisper plus browser Web Speech API fallback.
- **Multi-User Auth, OpenAI `/v1` Gateway, Evals & Observability ([`grok_bot/auth.py`](grok_bot/auth.py), [`grok_bot/openai_compat.py`](grok_bot/openai_compat.py), [`grok_bot/evals`](grok_bot/evals)):** `scrypt` auth, invite codes, daily quotas, scheduled tasks, personal `/v1` API keys, account ZIP export, `/metrics` Prometheus endpoint, 20-case evaluation suite, and async SSE load tester.
