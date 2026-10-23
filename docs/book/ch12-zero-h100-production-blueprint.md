# Chapter 12 — The Zero-H100 Production Blueprint (Any 8 GB Laptop, Free Cloud Tier, or Local Hardware)

> **Source Files Covered:** [`providers.example.json`](../../providers.example.json) · [`mcp.example.json`](../../mcp.example.json) · [`grok_bot/config.py`](../../grok_bot/config.py) · [`grok_bot/providers.py`](../../grok_bot/providers.py) · [`compose.yaml`](../../compose.yaml) · [`Caddyfile`](../../Caddyfile)

Every team building an open-weight AI assistant faces a fundamental question:
> ***"not mvp, actual product,, and what will hapen if we do not have. h100"***

And just as importantly: **most developers do not have a high-end 48 GB MacBook or an expensive gaming PC either.** Most people are working on an **8 GB or 16 GB Windows, Linux, or Mac laptop**, a Chromebook, or a **\$5/month VPS**.

Because we built Grok Bot around the OpenAI wire protocol (`grok_bot/llm.py`), multi-provider spillover (`grok_bot/providers.py`), and automatic SQLite `FTS5` fallbacks (`grok_bot/memory.py`, `grok_bot/files.py`), the exact same codebase runs in production across **three zero-H100 hardware blueprints**:

---

## 1. Choose Your Hardware Blueprint

### Blueprint A — Any 8 GB Laptop / Windows / Linux / \$5 VPS (\$0 Free Cloud APIs, `0 GB` Local VRAM)
If your laptop has 8 GB of RAM and no dedicated GPU, you don't need to download a single gigabyte of model weights:
- **Chat & Reasoning (`Think`):** Free API key from **Groq** (`groq/openai/gpt-oss-120b`, `groq/openai/gpt-oss-20b`, `groq/qwen/qwen3.8-27b`, or `groq/llama-3.3-70b-versatile`) or **OpenRouter** (`openrouter/...`).
- **Multimodal Vision:** Free API key from **Google AI Studio / Gemini** (`gemini/gemini-3.8-flash` or `gemini/gemini-3.5-flash-lite`) or **Groq** (`groq/qwen/qwen3.8-27b`).
- **Long-Term Memory & Document RAG:** Set `GROK_EMBED_MODEL=""` — `grok_bot/memory.py` and `grok_bot/files.py` automatically use **SQLite FTS5 BM25 keyword search** (`0 MB` of RAM/VRAM).
- **Voice STT & TTS:** Set `GROK_STT_URL=https://api.groq.com/openai/v1`, `GROK_STT_KEY=$GROQ_API_KEY`, and `GROK_STT_MODEL=whisper-large-v3-turbo` (or leave unset to use browser `SpeechRecognition` + `speechSynthesis`).
- **LoRA Fine-Tuning:** Run `python -m training.train_unsloth` on a **free Google Colab T4 GPU**.

### Blueprint B — Everyday 8 GB–16 GB Laptop (Windows / Linux / Base Mac with Local Ollama)
If you want private offline inference on a standard 8 GB–16 GB laptop, run compact 4-bit models in Ollama (`~2–5 GB` RAM total) and let `ProviderRouter` spill heavy `Think` turns or multi-user traffic over to Groq/Gemini:

| Role | Env Variable | Model for 8 GB–16 GB Laptop | RAM / VRAM Footprint |
| :--- | :--- | :--- | :---: |
| **Fast Chat & Tools** | `LLM_MODEL` | `qwen2.5:3b` (or `llama3.2:3b`) | ~1.9 GB |
| **Multimodal Vision** | `GROK_VISION_MODEL` | `gemma3:4b` (or `gemini/gemini-2.5-flash`) | ~3.1 GB (on-demand) |
| **RAG Embeddings** | `GROK_EMBED_MODEL` | `nomic-embed-text:latest` | ~0.3 GB |
| **Deep Reasoning (`Think`)** | `GROK_THINK_MODEL` | `deepseek-r1:7b` locally, or `groq/qwen/qwen3-32b` | ~4.7 GB (or `0 GB` cloud) |

### Blueprint C — High-RAM Workstation / Local GPU Server (32 GB–48 GB+ Unified or Discrete RAM)
If you have a desktop GPU box or a high-memory machine (such as the 48 GB workstation used for our 20-user stress tests in Chapter 11), you can run 20B–35B models locally and still use `ProviderRouter` to burst to cloud APIs when 20+ users arrive simultaneously:

| Role | Env Variable / Service | Local Workstation Model | Memory Footprint | Verified Benchmark |
| :--- | :--- | :--- | :---: | :--- |
| **Fast Chat & Tools** | `LLM_MODEL` | `mac/gpt-oss:latest` (20B) | ~13.0 GB | `8.8 turns/min` solo; `80.2 turns/min` with spillover |
| **Deep Reasoning (`Think`)** | `GROK_THINK_MODEL` | `mac/qwen3.6:35b` (35B MoE, 3B active) | ~22.0 GB (on-demand) | Streams native `<think>` traces |
| **Multimodal Vision** | `GROK_VISION_MODEL` | `mac/gemma4:12b` | ~7.8 GB (on-demand) | 100% pass on inline PNG vision eval |
| **RAG Embeddings** | `GROK_EMBED_MODEL` | `mac/nomic-embed-text:latest` | ~0.3 GB | 768-dim cosine recall in SQLite |

---

## 2. Production Multi-Provider Configuration (`providers.example.json`)

Here is the complete `providers.example.json` wiring local Ollama, optional local LoRA serving (`mlx`), and free/low-cost cloud providers (`groq` / `openrouter` / `gemini`):

`providers.example.json` (lines 1–14):

```json
[
  {"name": "mac", "base_url": "http://localhost:11434/v1",
   "models": ["gpt-oss:latest", "qwen3.6:35b", "gemma4:12b"], "max_concurrent": 4},
  {"name": "mlx", "base_url": "http://127.0.0.1:8088/v1",
   "models": ["grok-lora"], "max_concurrent": 1},
  {"name": "h100", "base_url": "http://H100_HOST:8000/v1",
   "models": ["openai/gpt-oss-120b"]},
  {"name": "groq", "base_url": "https://api.groq.com/openai/v1", "api_key": "$GROQ_API_KEY",
   "models": []},
  {"name": "openrouter", "base_url": "https://openrouter.ai/api/v1",
   "api_key": "$OPENROUTER_API_KEY", "models": []},
  {"name": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
   "api_key": "$GEMINI_API_KEY", "models": []}
]
```

Combined with `GROK_MAX_CONCURRENT_PRIMARY=4` and `GROK_CONNECT_TIMEOUT=5.0` (`grok_bot/config.py`), this configuration guarantees:
- **Works With or Without Local Hardware:** If you aren't running Ollama locally (or your laptop is asleep), `ProviderRouter` fails over in `5.0s`, marks the offline provider down for `30s`, and serves requests directly from `groq` / `openrouter` / `gemini` with `0ms` failover penalty on subsequent turns.
- **Instant Burst Spillover (`0.23s` p50 TTFT):** When local `max_concurrent` slots are full, burst requests immediately spill over to the cloud pool without waiting in a local GPU queue.

---

## 3. Try it

### Option 1: Run on Any 8 GB Laptop with Zero Local GPU (Free Groq API + SQLite FTS5 RAG)
```bash
export GROQ_API_KEY="gsk_..."
LLM_BASE_URL=https://api.groq.com/openai/v1 \
LLM_API_KEY=$GROQ_API_KEY \
LLM_MODEL=openai/gpt-oss-120b \
GROK_EMBED_MODEL="" \
GROK_STT_URL=https://api.groq.com/openai/v1 \
GROK_STT_KEY=$GROQ_API_KEY \
GROK_STT_MODEL=whisper-large-v3-turbo \
.venv/bin/uvicorn grok_bot.app:create_app --factory --host 127.0.0.1 --port 8000
```

### Option 2: Run on an 8 GB–16 GB Laptop with Compact Local Ollama (`qwen3:4b`)
```bash
ollama pull qwen2.5:3b && ollama pull nomic-embed-text
LLM_BASE_URL=http://localhost:11434/v1 \
LLM_MODEL=qwen2.5:3b \
GROK_EMBED_MODEL=nomic-embed-text \
.venv/bin/uvicorn grok_bot.app:create_app --factory --host 127.0.0.1 --port 8000
```

### Option 3: Verify `/api/health`
```bash
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool
```

Real output:
```json
{
    "status": "ok",
    "provider_ok": true,
    "model": "mac/gpt-oss:latest",
    "think_model": "mac/qwen3.6:35b",
    "vision_model": "mac/gemma4:12b",
    "base_url": "http://localhost:11434/v1",
    "embedding": "mac/nomic-embed-text:latest",
    "auth": true,
    "signup": "open",
    "daily_limit": null,
    "stt": true
}
```

---

## 4. Exercises

1. **8 GB Laptop Preset Flag (`--preset cloud-free` / `--preset laptop-8gb`):** Add a `--preset` CLI flag in `grok_bot/__main__.py` that pre-populates `Settings` for `cloud-free` (Groq + FTS5) or `laptop-8gb` (`qwen3:4b` + `nomic-embed-text`) so new users don't even have to edit `.env`.
2. **Dynamic Spillover Threshold:** Extend `ProviderRouter` in `grok_bot/providers.py` to track a rolling p95 TTFT window on the primary provider and dynamically lower `max_concurrent` when latency exceeds 10 seconds.

---

[← Chapter 11: Playwright Browser E2E & 20-User Load Testing](ch11-playwright-e2e-and-load-testing.md) · [Book Index](README.md)
