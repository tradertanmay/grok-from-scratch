# Persona Customization & QLoRA Fine-Tuning (Works on Any Hardware)

You **do not** need an H100 or a high-end MacBook to customize or fine-tune Grok Bot. Pick the path that matches your hardware:

1. **Path A — Any 8 GB Laptop (Zero Training / Instant Ollama Modelfile):** Bake Grok's persona and temperature into any small Ollama model (`qwen3:1.7b`, `qwen3:4b`, `qwen3.5:4b`, `llama3.2:3b`) in 2 seconds.
2. **Path B — Free Cloud GPU (Google Colab T4 / Kaggle) or Any NVIDIA PC (`training/train_unsloth.py`):** Run 4-bit QLoRA with Unsloth on a free 16 GB Colab T4 GPU and export a quantized `q4_k_m` GGUF that runs in ~2 GB RAM on any laptop.
3. **Path C — Apple Silicon Mac (`training/train_mac_mlx.py`):** Run 4-bit QLoRA locally with `mlx-lm` and serve via `python -m grok_bot.mlx_llm`.

---

## 1. Prepare SFT Dataset (`train.jsonl` & `valid.jsonl`)

Combines the curated seed dataset (`training/seed_persona.jsonl`) with any high-quality conversations from your local `grok_bot.sqlite3`:

```bash
python -m training.prepare_dataset --out training/data --db grok_bot.sqlite3
```

## 2A. Instant Custom Model on Any 8 GB Laptop (Ollama Modelfile)

```bash
python -m training.train_mac_mlx --ollama-base qwen3:4b --dry-run
ollama create grok-persona -f training/Modelfile
```

## 2B. Fine-Tune on a Free Google Colab T4 or NVIDIA GPU (`Unsloth` + GGUF Export)

```bash
pip install unsloth trl datasets
python -m training.train_unsloth --model unsloth/Qwen2.5-3B-Instruct-bnb-4bit --steps 60
ollama create grok-persona -f training/gguf/Modelfile
```

## 2C. Fine-Tune Locally on macOS (`mlx-lm`)

```bash
uv pip install mlx-lm
python -m training.train_mac_mlx --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --iters 100 --batch-size 1
```
