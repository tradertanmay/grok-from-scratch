"""Free Cloud GPU (Google Colab T4 / Kaggle) & NVIDIA QLoRA runner using Unsloth + GGUF export."""

import argparse
from pathlib import Path
from typing import Any

from .prepare_dataset import build_dataset
from .train_mac_mlx import generate_ollama_modelfile

DEFAULT_UNSLOTH_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"


def build_unsloth_config(
    model: str = DEFAULT_UNSLOTH_MODEL,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
    max_steps: int = 60,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    quantization_method: str = "q4_k_m",
) -> dict[str, Any]:
    """Return the QLoRA & GGUF export configuration for a free 16 GB Colab T4 or local GPU."""
    return {
        "model": model,
        "max_seq_length": max_seq_length,
        "load_in_4bit": True,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "max_steps": max_steps,
        "per_device_train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "quantization_method": quantization_method,
    }


def resolve_gguf_path(out_dir: str | Path, quant: str = "q4_k_m") -> Path:
    """Find the actual exported `*.gguf` file in `out_dir`, or return a fallback filename."""
    out = Path(out_dir)
    found = sorted(out.glob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if found:
        return found[0].resolve()
    return (out / f"unsloth.{quant.upper()}.gguf").resolve()


def write_gguf_modelfile(out_dir: str | Path, quant: str = "q4_k_m") -> Path:
    """Write `Modelfile` pointing to the discovered `*.gguf` file in `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gguf_file = resolve_gguf_path(out, quant)
    modelfile_path = out / "Modelfile"
    modelfile_path.write_text(generate_ollama_modelfile(str(gguf_file)), encoding="utf-8")
    return modelfile_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune Grok Persona QLoRA on a free Colab T4 / NVIDIA GPU and export GGUF"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_UNSLOTH_MODEL,
        help="4-bit Unsloth model ID (fits easily on a free 16 GB Colab T4 or 8 GB RTX GPU)",
    )
    parser.add_argument("--data-dir", default="training/data", help="Directory for SFT JSONL files")
    parser.add_argument("--out-dir", default="training/gguf", help="Output directory for GGUF")
    parser.add_argument("--steps", type=int, default=60, help="Number of QLoRA training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--quant", default="q4_k_m", help="GGUF quantization method for Ollama")
    parser.add_argument("--db", default=None, help="Optional grok_bot.sqlite3 to include in SFT")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare dataset, Modelfile, and print Unsloth config without running GPU training",
    )
    args = parser.parse_args(argv)

    counts = build_dataset(args.data_dir, db_path=args.db)
    cfg = build_unsloth_config(
        model=args.model,
        max_steps=args.steps,
        batch_size=args.batch_size,
        quantization_method=args.quant,
    )
    out_dir = Path(args.out_dir)
    modelfile_path = write_gguf_modelfile(out_dir, args.quant)

    print(
        f"[1/3] Dataset ready ({counts['train']} train, {counts['valid']} valid) in {args.data_dir}"
    )
    print(f"[2/3] Wrote Ollama GGUF Modelfile to {modelfile_path}")
    print(
        f"[3/3] Unsloth QLoRA Config: {cfg['model']} (r={cfg['lora_r']}, steps={cfg['max_steps']})"
    )

    if args.dry_run:
        return 0

    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError:
        print(
            "Note: `unsloth` is not installed locally. Run on a free Google Colab T4 GPU with:\n"
            "  pip install unsloth trl datasets\n"
            "  python -m training.train_unsloth"
        )
        return 0

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
    )
    ds = load_dataset("json", data_files=str(Path(args.data_dir) / "train.jsonl"), split="train")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(out_dir / "checkpoints"),
            max_steps=cfg["max_steps"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            learning_rate=cfg["learning_rate"],
        ),
    )
    trainer.train()
    model.save_pretrained_gguf(
        str(out_dir), tokenizer, quantization_method=cfg["quantization_method"]
    )
    write_gguf_modelfile(out_dir, args.quant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
