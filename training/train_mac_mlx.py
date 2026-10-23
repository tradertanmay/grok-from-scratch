"""Local Apple Silicon MLX LoRA fine-tuning runner + Ollama Modelfile exporter."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from grok_bot.persona import system_prompt

from .prepare_dataset import build_dataset


def generate_ollama_modelfile(
    base_or_fused_model: str,
    adapter_path: str | Path | None = None,
    temperature: float = 0.7,
) -> str:
    """Create an Ollama Modelfile with the Grok persona and optional LoRA adapter."""
    sys_text = "\n".join(
        ln
        for ln in system_prompt("fun").splitlines()
        if not ln.startswith("Current date and time (UTC):")
    ).strip()
    lines = [
        f"FROM {base_or_fused_model}",
    ]
    if adapter_path:
        lines.append(f"ADAPTER {Path(adapter_path).resolve()}")
    lines.extend(
        [
            f"PARAMETER temperature {temperature}",
            'SYSTEM """',
            sys_text,
            '"""',
        ]
    )
    return "\n".join(lines) + "\n"


def build_mlx_lora_cmd(
    model: str,
    data_dir: str | Path,
    adapter_path: str | Path,
    iters: int = 200,
    batch_size: int = 2,
    lora_layers: int = 8,
    learning_rate: float = 1e-4,
) -> list[str]:
    """Construct the `mlx_lm.lora` command for Apple Silicon unified memory."""
    return [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_path),
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--num-layers",
        str(lora_layers),
        "--learning-rate",
        str(learning_rate),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune Grok Persona LoRA on Apple Silicon (MLX) or export Modelfile"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        help="HuggingFace / MLX model ID for local Mac LoRA fine-tuning",
    )
    parser.add_argument(
        "--ollama-base",
        default="qwen3.6:35b",
        help="Base Ollama model for Modelfile generation",
    )
    parser.add_argument(
        "--data-dir", default="training/data", help="Directory for train/valid.jsonl"
    )
    parser.add_argument(
        "--adapter-dir", default="training/adapters", help="Output LoRA adapter path"
    )
    parser.add_argument("--iters", type=int, default=200, help="Number of LoRA training iterations")
    parser.add_argument("--batch-size", type=int, default=2, help="Minibatch size")
    parser.add_argument(
        "--db", default=None, help="Optional grok_bot.sqlite3 to include in SFT data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare dataset and Modelfile and print MLX command without executing training",
    )
    args = parser.parse_args(argv)

    counts = build_dataset(args.data_dir, db_path=args.db)
    print(
        f"[1/3] Dataset ready ({counts['train']} train, {counts['valid']} valid) in {args.data_dir}"
    )

    modelfile_path = Path(args.data_dir).parent / "Modelfile"
    modelfile_content = generate_ollama_modelfile(args.ollama_base)
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    print(f"[2/3] Wrote Ollama Modelfile to {modelfile_path}")

    cmd = build_mlx_lora_cmd(
        model=args.model,
        data_dir=args.data_dir,
        adapter_path=args.adapter_dir,
        iters=args.iters,
        batch_size=args.batch_size,
    )
    print(f"[3/3] MLX LoRA Command: {' '.join(cmd)}")

    if args.dry_run:
        return 0

    if shutil.which("mlx_lm.lora") is None:
        try:
            __import__("mlx_lm")
        except ImportError:
            print(
                "Note: `mlx-lm` is not installed in this venv. Install on Apple Silicon with:\n"
                "  uv pip install mlx-lm\n"
                "Or create the custom Ollama persona model immediately with:\n"
                f"  ollama create grok-custom -f {modelfile_path}"
            )
            return 0

    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
