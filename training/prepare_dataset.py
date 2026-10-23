"""Prepare SFT JSONL datasets (`train.jsonl`, `valid.jsonl`) for Apple Silicon MLX LoRA or TRL."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from grok_bot.persona import system_prompt

DEFAULT_SEED = Path(__file__).resolve().parent / "seed_persona.jsonl"


def load_seed_examples(seed_path: Path = DEFAULT_SEED) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if not seed_path.is_file():
        return examples
    for line in seed_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
            examples.append({"messages": obj["messages"]})
    return examples


def extract_db_examples(db_path: str | Path, min_assistant_chars: int = 40) -> list[dict[str, Any]]:
    """Extract clean user->assistant dialogue turns from a Grok Bot SQLite database."""
    p = Path(db_path)
    if not p.is_file():
        return []
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        sessions = conn.execute("SELECT id FROM sessions ORDER BY created_at ASC").fetchall()
        examples: list[dict[str, Any]] = []
        sys_msg = {"role": "system", "content": system_prompt("fun")}
        for s in sessions:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (s["id"],),
            ).fetchall()
            convo: list[dict[str, str]] = [sys_msg]
            has_valid_assistant = False
            for r in rows:
                role = r["role"]
                content = (r["content"] or "").strip()
                if role == "user" and content:
                    convo.append({"role": "user", "content": content})
                elif role == "assistant" and len(content) >= min_assistant_chars:
                    convo.append({"role": "assistant", "content": content})
                    has_valid_assistant = True
            if has_valid_assistant and len(convo) >= 3:
                examples.append({"messages": convo})
        return examples
    finally:
        conn.close()


def build_dataset(
    output_dir: str | Path,
    seed_path: Path = DEFAULT_SEED,
    db_path: str | Path | None = None,
    valid_ratio: float = 0.2,
) -> dict[str, int]:
    """Write `train.jsonl` and `valid.jsonl` in MLX-LM / HuggingFace ChatML format."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_examples = load_seed_examples(seed_path)
    if db_path:
        all_examples.extend(extract_db_examples(db_path))

    if not all_examples:
        raise ValueError("No training examples found.")

    n_valid = max(1, int(len(all_examples) * valid_ratio)) if len(all_examples) > 1 else 1
    train_items = all_examples[:-n_valid] if len(all_examples) > 1 else all_examples
    valid_items = all_examples[-n_valid:]

    train_file = out / "train.jsonl"
    valid_file = out / "valid.jsonl"

    train_file.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in train_items) + "\n",
        encoding="utf-8",
    )
    valid_file.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in valid_items) + "\n",
        encoding="utf-8",
    )
    return {"total": len(all_examples), "train": len(train_items), "valid": len(valid_items)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Grok Persona LoRA SFT dataset")
    parser.add_argument(
        "--out", default="training/data", help="Output directory for train/valid.jsonl"
    )
    parser.add_argument(
        "--db", default=None, help="Optional path to grok_bot.sqlite3 to include chats"
    )
    args = parser.parse_args(argv)

    counts = build_dataset(args.out, db_path=args.db)
    print(f"Prepared dataset in {args.out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
