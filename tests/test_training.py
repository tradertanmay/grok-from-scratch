import json

from grok_bot.store import Store
from training.prepare_dataset import build_dataset, extract_db_examples, load_seed_examples
from training.train_mac_mlx import build_mlx_lora_cmd, generate_ollama_modelfile, main
from training.train_unsloth import build_unsloth_config
from training.train_unsloth import main as unsloth_main


def test_seed_and_db_dataset_builder(tmp_path):
    seeds = load_seed_examples()
    assert len(seeds) >= 10
    assert all("messages" in x for x in seeds)

    db_file = tmp_path / "chats.db"
    store = Store(str(db_file))
    sid = store.create_session("Training sample")
    store.add_message(sid, {"role": "user", "content": "Explain TCP vs UDP briefly."})
    store.add_message(
        sid,
        {
            "role": "assistant",
            "content": (
                "TCP guarantees ordered, reliable delivery via handshakes; "
                "UDP fires packets with minimal latency."
            ),
        },
    )

    extracted = extract_db_examples(db_file)
    assert len(extracted) == 1
    assert extracted[0]["messages"][1]["role"] == "user"
    assert extracted[0]["messages"][2]["role"] == "assistant"

    out_dir = tmp_path / "data"
    counts = build_dataset(out_dir, db_path=db_file)
    assert counts["total"] == len(seeds) + 1
    assert (out_dir / "train.jsonl").is_file()
    assert (out_dir / "valid.jsonl").is_file()

    first_row = json.loads((out_dir / "train.jsonl").read_text().splitlines()[0])
    assert first_row["messages"][0]["role"] == "system"


def test_mlx_cmd_and_modelfile_generation(tmp_path):
    mf = generate_ollama_modelfile("qwen3.6:35b", adapter_path=tmp_path / "adapters")
    assert "FROM qwen3.6:35b" in mf
    assert "ADAPTER " in mf
    assert "PARAMETER temperature" in mf
    assert "Current date and time (UTC):" not in mf

    cmd = build_mlx_lora_cmd("mlx-community/Qwen2.5-7B-Instruct-4bit", tmp_path, tmp_path / "out")
    assert "mlx_lm.lora" in cmd
    assert "--train" in cmd

    rc = main(["--data-dir", str(tmp_path / "d"), "--dry-run"])
    assert rc == 0
    assert (tmp_path / "Modelfile").is_file()


def test_unsloth_config_and_dry_run(tmp_path):
    from training.train_unsloth import resolve_gguf_path, write_gguf_modelfile

    cfg = build_unsloth_config(max_steps=30, quantization_method="q4_k_m")
    assert cfg["load_in_4bit"] is True
    assert cfg["max_steps"] == 30
    assert cfg["quantization_method"] == "q4_k_m"

    gguf_dir = tmp_path / "gguf"
    rc = unsloth_main(
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--out-dir",
            str(gguf_dir),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert (gguf_dir / "Modelfile").is_file()

    # Simulate Unsloth's save_pretrained_gguf writing a custom-named *.gguf file
    fake_gguf = gguf_dir / "Qwen2.5-3B-Instruct.Q4_K_M.gguf"
    fake_gguf.write_bytes(b"GGUF\x00\x00\x00\x03")
    assert resolve_gguf_path(gguf_dir, "q4_k_m") == fake_gguf.resolve()
    mf_path = write_gguf_modelfile(gguf_dir, "q4_k_m")
    assert f"FROM {fake_gguf.resolve()}" in mf_path.read_text(encoding="utf-8")
