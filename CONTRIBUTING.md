# Contributing to Grok Bot

Thank you for contributing to **Grok Bot** and the companion 12-chapter engineering book, [***Building Grok Bot from Scratch***](docs/book/README.md).

---

## 1. Local Development Setup

Grok Bot requires **Python 3.11+** and works on Linux, macOS, and Windows (no GPU required for development or tests):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,mcp]"
```

---

## 2. Mandatory Pre-PR Verification (No GitHub Actions)

GitHub Actions is intentionally disabled on this repository. Every contributor **must** run the full lint, format, and test suite locally before opening or merging a Pull Request:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest -q
```

Or via `make`:

```bash
make check
```

---

## 3. Keeping Book Excerpts in Sync (`docs/book/`)

Every code excerpt in [`docs/book/ch01-*.md`](docs/book/README.md) through [`ch12-*.md`](docs/book/README.md) labeled `` `path` (lines a–b): `` is mechanically verified against lines `a..b` of `path` by `test_book_excerpts_match_source_files_verbatim` in [`tests/test_mlx_llm.py`](tests/test_mlx_llm.py).

If your Pull Request edits or shifts lines in a file quoted by the book:
1. Run `.venv/bin/pytest tests/test_mlx_llm.py -k "book_excerpts" -v` to see which chapter and line range shifted.
2. Update the `(lines a–b)` label and verbatim code block in the corresponding `docs/book/chXX-*.md` file in the same commit.

---

## 4. Pull Request Workflow

1. Create a feature branch from `main`.
2. Add unit or integration tests in `tests/` using `FakeLLM` (`tests/conftest.py`) so tests run offline and deterministically.
3. Run `make check` (`ruff check . && ruff format --check . && pytest -q`) and include the output in your PR description.
