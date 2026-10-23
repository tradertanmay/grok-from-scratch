"""Online SQLite backups: `python -m grok_bot.backup [--dir backups] [--keep 7] [--every 24]`.

Everything (users, chats, memories, uploaded files, usage) lives in the one SQLite
database, so a single consistent copy is a full backup. Uses SQLite's online backup
API, so it is safe while the app is running. Restore with the app stopped:
`python -m grok_bot.backup --restore backups/grok-....sqlite3.gz`.
"""

import argparse
import gzip
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings

PREFIX = "grok-"
SUFFIX = ".sqlite3.gz"


def _check(path: Path) -> None:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()
    if result != "ok":
        raise RuntimeError(f"integrity check failed for {path}: {result}")


def backup(db: str | Path, out_dir: str | Path, keep: int = 7) -> Path:
    db, out_dir = Path(db), Path(out_dir)
    if not db.exists():
        raise FileNotFoundError(db)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = out_dir / f"{PREFIX}{stamp}{SUFFIX}"
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        raw = Path(tmp) / "copy.sqlite3"
        src, dst = sqlite3.connect(db), sqlite3.connect(raw)
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        _check(raw)
        partial = Path(tmp) / target.name
        with raw.open("rb") as f, gzip.open(partial, "wb", compresslevel=6) as g:
            shutil.copyfileobj(f, g)
        os.replace(partial, target)
    prune(out_dir, keep)
    return target


def prune(out_dir: Path, keep: int) -> list[Path]:
    if keep <= 0:
        return []
    old = sorted(out_dir.glob(f"{PREFIX}*{SUFFIX}"))[:-keep]
    for p in old:
        p.unlink()
    return old


def restore(archive: str | Path, db: str | Path) -> None:
    archive, db = Path(archive), Path(db)
    with tempfile.TemporaryDirectory(dir=db.parent if db.parent.exists() else None) as tmp:
        raw = Path(tmp) / "restore.sqlite3"
        opener = gzip.open if archive.suffix == ".gz" else open
        with opener(archive, "rb") as f, raw.open("wb") as g:
            shutil.copyfileobj(f, g)
        _check(raw)
        if db.exists():
            os.replace(db, db.with_name(db.name + ".before-restore"))
        for side in ("-wal", "-shm"):
            Path(str(db) + side).unlink(missing_ok=True)
        shutil.move(raw, db)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m grok_bot.backup", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--db", default=Settings.from_env().db_path, help="database (GROK_DB_PATH)")
    ap.add_argument("--dir", default=os.environ.get("GROK_BACKUP_DIR", "backups"))
    ap.add_argument(
        "--keep",
        type=int,
        default=int(os.environ.get("GROK_BACKUP_KEEP", "7")),
        help="newest backups to keep (0 = keep all)",
    )
    ap.add_argument(
        "--every", type=float, default=0, help="hours between backups; 0 = run once and exit"
    )
    ap.add_argument(
        "--restore", metavar="ARCHIVE", help="replace --db with ARCHIVE (stop the app first)"
    )
    args = ap.parse_args(argv)
    if args.restore:
        restore(args.restore, args.db)
        print(f"restored {args.restore} -> {args.db} (old copy: {args.db}.before-restore)")
        return 0
    while True:
        try:
            print(f"backup written: {backup(args.db, args.dir, args.keep)}", flush=True)
        except Exception as e:
            print(f"backup failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            if not args.every:
                return 1
        if not args.every:
            return 0
        time.sleep(args.every * 3600)


if __name__ == "__main__":
    raise SystemExit(main())
