import gzip
import sqlite3

import pytest

from grok_bot.backup import backup, main, prune, restore
from grok_bot.store import Store


def make_db(path):
    store = Store(str(path))
    user = store.create_user("ann", "hash")
    sid = store.create_session(user_id=user["id"])
    store.add_message(sid, {"role": "user", "content": "keep me"})
    return store


def contents(path):
    conn = sqlite3.connect(path)
    try:
        return [r[0] for r in conn.execute("SELECT content FROM messages")]
    finally:
        conn.close()


def test_backup_while_open_and_restore(tmp_path):
    db = tmp_path / "live.sqlite3"
    store = make_db(db)
    archive = backup(db, tmp_path / "bk")
    assert archive.name.startswith("grok-") and archive.name.endswith(".sqlite3.gz")
    raw = tmp_path / "raw.sqlite3"
    raw.write_bytes(gzip.decompress(archive.read_bytes()))
    assert contents(raw) == ["keep me"]

    store._conn.execute("DELETE FROM messages")
    store._conn.commit()
    store._conn.close()
    restore(archive, db)
    assert contents(db) == ["keep me"]
    assert (tmp_path / "live.sqlite3.before-restore").exists()


def test_prune_keeps_newest(tmp_path):
    db = tmp_path / "live.sqlite3"
    make_db(db)
    made = [backup(db, tmp_path / "bk", keep=0) for _ in range(4)]
    assert prune(tmp_path / "bk", 2) == made[:2]
    assert sorted((tmp_path / "bk").iterdir()) == made[2:]


def test_restore_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.sqlite3.gz"
    bad.write_bytes(gzip.compress(b"not a database" * 100))
    db = tmp_path / "live.sqlite3"
    db.write_bytes(b"original")
    with pytest.raises(sqlite3.DatabaseError):
        restore(bad, db)
    assert db.read_bytes() == b"original"


def test_cli_once(tmp_path, capsys):
    db = tmp_path / "live.sqlite3"
    make_db(db)
    assert main(["--db", str(db), "--dir", str(tmp_path / "bk"), "--keep", "3"]) == 0
    assert "backup written" in capsys.readouterr().out
    assert main(["--db", str(tmp_path / "missing.sqlite3"), "--dir", str(tmp_path)]) == 1
