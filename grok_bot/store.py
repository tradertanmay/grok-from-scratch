"""SQLite persistence for chat sessions and messages."""

import hashlib
import json
import re
import secrets
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'local'
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    name TEXT,
    reasoning TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    is_admin INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    prefix TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    prompt TEXT NOT NULL,
    schedule TEXT NOT NULL,
    at TEXT,
    time TEXT,
    weekday INTEGER,
    mode TEXT NOT NULL DEFAULT 'fun',
    think INTEGER NOT NULL DEFAULT 0,
    deepsearch INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    next_run_at TEXT,
    last_run_at TEXT,
    last_status TEXT,
    last_session_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(enabled, next_run_at);
CREATE TABLE IF NOT EXISTS invites (
    id TEXT PRIMARY KEY,
    code_hash TEXT NOT NULL UNIQUE,
    created_by TEXT REFERENCES users(id) ON DELETE SET NULL,
    uses_left INTEGER NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    day TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_chars INTEGER NOT NULL,
    completion_chars INTEGER NOT NULL,
    tool_calls INTEGER NOT NULL,
    ms INTEGER NOT NULL,
    error INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_user_day ON usage(user_id, day);
CREATE TABLE IF NOT EXISTS shares (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding BLOB,
    embed_model TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, created_at);
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    mime TEXT NOT NULL,
    size INTEGER NOT NULL,
    chars INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER NOT NULL DEFAULT 0,
    data BLOB,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_user ON files(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);
CREATE TABLE IF NOT EXISTS file_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,
    embed_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON file_chunks(file_id, idx);
"""

FTS_SCHEMA = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts"
    " USING fts5(content, memory_id UNINDEXED, user_id UNINDEXED)"
)
CHUNKS_FTS_SCHEMA = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS file_chunks_fts"
    " USING fts5(content, chunk_id UNINDEXED, file_id UNINDEXED)"
)
MEMORY_COLS = "id, content, created_at"
FILE_COLS = "id, user_id, session_id, name, mime, size, chars, chunks, created_at"
CHUNK_COLS = "c.id, c.file_id, c.idx, c.content"


def fts_query(text: str) -> str:
    """OR-query of the distinct words in `text`, quoted so FTS syntax can't be injected."""
    words = dict.fromkeys(w for w in re.findall(r"\w+", text.lower()) if len(w) > 2)
    return " OR ".join(f'"{w}"' for w in words)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _public_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "is_admin": bool(row["is_admin"]),
        "disabled": bool(row["disabled"]),
        "created_at": row["created_at"],
    }


class Store:
    def __init__(self, path: str = "grok_bot.sqlite3"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._add_column("messages", "reasoning", "TEXT")
        self._add_column("sessions", "user_id", "TEXT NOT NULL DEFAULT 'local'")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at)"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO users (id, username, password_hash, created_at)"
            " VALUES ('local', 'local', '!', ?)",
            (_now(),),
        )
        try:
            self._conn.execute(FTS_SCHEMA)
            self._conn.execute(CHUNKS_FTS_SCHEMA)
            self.has_fts = True
        except sqlite3.OperationalError:
            self.has_fts = False
        self._add_column("users", "disabled", "INTEGER NOT NULL DEFAULT 0")
        self._conn.commit()
        self._lock = threading.Lock()

    def _add_column(self, table: str, column: str, decl: str) -> None:
        cols = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    # --- users & login tokens ---

    def create_user(
        self, username: str, password_hash: str, is_admin: bool = False
    ) -> dict[str, Any] | None:
        """Return the new user, or None if the username (case-insensitive) is taken."""
        uid = uuid.uuid4().hex
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO users (id, username, password_hash, is_admin, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (uid, username, password_hash, int(is_admin), _now()),
                )
        except sqlite3.IntegrityError:
            return None
        return self.get_user(uid)

    def upsert_admin(self, username: str, password_hash: str) -> dict[str, Any]:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO users (id, username, password_hash, is_admin, created_at)"
                " VALUES (?, ?, ?, 1, ?) ON CONFLICT(username) DO UPDATE SET"
                " password_hash = excluded.password_hash, is_admin = 1",
                (uuid.uuid4().hex, username, password_hash, _now()),
            )
        user, _ = self.get_credentials(username) or (None, None)
        assert user is not None
        return user

    def get_user(self, uid: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        return _public_user(row) if row else None

    def get_credentials(self, username: str) -> tuple[dict[str, Any], str] | None:
        row = self._conn.execute(
            "SELECT * FROM users WHERE username = ? AND id != 'local' AND disabled = 0",
            (username,),
        ).fetchone()
        return (_public_user(row), row["password_hash"]) if row else None

    def count_users(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM users WHERE id != 'local'").fetchone()[0]

    def list_users(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM users WHERE id != 'local' ORDER BY created_at")
        return [_public_user(r) for r in rows]

    def create_token(self, user_id: str, ttl: timedelta = timedelta(days=30)) -> str:
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(UTC) + ttl).isoformat(timespec="seconds")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO auth_tokens (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                (_token_hash(token), user_id, expires),
            )
        return token

    def user_for_token(self, token: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT u.* FROM auth_tokens t JOIN users u ON u.id = t.user_id"
            " WHERE t.token_hash = ? AND t.expires_at > ? AND u.disabled = 0",
            (_token_hash(token), _now()),
        ).fetchone()
        return _public_user(row) if row else None

    # --- tasks ---

    TASK_FIELDS = (
        "title", "prompt", "schedule", "at", "time", "weekday", "mode", "think",
        "deepsearch", "enabled", "next_run_at", "last_run_at", "last_status", "last_session_id",
    )  # fmt: skip

    @staticmethod
    def _task(row: sqlite3.Row) -> dict[str, Any]:
        t = dict(row)
        for k in ("think", "deepsearch", "enabled"):
            t[k] = bool(t[k])
        return t

    def create_task(self, user_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        tid = uuid.uuid4().hex
        cols = [k for k in self.TASK_FIELDS if k in fields]
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO tasks (id, user_id, created_at, {', '.join(cols)})"
                f" VALUES (?, ?, ?, {', '.join('?' for _ in cols)})",
                (tid, user_id, _now(), *(fields[k] for k in cols)),
            )
        return self.get_task(tid)  # type: ignore[return-value]

    def get_task(self, tid: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
        return self._task(row) if row else None

    def list_tasks(self, user_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at, rowid", (user_id,)
        )
        return [self._task(r) for r in rows]

    def update_task(self, tid: str, **fields: Any) -> dict[str, Any] | None:
        cols = [k for k in self.TASK_FIELDS if k in fields]
        if cols:
            with self._lock, self._conn:
                self._conn.execute(
                    f"UPDATE tasks SET {', '.join(f'{k} = ?' for k in cols)} WHERE id = ?",
                    (*(fields[k] for k in cols), tid),
                )
        return self.get_task(tid)

    def delete_task(self, user_id: str, tid: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM tasks WHERE id = ? AND user_id = ?", (tid, user_id)
            )
        return cur.rowcount > 0

    def due_tasks(self, now: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE enabled = 1 AND next_run_at IS NOT NULL"
            " AND next_run_at <= ? ORDER BY next_run_at, rowid",
            (now,),
        )
        return [self._task(r) for r in rows]

    def claim_task(self, tid: str, expected: str, next_run_at: str | None) -> bool:
        """Advance a due task exactly once even if two runners race."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE tasks SET next_run_at = ?, enabled = ? WHERE id = ? AND next_run_at = ?",
                (next_run_at, int(next_run_at is not None), tid, expected),
            )
        return cur.rowcount > 0

    # --- API keys ---

    def create_api_key(self, user_id: str, name: str) -> dict[str, Any]:
        key = "gk_" + secrets.token_urlsafe(32)
        row = {
            "id": uuid.uuid4().hex,
            "name": name,
            "prefix": key[:10],
            "created_at": _now(),
            "last_used_at": None,
        }
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO api_keys (id, user_id, name, key_hash, prefix, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (row["id"], user_id, name, _token_hash(key), row["prefix"], row["created_at"]),
            )
        return {**row, "key": key}

    def list_api_keys(self, user_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, name, prefix, created_at, last_used_at FROM api_keys"
            " WHERE user_id = ? ORDER BY created_at DESC, rowid DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    def delete_api_key(self, user_id: str, kid: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM api_keys WHERE id = ? AND user_id = ?", (kid, user_id)
            )
        return cur.rowcount > 0

    def user_for_api_key(self, key: str) -> dict[str, Any] | None:
        h = _token_hash(key)
        row = self._conn.execute(
            "SELECT u.* FROM api_keys k JOIN users u ON u.id = k.user_id"
            " WHERE k.key_hash = ? AND u.disabled = 0",
            (h,),
        ).fetchone()
        if row is None:
            return None
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?", (_now(), h)
            )
        return _public_user(row)

    def username_taken(self, username: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        return row.fetchone() is not None

    def update_user(
        self, uid: str, disabled: bool | None = None, is_admin: bool | None = None
    ) -> dict[str, Any] | None:
        with self._lock, self._conn:
            if disabled is not None:
                self._conn.execute(
                    "UPDATE users SET disabled = ? WHERE id = ?", (int(disabled), uid)
                )
                if disabled:
                    self._conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (uid,))
            if is_admin is not None:
                self._conn.execute(
                    "UPDATE users SET is_admin = ? WHERE id = ?", (int(is_admin), uid)
                )
        return self.get_user(uid)

    def set_password(self, uid: str, password_hash: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, uid)
            )
            self._conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (uid,))

    def delete_user(self, uid: str) -> bool:
        with self._lock, self._conn:
            if self.has_fts:
                self._conn.execute("DELETE FROM memories_fts WHERE user_id = ?", (uid,))
                self._conn.execute(
                    "DELETE FROM file_chunks_fts WHERE file_id IN"
                    " (SELECT id FROM files WHERE user_id = ?)",
                    (uid,),
                )
            self._conn.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
            cur = self._conn.execute("DELETE FROM users WHERE id = ? AND id != 'local'", (uid,))
        return cur.rowcount > 0

    # --- usage ---

    def add_usage(
        self,
        user_id: str,
        model: str,
        prompt_chars: int,
        completion_chars: int,
        tool_calls: int,
        ms: int,
        error: bool,
    ) -> None:
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO usage (user_id, day, model, prompt_chars, completion_chars,"
                " tool_calls, ms, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    now[:10],
                    model,
                    prompt_chars,
                    completion_chars,
                    tool_calls,
                    ms,
                    int(error),
                    now,
                ),
            )

    def messages_today(self, user_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM usage WHERE user_id = ? AND day = ?", (user_id, _now()[:10])
        ).fetchone()
        return row[0]

    def usage_summary(self, since_day: str, user_id: str | None = None) -> list[dict[str, Any]]:
        """Per user/day/model totals since `since_day` (YYYY-MM-DD), newest first."""
        where, args = "u.day >= ?", [since_day]
        if user_id:
            where += " AND u.user_id = ?"
            args.append(user_id)
        rows = self._conn.execute(
            "SELECT u.user_id, us.username, u.day, u.model, COUNT(*) AS requests,"
            " SUM(u.error) AS errors, SUM(u.tool_calls) AS tool_calls,"
            " SUM(u.prompt_chars) AS prompt_chars, SUM(u.completion_chars) AS completion_chars,"
            " CAST(AVG(u.ms) AS INTEGER) AS avg_ms FROM usage u"
            " LEFT JOIN users us ON us.id = u.user_id"
            f" WHERE {where} GROUP BY u.user_id, u.day, u.model"
            " ORDER BY u.day DESC, requests DESC",
            args,
        )
        out = []
        for r in rows:
            d = dict(r)
            d["est_tokens"] = (d["prompt_chars"] + d["completion_chars"]) // 4
            out.append(d)
        return out

    # --- invites ---

    def create_invite(self, created_by: str, uses: int, ttl: timedelta) -> dict[str, Any]:
        code = secrets.token_urlsafe(12)
        iid = uuid.uuid4().hex[:12]
        expires = (datetime.now(UTC) + ttl).isoformat(timespec="seconds")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO invites (id, code_hash, created_by, uses_left, expires_at,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (iid, _token_hash(code), created_by, uses, expires, _now()),
            )
        return {"id": iid, "code": code, "uses_left": uses, "expires_at": expires}

    def list_invites(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, created_by, uses_left, expires_at, created_at FROM invites"
            " WHERE uses_left > 0 AND expires_at > ? ORDER BY created_at DESC",
            (_now(),),
        )
        return [dict(r) for r in rows]

    def delete_invite(self, iid: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM invites WHERE id = ?", (iid,))
        return cur.rowcount > 0

    def use_invite(self, code: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE invites SET uses_left = uses_left - 1"
                " WHERE code_hash = ? AND uses_left > 0 AND expires_at > ?",
                (_token_hash(code), _now()),
            )
        return cur.rowcount > 0

    def delete_token(self, token: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM auth_tokens WHERE token_hash = ?", (_token_hash(token),)
            )

    # --- chat sessions ---

    def create_session(self, title: str = "New chat", user_id: str = "local") -> str:
        sid = uuid.uuid4().hex
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, user_id)"
                " VALUES (?, ?, ?, ?, ?)",
                (sid, title[:80] or "New chat", now, now, user_id),
            )
        return sid

    def get_session(self, sid: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC, rowid DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    def search_sessions(self, q: str, user_id: str) -> list[dict[str, Any]]:
        like = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE user_id = ?2 AND (title LIKE ?1 ESCAPE '\\' OR id IN"
            " (SELECT session_id FROM messages WHERE role IN ('user', 'assistant')"
            " AND content LIKE ?1 ESCAPE '\\')) ORDER BY updated_at DESC, rowid DESC",
            (like, user_id),
        )
        return [dict(r) for r in rows]

    def rename_session(self, sid: str, title: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title[:80] or "New chat", sid)
            )
        return cur.rowcount > 0

    def truncate_messages(self, sid: str, from_index: int) -> int:
        """Delete the message at position `from_index` (0-based) and everything after it."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?1 AND id >= (SELECT id FROM messages"
                " WHERE session_id = ?1 ORDER BY id LIMIT 1 OFFSET ?2)",
                (sid, from_index),
            )
        return cur.rowcount

    def share_session(self, sid: str) -> str:
        row = self._conn.execute("SELECT id FROM shares WHERE session_id = ?", (sid,)).fetchone()
        if row:
            return row["id"]
        share_id = secrets.token_urlsafe(12)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO shares (id, session_id, created_at) VALUES (?, ?, ?)",
                (share_id, sid, _now()),
            )
        return share_id

    def shared_session_id(self, share_id: str) -> str | None:
        row = self._conn.execute("SELECT session_id FROM shares WHERE id = ?", (share_id,))
        found = row.fetchone()
        return found["session_id"] if found else None

    def delete_session(self, sid: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        return cur.rowcount > 0

    def add_message(self, sid: str, message: dict[str, Any]) -> None:
        tool_calls = message.get("tool_calls")
        now = _now()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, name,"
                " reasoning, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    message["role"],
                    message.get("content"),
                    json.dumps(tool_calls) if tool_calls else None,
                    message.get("tool_call_id"),
                    message.get("name"),
                    message.get("reasoning"),
                    now,
                ),
            )
            self._conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, sid))

    def get_messages(self, sid: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (sid,)
        ).fetchall()
        out = []
        for r in rows:
            m: dict[str, Any] = {
                "role": r["role"],
                "content": r["content"],
                "created_at": r["created_at"],
            }
            if r["tool_calls"]:
                m["tool_calls"] = json.loads(r["tool_calls"])
            if r["tool_call_id"]:
                m["tool_call_id"] = r["tool_call_id"]
            if r["name"]:
                m["name"] = r["name"]
            if r["reasoning"]:
                m["reasoning"] = r["reasoning"]
            out.append(m)
        return out

    # --- long-term memory ---

    def add_memory(
        self, user_id: str, content: str, embedding: bytes | None, embed_model: str | None
    ) -> dict[str, Any]:
        mid = uuid.uuid4().hex[:12]
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO memories (id, user_id, content, embedding, embed_model, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (mid, user_id, content, embedding, embed_model, _now()),
            )
            if self.has_fts:
                self._conn.execute(
                    "INSERT INTO memories_fts (content, memory_id, user_id) VALUES (?, ?, ?)",
                    (content, mid, user_id),
                )
        row = self._conn.execute(
            f"SELECT {MEMORY_COLS} FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        return dict(row)

    def find_memory(self, user_id: str, content: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            f"SELECT {MEMORY_COLS} FROM memories WHERE user_id = ? AND content = ? COLLATE NOCASE",
            (user_id, content),
        ).fetchone()
        return dict(row) if row else None

    def list_memories(self, user_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            f"SELECT {MEMORY_COLS} FROM memories WHERE user_id = ?"
            " ORDER BY created_at DESC, rowid DESC",
            (user_id,),
        )
        return [dict(r) for r in rows]

    def memory_vectors(self, user_id: str, embed_model: str) -> list[tuple[dict[str, Any], bytes]]:
        rows = self._conn.execute(
            f"SELECT {MEMORY_COLS}, embedding FROM memories"
            " WHERE user_id = ? AND embed_model = ? AND embedding IS NOT NULL",
            (user_id, embed_model),
        )
        return [({k: r[k] for k in ("id", "content", "created_at")}, r["embedding"]) for r in rows]

    def search_memories(self, user_id: str, query: str, k: int = 5) -> list[dict[str, Any]]:
        match = fts_query(query)
        if not match:
            return []
        if self.has_fts:
            rows = self._conn.execute(
                "SELECT m.id, m.content, m.created_at FROM memories_fts f"
                " JOIN memories m ON m.id = f.memory_id"
                " WHERE memories_fts MATCH ? AND f.user_id = ? AND m.user_id = ?"
                " ORDER BY bm25(memories_fts) LIMIT ?",
                (match, user_id, user_id, k),
            )
            return [dict(r) for r in rows]
        words = [w.strip('"') for w in match.split(" OR ")]
        hits = [
            (sum(w in m["content"].lower() for w in words), m) for m in self.list_memories(user_id)
        ]
        return [m for n, m in sorted(hits, key=lambda h: -h[0]) if n][:k]

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?", (memory_id, user_id)
            )
            if cur.rowcount and self.has_fts:
                self._conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        return cur.rowcount > 0

    # --- uploaded files ---

    def add_file(
        self,
        user_id: str,
        session_id: str | None,
        name: str,
        mime: str,
        size: int,
        chars: int,
        data: bytes | None,
        chunks: list[tuple[str, bytes | None]],
        embed_model: str | None,
    ) -> dict[str, Any]:
        fid = uuid.uuid4().hex[:16]
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO files (id, user_id, session_id, name, mime, size, chars, chunks,"
                " data, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (fid, user_id, session_id, name, mime, size, chars, len(chunks), data, _now()),
            )
            for idx, (content, emb) in enumerate(chunks):
                cur = self._conn.execute(
                    "INSERT INTO file_chunks (file_id, idx, content, embedding, embed_model)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (fid, idx, content, emb, embed_model if emb else None),
                )
                if self.has_fts:
                    self._conn.execute(
                        "INSERT INTO file_chunks_fts (content, chunk_id, file_id) VALUES (?, ?, ?)",
                        (content, cur.lastrowid, fid),
                    )
        file = self.get_file(fid)
        assert file is not None
        return file

    def get_file(self, file_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(f"SELECT {FILE_COLS} FROM files WHERE id = ?", (file_id,))
        found = row.fetchone()
        return dict(found) if found else None

    def file_data(self, file_id: str) -> bytes | None:
        row = self._conn.execute("SELECT data FROM files WHERE id = ?", (file_id,)).fetchone()
        return row["data"] if row else None

    def list_files(self, user_id: str, session_id: str | None = None) -> list[dict[str, Any]]:
        sql = f"SELECT {FILE_COLS} FROM files WHERE user_id = ?"
        args: list[Any] = [user_id]
        if session_id:
            sql += " AND session_id = ?"
            args.append(session_id)
        rows = self._conn.execute(sql + " ORDER BY created_at DESC, rowid DESC", args)
        return [dict(r) for r in rows]

    def delete_file(self, user_id: str, file_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM files WHERE id = ? AND user_id = ?", (file_id, user_id)
            )
            if cur.rowcount and self.has_fts:
                self._conn.execute("DELETE FROM file_chunks_fts WHERE file_id = ?", (file_id,))
        return cur.rowcount > 0

    def attach_files(self, user_id: str, file_ids: list[str], session_id: str) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "UPDATE files SET session_id = ? WHERE id = ? AND user_id = ?"
                " AND session_id IS NULL",
                [(session_id, fid, user_id) for fid in file_ids],
            )

    def session_file_ids(self, session_id: str | None) -> list[str]:
        if not session_id:
            return []
        rows = self._conn.execute(
            "SELECT id FROM files WHERE session_id = ? ORDER BY created_at", (session_id,)
        )
        return [r["id"] for r in rows]

    @staticmethod
    def _marks(ids: list[str]) -> str:
        return ",".join("?" * len(ids))

    def file_chunks(self, file_ids: list[str], max_idx: int | None = None) -> list[dict[str, Any]]:
        sql = f"SELECT {CHUNK_COLS} FROM file_chunks c WHERE c.file_id IN ({self._marks(file_ids)})"
        args: list[Any] = list(file_ids)
        if max_idx is not None:
            sql += " AND c.idx < ?"
            args.append(max_idx)
        rows = self._conn.execute(sql + " ORDER BY c.file_id, c.idx", args)
        return [dict(r) for r in rows]

    def chunk_vectors(
        self, file_ids: list[str], embed_model: str
    ) -> list[tuple[dict[str, Any], bytes]]:
        rows = self._conn.execute(
            f"SELECT {CHUNK_COLS}, c.embedding FROM file_chunks c"
            f" WHERE c.file_id IN ({self._marks(file_ids)}) AND c.embed_model = ?"
            " AND c.embedding IS NOT NULL",
            [*file_ids, embed_model],
        )
        keys = ("id", "file_id", "idx", "content")
        return [({k: r[k] for k in keys}, r["embedding"]) for r in rows]

    def search_chunks(self, file_ids: list[str], query: str, k: int = 6) -> list[dict[str, Any]]:
        match = fts_query(query)
        if not match or not self.has_fts:
            return []
        rows = self._conn.execute(
            f"SELECT {CHUNK_COLS} FROM file_chunks_fts f JOIN file_chunks c ON c.id = f.chunk_id"
            f" WHERE file_chunks_fts MATCH ? AND f.file_id IN ({self._marks(file_ids)})"
            " ORDER BY bm25(file_chunks_fts) LIMIT ?",
            [match, *file_ids, k],
        )
        return [dict(r) for r in rows]
