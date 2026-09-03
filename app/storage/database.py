"""SQLite persistence layer for sessions, messages, memory, troubleshooting, docs and vectors."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Session',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    current_objective TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    summary_msg_id INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS memories (
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, key)
);

CREATE TABLE IF NOT EXISTS troubleshooting (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    problem TEXT NOT NULL,
    attempt TEXT NOT NULL,
    result TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(session_id, problem, attempt)
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    source_type TEXT NOT NULL,
    chunks INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS terminal_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    command TEXT NOT NULL DEFAULT '',
    cwd TEXT NOT NULL DEFAULT '',
    output TEXT NOT NULL,
    exit_code INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    doc_id INTEGER,
    source TEXT NOT NULL,
    text TEXT NOT NULL,
    vector TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_session ON embeddings(session_id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token)."""
    return max(1, len(text) // 4)


class Database:
    """Thread-safe synchronous SQLite wrapper. Fine for a single-user local assistant."""

    def __init__(self, db_path: str | Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ sessions

    def create_session(self, session_id: str | None = None, title: str = "New Session") -> dict:
        sid = session_id or uuid.uuid4().hex[:12]
        now = utcnow()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (sid, title, now, now),
            )
            self._conn.commit()
        return self.get_session(sid)  # type: ignore[return-value]

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return self._session_to_dict(row) if row else None

    def latest_session(self) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return self._session_to_dict(row) if row else None

    def list_sessions(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [self._session_to_dict(r) for r in rows]

    def touch_session(self, session_id: str, title: str | None = None) -> None:
        with self._lock:
            if title:
                self._conn.execute(
                    "UPDATE sessions SET updated_at = ?, title = ? WHERE id = ? AND title = 'New Session'",
                    (utcnow(), title, session_id),
                )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (utcnow(), session_id)
            )
            self._conn.commit()

    def set_objective(self, session_id: str, objective: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET current_objective = ?, updated_at = ? WHERE id = ?",
                (objective, utcnow(), session_id),
            )
            self._conn.commit()

    def set_summary(self, session_id: str, summary: str, summary_msg_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET summary = ?, summary_msg_id = ?, updated_at = ? WHERE id = ?",
                (summary, summary_msg_id, utcnow(), session_id),
            )
            self._conn.commit()

    def delete_session(self, session_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            for table in ("messages", "memories", "troubleshooting", "documents",
                          "terminal_logs", "embeddings"):
                self._conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
            self._conn.commit()
        return cur.rowcount

    @staticmethod
    def _session_to_dict(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "current_objective": row["current_objective"],
            "summary": row["summary"],
            "summary_msg_id": row["summary_msg_id"],
        }

    # ------------------------------------------------------------------ messages

    def add_message(self, session_id: str, role: str, content: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages (session_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, approx_tokens(content), utcnow()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent_messages(self, session_id: str, limit: int, max_id: int | None = None) -> list[dict]:
        query = "SELECT * FROM messages WHERE session_id = ? AND role != 'system'"
        params: list[Any] = [session_id]
        if max_id is not None:
            query += " AND id > ?"
            params.append(max_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in reversed(rows)]

    def messages_since(self, session_id: str, min_id: int, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? AND role != 'system' "
                "ORDER BY id ASC LIMIT ?",
                (session_id, min_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def history(self, session_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND role != 'system' "
                "ORDER BY id ASC LIMIT ? OFFSET ?",
                (session_id, limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    def max_message_id(self, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(id), 0) AS m FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["m"])

    # ------------------------------------------------------------------ long-term memory

    def set_memory(self, session_id: str, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (session_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (session_id, key.strip(), value.strip(), utcnow()),
            )
            self._conn.commit()

    def get_memories(self, session_id: str) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM memories WHERE session_id = ? ORDER BY key",
                (session_id,),
            ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def delete_memory(self, session_id: str, key: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE session_id = ? AND key = ?", (session_id, key)
            )
            self._conn.commit()
        return cur.rowcount

    # ------------------------------------------------------------------ troubleshooting

    def add_attempt(self, session_id: str, problem: str, attempt: str, result: str = "") -> bool:
        """Record a troubleshooting attempt. Returns False if it was already recorded."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO troubleshooting (session_id, problem, attempt, result, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, problem.strip(), attempt.strip(), result.strip(), utcnow()),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def get_attempts(self, session_id: str, problem: str | None = None) -> list[dict]:
        query = "SELECT * FROM troubleshooting WHERE session_id = ?"
        params: list[Any] = [session_id]
        if problem:
            query += " AND problem = ?"
            params.append(problem.strip())
        query += " ORDER BY id ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ documents

    def add_document(self, session_id: str, filename: str, source_type: str, chunks: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO documents (session_id, filename, source_type, chunks, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, filename, source_type, chunks, utcnow()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_documents(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE session_id = ? ORDER BY id DESC", (session_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ terminal logs

    def add_terminal_log(self, session_id: str, command: str, cwd: str, output: str,
                         exit_code: int | None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO terminal_logs (session_id, command, cwd, output, exit_code, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, command, cwd, output, exit_code, utcnow()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent_terminal_logs(self, session_id: str, limit: int = 5) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM terminal_logs WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------ vector storage

    def add_embedding(self, session_id: str, doc_id: int | None, source: str,
                      text: str, vector: dict | list) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO embeddings (session_id, doc_id, source, text, vector, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, doc_id, source, text, json.dumps(vector), utcnow()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def load_embeddings(self, session_id: str | None = None) -> list[dict]:
        if session_id:
            rows = self._conn.execute(
                "SELECT chunk_id, session_id, doc_id, source, text, vector FROM embeddings "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT chunk_id, session_id, doc_id, source, text, vector FROM embeddings"
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        with self._lock:
            self._conn.close()
