"""Memory manager: session lifecycle, short-term history, long-term environment memory,
and troubleshooting memory with duplicate-attempt prevention."""

from __future__ import annotations

import logging

logger = logging.getLogger("aethra.memory")


class MemoryManager:
    def __init__(self, database, settings):
        self._db = database
        self._settings = settings

    # ------------------------------------------------------------------ sessions

    def get_or_create_session(self, session_id: str | None = None,
                              title: str | None = None) -> dict:
        if session_id:
            session = self._db.get_session(session_id)
            if session is None:
                session = self._db.create_session(session_id, title or "New Session")
            if title:
                self._db.touch_session(session_id, title)
            return session
        session = self._db.latest_session()
        if session is None:
            session = self._db.create_session(title=title or "New Session")
        return session

    # ------------------------------------------------------------------ short-term memory

    def add_message(self, session_id: str, role: str, content: str) -> int:
        return self._db.add_message(session_id, role, content)

    def recent_messages(self, session_id: str) -> list[dict]:
        """Recent turns AFTER the last summarization point."""
        session = self._db.get_session(session_id)
        max_id = session["summary_msg_id"] if session else None
        return self._db.recent_messages(session_id, self._settings.recent_window, max_id=max_id)

    def needs_summarization(self, session_id: str) -> bool:
        session = self._db.get_session(session_id)
        if not session:
            return False
        since = self._db.messages_since(session_id, session["summary_msg_id"], limit=1000)
        return len(since) >= self._settings.summarize_threshold

    # ------------------------------------------------------------------ long-term memory

    def update_memory(self, session_id: str, updates: dict[str, str]) -> dict[str, str]:
        for key, value in updates.items():
            key, value = key.strip(), str(value).strip()
            if not key:
                continue
            self._db.set_memory(session_id, key, value)
            logger.info("Memory updated: %s (session=%s)", key, session_id)
        return self._db.get_memories(session_id)

    def remove_memory(self, session_id: str, key: str) -> bool:
        return self._db.delete_memory(session_id, key) > 0

    def env_block(self, session_id: str) -> str:
        memories = self._db.get_memories(session_id)
        if not memories:
            return "(no environment details remembered yet)"
        lines = [f"- {key}: {value}" for key, value in memories.items()]
        return "\n".join(lines)

    # ------------------------------------------------------------------ troubleshooting memory

    def record_attempt(self, session_id: str, problem: str, attempt: str,
                       result: str = "") -> bool:
        """Record an attempt; returns False (and stores nothing) if it is a duplicate."""
        added = self._db.add_attempt(session_id, problem, attempt, result)
        if added:
            logger.info("Troubleshooting attempt recorded (session=%s, problem=%s)",
                        session_id, problem[:60])
        return added

    def attempts_block(self, session_id: str) -> str:
        attempts = self._db.get_attempts(session_id)
        if not attempts:
            return "(no troubleshooting attempts recorded)"
        lines = []
        current_problem = None
        for a in attempts:
            if a["problem"] != current_problem:
                current_problem = a["problem"]
                lines.append(f"Problem: {current_problem}")
            result = f" -> result: {a['result']}" if a["result"] else ""
            lines.append(f"  - tried: {a['attempt']}{result}")
        lines.append(
            "IMPORTANT: Do not suggest any of these steps again unless the user explicitly asks."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ uploads / terminal

    def recent_terminal_block(self, session_id: str, limit: int = 3) -> str:
        logs = self._db.recent_terminal_logs(session_id, limit)
        if not logs:
            return "(no recent terminal output)"
        parts = []
        for log in logs:
            header = log["command"] or "(output only)"
            cwd = f" (cwd: {log['cwd']})" if log["cwd"] else ""
            code = f" [exit code: {log['exit_code']}]" if log["exit_code"] is not None else ""
            parts.append(f"$ {header}{cwd}{code}\n{log['output']}")
        return "\n\n".join(parts)
