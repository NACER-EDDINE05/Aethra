"""Conversation summarizer. Produces structured summaries used instead of full history."""

from __future__ import annotations

import logging

logger = logging.getLogger("aethra.summarizer")

SUMMARY_PROMPT = """You are summarizing a technical assistance conversation so it can be used \
as compact memory in future prompts. Produce a structured summary in EXACTLY this format:

CURRENT OBJECTIVE: <one sentence, or 'unchanged'>
ENVIRONMENT: <key environment facts, comma separated, or 'unchanged'>
IMPORTANT DISCOVERIES: <bullet list of key findings, or 'none'>
RESOLVED ISSUES: <bullet list, or 'none'>
PENDING QUESTIONS: <bullet list of open items, or 'none'>
TROUBLESHOOTING STATE: <attempts already made and their outcome, or 'none'>

Be concise and factual. Use only information present in the conversation.

--- SESSION MEMORY ---
{env}

--- CURRENT OBJECTIVE ---
{objective}

--- EXISTING SUMMARY (earlier turns) ---
{existing_summary}

--- RECENT CONVERSATION TO SUMMARIZE ---
{conversation}
"""


class Summarizer:
    def __init__(self, llm_client, memory: "MemoryManager", database):
        self._llm = llm_client
        self._memory = memory
        self._db = database

    async def summarize(self, session_id: str) -> str:
        """Generate and persist a structured summary. Returns the summary text."""
        session = self._db.get_session(session_id)
        if not session:
            raise ValueError(f"Unknown session: {session_id}")

        recent = self._db.messages_since(session_id, session["summary_msg_id"], limit=200)
        if not recent:
            return session["summary"] or "(nothing to summarize yet)"

        conversation = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in recent
        )
        prompt = SUMMARY_PROMPT.format(
            env=self._memory.env_block(session_id),
            objective=session["current_objective"] or "(not set)",
            existing_summary=session["summary"] or "(none)",
            conversation=conversation,
        )

        try:
            summary = await self._llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
            )
        except Exception as exc:  # noqa: BLE001 - summarization must never break the chat flow
            logger.warning("LLM summarization failed, using extractive fallback: %s", exc)
            summary = self._extractive_fallback(session_id)

        marker_id = self._db.max_message_id(session_id)
        self._db.set_summary(session_id, summary, marker_id)
        logger.info("Session summarized (session=%s, messages_covered=%d)", session_id, len(recent))
        return summary

    def _extractive_fallback(self, session_id: str) -> str:
        """Dependency-free fallback if the LLM is unavailable."""
        session = self._db.get_session(session_id) or {}
        attempts = self._db.get_attempts(session_id)
        lines = [
            f"CURRENT OBJECTIVE: {session.get('current_objective') or '(not set)'}",
            "ENVIRONMENT: " + (self._memory.env_block(session_id).replace("\n", "; ") or "none"),
            "IMPORTANT DISCOVERIES: (see history; auto-summary unavailable)",
            "RESOLVED ISSUES: unknown",
            "PENDING QUESTIONS: see recent messages",
            "TROUBLESHOOTING STATE: "
            + ("; ".join(f"{a['problem']}: tried '{a['attempt']}'" for a in attempts) or "none"),
        ]
        return "\n".join(lines)
