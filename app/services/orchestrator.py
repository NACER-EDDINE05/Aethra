"""Orchestrator: the full request pipeline.

User -> Context Builder -> Memory Manager -> RAG Engine -> Prompt Builder -> Local LLM -> Response
"""

from __future__ import annotations

import logging
import time

from app.context.builder import ContextBundle
from app.context.prompt_builder import build_messages, estimate_prompt_tokens
from app.memory.manager import MemoryManager
from app.memory.summarizer import Summarizer
from app.models import schemas

logger = logging.getLogger("aethra.orchestrator")


class AethraService:
    """Wires memory, RAG, prompt building and the LLM into the request pipeline."""

    def __init__(self, settings, database, llm_client, rag_engine):
        self.settings = settings
        self.db = database
        self.llm = llm_client
        self.rag = rag_engine
        self.memory = MemoryManager(database, settings)
        self.summarizer = Summarizer(llm_client, self.memory, database)

    # ---------------------------------------------------------------- internals

    async def _build_bundle(self, session: dict, user_message: str | None,
                            use_rag: bool) -> ContextBundle:
        session_id = session["id"]
        retrieved = []
        if use_rag and user_message:
            try:
                retrieved = self.rag.retrieve(user_message, session_id=session_id)
            except Exception as exc:  # noqa: BLE001 - retrieval failure must not kill chat
                logger.warning("RAG retrieval failed (continuing without docs): %s", exc)
        return ContextBundle(
            session=session,
            env_block=self.memory.env_block(session_id),
            objective=session["current_objective"],
            summary=session["summary"],
            attempts_block=self.memory.attempts_block(session_id),
            terminal_block=self.memory.recent_terminal_block(session_id),
            retrieved=retrieved,
            recent_messages=self.memory.recent_messages(session_id),
        )

    async def _respond(self, session_id: str, user_message: str,
                       use_rag: bool) -> tuple[str, list, int]:
        start = time.perf_counter()
        session = self.memory.get_or_create_session(session_id)
        self.memory.add_message(session["id"], "user", user_message)

        bundle = await self._build_bundle(session, user_message, use_rag)
        prompt_messages = build_messages(bundle, user_message)
        prompt_tokens = estimate_prompt_tokens(prompt_messages)

        answer = await self.llm.chat(prompt_messages)
        self.memory.add_message(session["id"], "assistant", answer)
        self.db.touch_session(session["id"], title=user_message[:60])

        # Auto-summarize when history grows (non-fatal on failure).
        if self.memory.needs_summarization(session["id"]):
            try:
                await self.summarizer.summarize(session["id"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Auto-summarization failed: %s", exc)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "chat done (session=%s, retrieved=%d, prompt_tokens~%d, answer_tokens~%d, %dms)",
            session["id"], len(bundle.retrieved), prompt_tokens,
            len(answer) // 4, elapsed_ms,
        )
        return answer, bundle.retrieved, elapsed_ms

    @staticmethod
    def _retrieved_info(retrieved) -> list[schemas.RetrievedChunkInfo]:
        return [
            schemas.RetrievedChunkInfo(
                source=c.source, score=round(c.score, 4), excerpt=c.text[:300]
            )
            for c in retrieved
        ]

    # ---------------------------------------------------------------- chat

    async def chat(self, request: schemas.ChatRequest) -> schemas.ChatResponse:
        session = self.memory.get_or_create_session(request.session_id)
        answer, retrieved, ms = await self._respond(
            session["id"], request.message, request.use_rag
        )
        return schemas.ChatResponse(
            session_id=session["id"], answer=answer,
            retrieved=self._retrieved_info(retrieved), processing_ms=ms,
        )

    # ---------------------------------------------------------------- terminal

    async def process_terminal(self, request: schemas.TerminalRequest) -> schemas.TerminalResponse:
        session = self.memory.get_or_create_session(request.session_id)
        log_id = self.db.add_terminal_log(
            session["id"], request.command or "", request.cwd or "",
            request.output, request.exit_code,
        )
        # Also index the output for future retrieval.
        try:
            self.rag.index_text(
                session["id"], source=f"terminal:{request.command or 'output'}",
                text=request.output,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Terminal output indexing failed: %s", exc)

        if not request.explain:
            return schemas.TerminalResponse(session_id=session["id"], log_id=log_id)

        user_message = (
            "Explain this terminal output. Identify errors, warnings and their cause, "
            "and suggest next steps."
            + (f" The command was: {request.command}" if request.command else "")
            + (f" Working directory: {request.cwd}" if request.cwd else "")
            + (f" Exit code: {request.exit_code}" if request.exit_code is not None else "")
            + f"\n\nOutput:\n{request.output}"
        )
        answer, retrieved, ms = await self._respond(session["id"], user_message, use_rag=False)
        return schemas.TerminalResponse(
            session_id=session["id"], log_id=log_id, answer=answer,
            retrieved=self._retrieved_info(retrieved), processing_ms=ms,
        )

    # ---------------------------------------------------------------- screenshot

    async def process_screenshot(self, session_id: str | None, ocr_text: str,
                                 question: str | None) -> schemas.ScreenshotResponse:
        session = self.memory.get_or_create_session(session_id)
        self.memory.add_message(session["id"], "ocr", f"[Screenshot OCR]\n{ocr_text}")
        try:
            self.rag.index_text(session["id"], source="screenshot-ocr", text=ocr_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Screenshot OCR indexing failed: %s", exc)

        if not question:
            return schemas.ScreenshotResponse(
                session_id=session["id"], ocr_text=ocr_text, processing_ms=0
            )
        answer, _, ms = await self._respond(
            session["id"],
            f"Regarding this screenshot content:\n{ocr_text}\n\n{question}",
            use_rag=True,
        )
        return schemas.ScreenshotResponse(
            session_id=session["id"], ocr_text=ocr_text, answer=answer, processing_ms=ms
        )

    # ---------------------------------------------------------------- upload

    def upload_document(self, session_id: str | None, filename: str, text: str,
                        index: bool = True) -> schemas.UploadResponse:
        session = self.memory.get_or_create_session(session_id)
        chunks_indexed = 0
        doc_id = self.db.add_document(session["id"], filename, "uploaded", 0)
        if index and text.strip():
            try:
                chunks_indexed = self.rag.index_text(
                    session["id"], source=filename, text=text, doc_id=doc_id
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Document indexing failed: %s", exc)
        return schemas.UploadResponse(
            session_id=session["id"], doc_id=doc_id, filename=filename,
            chunks_indexed=chunks_indexed, indexed=chunks_indexed > 0,
        )

    # ---------------------------------------------------------------- session state / history

    def session_state(self, session_id: str | None) -> schemas.SessionStateResponse:
        session = self.memory.get_or_create_session(session_id)
        sid = session["id"]
        return schemas.SessionStateResponse(
            id=sid,
            title=session["title"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
            current_objective=session["current_objective"],
            memories=self.db.get_memories(sid),
            summary=session["summary"],
            message_count=len(self.db.history(sid, limit=10000)),
            recent_messages=[
                schemas.MessageOut(**m) for m in self.memory.recent_messages(sid)
            ],
            troubleshooting=[schemas.AttemptOut(**a) for a in self.db.get_attempts(sid)],
            documents=[schemas.DocumentOut(**d) for d in self.db.get_documents(sid)],
            recent_terminal=[
                schemas.TerminalLogOut(**t) for t in self.db.recent_terminal_logs(sid)
            ],
        )

    def history(self, session_id: str | None, limit: int, offset: int) -> schemas.HistoryResponse:
        session = self.memory.get_or_create_session(session_id)
        messages = self.db.history(session["id"], limit=limit, offset=offset)
        return schemas.HistoryResponse(
            session_id=session["id"], count=len(messages),
            messages=[schemas.MessageOut(**m) for m in messages],
        )

    # ---------------------------------------------------------------- memory / summarize

    def update_memory(self, request: schemas.MemoryUpdateRequest) -> schemas.MemoryResponse:
        session = self.memory.get_or_create_session(request.session_id)
        memories = self.memory.update_memory(session["id"], request.updates)
        return schemas.MemoryResponse(session_id=session["id"], memories=memories)

    async def summarize(self, session_id: str | None) -> schemas.SummarizeResponse:
        session = self.memory.get_or_create_session(session_id)
        summary = await self.summarizer.summarize(session["id"])
        return schemas.SummarizeResponse(session_id=session["id"], summary=summary)
