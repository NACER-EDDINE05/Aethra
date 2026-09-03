"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.routes import router
from app.config import Settings
from app.llm.client import LLMClient
from app.logging_setup import setup_logging
from app.rag.embedder import get_embedder
from app.rag.engine import RAGEngine
from app.rag.vector_store import VectorStore
from app.services.orchestrator import AethraService
from app.storage.database import Database

logger = logging.getLogger("aethra.main")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Aethra starting | LLM: %s (%s) | embeddings: auto->%s | data: %s",
            settings.llm_base_url, settings.llm_model,
            settings.embedding_backend, settings.data_dir,
        )
        yield
        app.state.service.db.close()
        logger.info("Aethra stopped")

    app = FastAPI(
        title="Aethra",
        description="Backend for the Aethra cybersecurity & DevOps AI assistant "
        "(locally hosted LLM, memory, RAG, OCR).",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ---- wiring -----------------------------------------------------------
    database = Database(settings.db_path)
    llm = LLMClient(settings)
    embedder = get_embedder(settings.embedding_backend, settings.embedding_model)
    vector_store = VectorStore(database)
    rag = RAGEngine(vector_store, embedder, settings)
    service = AethraService(settings, database, llm, rag)
    app.state.service = service

    app.include_router(router)

    # ---- request logging middleware (metadata only, never message bodies) --
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_id=%s unhandled error on %s %s",
                             request_id, request.method, request.url.path)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("request_id=%s %s %s -> %d (%.0fms)",
                    request_id, request.method, request.url.path,
                    response.status_code, elapsed_ms)
        response.headers["X-Request-ID"] = request_id
        return response

    return app
