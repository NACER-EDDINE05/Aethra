"""Retrieval-Augmented Generation engine: index documents and retrieve relevant chunks."""

from __future__ import annotations

import logging

from app.rag.ingest import chunk_text
from app.rag.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger("aethra.rag")


class RAGEngine:
    def __init__(self, vector_store: VectorStore, embedder, settings):
        self._store = vector_store
        self._embedder = embedder
        self._settings = settings

    @property
    def embedder(self):
        return self._embedder

    def index_text(self, session_id: str, source: str, text: str,
                   doc_id: int | None = None) -> int:
        """Chunk, embed and index a document. Returns the number of chunks indexed."""
        chunks = chunk_text(text, self._settings.chunk_size, self._settings.chunk_overlap)
        if not chunks:
            return 0
        count = self._store.add_chunks(session_id, doc_id, source, chunks, self._embedder)
        logger.info("Indexed %d chunks from '%s' (session=%s)", count, source, session_id)
        return count

    def retrieve(self, query: str, session_id: str | None = None,
                 k: int | None = None) -> list[RetrievedChunk]:
        top_k = k if k is not None else self._settings.top_k
        results = self._store.search(query, self._embedder, session_id=session_id, k=top_k)
        if results:
            logger.info("Retrieved %d chunks for query (session=%s)", len(results), session_id)
        return results
