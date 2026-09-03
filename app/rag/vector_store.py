"""Lightweight vector store backed by SQLite with cosine-similarity search."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


def cosine(a: dict | list, b: dict | list) -> float:
    """Cosine similarity supporting both dense lists and sparse dict vectors."""
    if isinstance(a, dict) and isinstance(b, dict):
        dot = sum(v * b.get(i, 0.0) for i, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
    elif isinstance(a, list) and isinstance(b, list):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
    else:  # mixed representations are incomparable
        return 0.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class RetrievedChunk:
    chunk_id: int
    session_id: str
    doc_id: int | None
    source: str
    text: str
    score: float


class VectorStore:
    """Stores chunk embeddings (JSON-serialized) and retrieves the top-k matches."""

    def __init__(self, database):
        self._db = database

    def add_chunks(self, session_id: str, doc_id: int | None, source: str,
                   chunks: list[str], embedder) -> int:
        """Embed and persist chunks. Returns the number of chunks stored."""
        vectors = embedder.embed(chunks)
        count = 0
        for text, vector in zip(chunks, vectors):
            self._db.add_embedding(session_id, doc_id, source, text, vector)
            count += 1
        return count

    def search(self, query: str, embedder, session_id: str | None = None,
               k: int = 5) -> list[RetrievedChunk]:
        if k <= 0 or not query.strip():
            return []
        query_vector = embedder.embed([query])[0]
        rows = self._db.load_embeddings(session_id)
        scored: list[RetrievedChunk] = []
        for row in rows:
            raw = json.loads(row["vector"])
            # JSON round-trips sparse dict keys as strings; restore ints for cosine().
            vector = (
                {int(k): v for k, v in raw.items()} if isinstance(raw, dict) else raw
            )
            score = cosine(query_vector, vector)
            if score <= 0.0:
                continue
            scored.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    session_id=row["session_id"],
                    doc_id=row["doc_id"],
                    source=row["source"],
                    text=row["text"],
                    score=score,
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]
