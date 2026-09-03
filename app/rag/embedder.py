"""Pluggable embedding backends.

Priority:
1. SentenceTransformerEmbedder - true semantic embeddings (requires sentence-transformers).
2. HashedTFIDFEmbedder        - dependency-free hashed bag-of-words fallback.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re

logger = logging.getLogger("aethra.rag")

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class Embedder:
    """Interface: embed(texts) -> list of vectors (dense list or sparse dict)."""

    name: str = "base"

    def embed(self, texts: list[str]) -> list:
        raise NotImplementedError


class HashedTFIDFEmbedder(Embedder):
    """Dependency-free hashed TF-IDF style embedder.

    Tokens are hashed into a fixed-size index space and weighted with
    sublinear term frequency, then L2-normalized. Works well for keyword /
    technical-term overlap retrieval, which suits man pages, configs and logs.
    """

    def __init__(self, dim: int = 1 << 16):
        self.dim = dim
        self.name = f"hashed-tfidf-{dim}"

    def embed_one(self, text: str) -> dict[int, float]:
        counts: dict[int, int] = {}
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.dim
            counts[index] = counts.get(index, 0) + 1
        if not counts:
            return {0: 0.0}
        vector = {i: 1.0 + math.log(c) for i, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vector.values()))
        return {i: v / norm for i, v in vector.items()}

    def embed(self, texts: list[str]) -> list:
        return [self.embed_one(t) for t in texts]


class SentenceTransformerEmbedder(Embedder):
    """Semantic embeddings via sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"

    def embed(self, texts: list[str]) -> list:
        encoded = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vec)) for vec in encoded]


def get_embedder(backend: str = "auto",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """Factory that selects the embedding backend with graceful degradation."""
    backend = (backend or "auto").lower()
    if backend in ("auto", "st"):
        try:
            embedder = SentenceTransformerEmbedder(model_name)
            logger.info("Embedding backend: %s", embedder.name)
            return embedder
        except ImportError:
            if backend == "st":
                raise RuntimeError(
                    "AETHRA_EMBEDDING_BACKEND=st but sentence-transformers is not installed. "
                    "Install it with: pip install sentence-transformers"
                )
            logger.info("sentence-transformers not installed; falling back to hashed TF-IDF")
    embedder = HashedTFIDFEmbedder()
    logger.info("Embedding backend: %s", embedder.name)
    return embedder
