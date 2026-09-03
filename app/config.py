from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into the environment (no external dependency).

    Existing environment variables always win over .env values.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(name: str, default: str) -> str:
    """Read an environment variable with the AETHRA_ prefix, falling back to a raw name."""
    value = os.environ.get(f"AETHRA_{name}")
    if value is not None:
        return value
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime configuration. All values can be overridden via environment variables."""

    app_name: str = "Aethra"

    # Local LLM API (OpenAI-compatible)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b"
    llm_api_key: str = "not-needed"
    llm_temperature: float = 0.4
    llm_max_tokens: int = 1024
    llm_timeout: float = 120.0

    # Embeddings / retrieval
    embedding_backend: str = "auto"  # auto | st | tfidf
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Memory
    recent_window: int = 12
    summarize_threshold: int = 24

    # OCR
    ocr_lang: str = "eng"

    # Storage
    data_dir: Path = field(default_factory=lambda: Path("./data").resolve())
    log_level: str = "INFO"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "aethra.sqlite3"

    @classmethod
    def load(cls) -> "Settings":
        # Pick up ./data/.env, project-root .env or cwd .env before reading env vars.
        for candidate in (Path.cwd() / ".env",
                          Path(__file__).resolve().parent.parent / ".env"):
            _load_dotenv(candidate)
        data_dir = Path(_env("DATA_DIR", "./data")).expanduser().resolve()
        settings = cls(
            llm_base_url=_env("LLM_BASE_URL", cls.llm_base_url).rstrip("/"),
            llm_model=_env("LLM_MODEL", cls.llm_model),
            llm_api_key=_env("LLM_API_KEY", cls.llm_api_key),
            llm_temperature=_env_float("LLM_TEMPERATURE", cls.llm_temperature),
            llm_max_tokens=_env_int("LLM_MAX_TOKENS", cls.llm_max_tokens),
            llm_timeout=_env_float("LLM_TIMEOUT", cls.llm_timeout),
            embedding_backend=_env("EMBEDDING_BACKEND", cls.embedding_backend).lower(),
            embedding_model=_env("EMBEDDING_MODEL", cls.embedding_model),
            top_k=_env_int("TOP_K", cls.top_k),
            chunk_size=_env_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", cls.chunk_overlap),
            recent_window=_env_int("RECENT_WINDOW", cls.recent_window),
            summarize_threshold=_env_int("SUMMARIZE_THRESHOLD", cls.summarize_threshold),
            ocr_lang=_env("OCR_LANG", cls.ocr_lang),
            data_dir=data_dir,
            log_level=_env("LOG_LEVEL", cls.log_level).upper(),
        )
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        return settings
