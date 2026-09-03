"""Document parsing and chunking for the RAG index."""

from __future__ import annotations

import json
import re


class UnsupportedFileError(ValueError):
    pass


# Extensions whose bytes are plain UTF-8 text.
TEXT_EXTENSIONS = {
    "txt", "md", "markdown", "json", "yaml", "yml", "ini", "conf", "cfg",
    "log", "xml", "csv", "toml", "env", "sh", "bash", "py", "service",
    "html", "css", "js", "sql", "dockerfile", "gitignore", "plist",
}


def source_type_for(filename: str) -> str:
    name = filename.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else name
    if ext == "pdf":
        return "pdf"
    if ext in TEXT_EXTENSIONS:
        return "text"
    return "text"


def parse_file(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded file."""
    name = filename.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""

    if ext == "pdf":
        return _parse_pdf(data)

    if not ext or ext in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="replace")

    # Unknown binary extension: reject if it contains NUL bytes (binary signature),
    # otherwise accept as leniently-decoded text.
    text = data.decode("utf-8", errors="replace")
    if "\x00" in text:
        raise UnsupportedFileError(
            f"Unsupported file type: '{filename}'. Supported: PDF, TXT, MD, JSON, YAML, "
            "INI, CONF, LOG, XML, CSV and other plain-text formats."
        )
    return text


def _parse_pdf(data: bytes) -> str:
    try:
        import io

        from pypdf import PdfReader  # optional dependency
    except ImportError as exc:
        raise UnsupportedFileError(
            "PDF support requires the 'pypdf' package. Install it with: pip install pypdf"
        ) from exc
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - corrupted page should not kill ingestion
            pages.append("")
    return "\n\n".join(p for p in pages if p.strip())


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    text = text.strip()
    if not text:
        return []
    size = max(size, 100)
    overlap = max(0, min(overlap, size // 2))

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())

    for para in paragraphs:
        if len(para) > size:
            flush()
            buf = ""
            step = max(1, size - overlap)
            for start in range(0, len(para), step):
                part = para[start:start + size]
                chunks.append(part.strip())
                if start + size >= len(para):
                    break
            continue

        if not buf:
            buf = para
        elif len(buf) + len(para) + 2 <= size:
            buf += "\n\n" + para
        else:
            flush()
            tail = buf[-overlap:] if overlap else ""
            if tail and len(tail) + len(para) + 2 <= size:
                buf = tail + "\n\n" + para
            else:
                buf = para
    flush()

    return [c for c in chunks if c]


def pretty_json_like(text: str) -> str:
    """Best-effort re-indent for JSON/YAML so chunks stay readable."""
    try:
        return json.dumps(json.loads(text), indent=2)
    except (ValueError, TypeError):
        return text
