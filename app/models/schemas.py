"""Pydantic API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- chat

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    use_rag: bool = True


class RetrievedChunkInfo(BaseModel):
    source: str
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    retrieved: list[RetrievedChunkInfo] = []
    processing_ms: int


# ---------------------------------------------------------------- terminal

class TerminalRequest(BaseModel):
    output: str = Field(min_length=1)
    session_id: str | None = None
    command: str | None = None
    cwd: str | None = None
    exit_code: int | None = None
    explain: bool = False


class TerminalResponse(BaseModel):
    session_id: str
    log_id: int
    answer: str | None = None
    retrieved: list[RetrievedChunkInfo] = []
    processing_ms: int = 0


# ---------------------------------------------------------------- memory

class MemoryUpdateRequest(BaseModel):
    session_id: str | None = None
    updates: dict[str, str]


class MemoryDeleteRequest(BaseModel):
    session_id: str
    key: str


class MemoryResponse(BaseModel):
    session_id: str
    memories: dict[str, str]


# ---------------------------------------------------------------- session / history

class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: str


class AttemptOut(BaseModel):
    id: int
    problem: str
    attempt: str
    result: str
    created_at: str


class DocumentOut(BaseModel):
    id: int
    filename: str
    source_type: str
    chunks: int
    created_at: str


class TerminalLogOut(BaseModel):
    id: int
    command: str
    cwd: str
    output: str
    exit_code: int | None
    created_at: str


class SessionStateResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    current_objective: str
    memories: dict[str, str]
    summary: str
    message_count: int
    recent_messages: list[MessageOut]
    troubleshooting: list[AttemptOut]
    documents: list[DocumentOut]
    recent_terminal: list[TerminalLogOut]


class HistoryResponse(BaseModel):
    session_id: str
    count: int
    messages: list[MessageOut]


# ---------------------------------------------------------------- summarize

class SummarizeRequest(BaseModel):
    session_id: str | None = None


class SummarizeResponse(BaseModel):
    session_id: str
    summary: str


# ---------------------------------------------------------------- upload / screenshot

class UploadResponse(BaseModel):
    session_id: str
    doc_id: int
    filename: str
    chunks_indexed: int
    indexed: bool


class ScreenshotResponse(BaseModel):
    session_id: str
    ocr_text: str
    answer: str | None = None
    processing_ms: int
