"""API routes."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.models import schemas
from app.ocr.ocr import OCRError, run_ocr
from app.rag.ingest import UnsupportedFileError, parse_file

router = APIRouter()


def get_service(request: Request):
    return request.app.state.service


@router.get("/health")
async def health(request: Request) -> dict:
    service = get_service(request)
    return {
        "status": "ok",
        "app": service.settings.app_name,
        "llm_base_url": service.llm.base_url,
        "llm_model": service.llm.model,
        "embedding_backend": service.rag.embedder.name,
    }


@router.post("/chat", response_model=schemas.ChatResponse)
async def chat(payload: schemas.ChatRequest, request: Request):
    """Process a user message and return an answer."""
    service = get_service(request)
    try:
        return await service.chat(payload)
    except Exception as exc:  # LLMError and friends
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/upload", response_model=schemas.UploadResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    index: bool = Form(True),
):
    """Upload a file (PDF, TXT, MD, JSON, YAML, INI, CONF, LOG, XML, CSV, ...) and index it."""
    service = get_service(request)
    data = await file.read()
    try:
        text = parse_file(file.filename or "upload.txt", data)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    return service.upload_document(session_id, file.filename or "upload.txt", text, index=index)


@router.post("/terminal", response_model=schemas.TerminalResponse)
async def terminal(payload: schemas.TerminalRequest, request: Request):
    """Store terminal output (cwd, command, output, exit code); optionally explain it."""
    service = get_service(request)
    try:
        return await service.process_terminal(payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/screenshot", response_model=schemas.ScreenshotResponse)
async def screenshot(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    question: str | None = Form(None),
):
    """Upload an image, perform OCR, store the extracted text; optionally answer a question."""
    service = get_service(request)
    data = await file.read()
    try:
        ocr_text = run_ocr(data, lang=service.settings.ocr_lang)
    except OCRError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not ocr_text:
        raise HTTPException(status_code=422, detail="OCR extracted no text from the image.")
    try:
        return await service.process_screenshot(session_id, ocr_text, question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/session", response_model=schemas.SessionStateResponse)
async def session_state(request: Request, session_id: str | None = None):
    """Return current session state (latest session if no id given)."""
    return get_service(request).session_state(session_id)


@router.post("/memory", response_model=schemas.MemoryResponse)
async def update_memory(payload: schemas.MemoryUpdateRequest, request: Request):
    """Update remembered environment details (os, distro, router, shell, project, ...)."""
    return get_service(request).update_memory(payload)


@router.delete("/memory", response_model=schemas.MemoryResponse)
async def delete_memory(session_id: str, key: str, request: Request):
    """Remove a remembered key from session memory."""
    service = get_service(request)
    session = service.memory.get_or_create_session(session_id)
    service.memory.remove_memory(session["id"], key)
    return schemas.MemoryResponse(
        session_id=session["id"], memories=service.db.get_memories(session["id"])
    )


@router.get("/history", response_model=schemas.HistoryResponse)
async def history(
    request: Request,
    session_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """Return conversation history."""
    return get_service(request).history(session_id, limit=limit, offset=offset)


@router.post("/summarize", response_model=schemas.SummarizeResponse)
async def summarize(payload: schemas.SummarizeRequest, request: Request):
    """Generate (or refresh) the structured session summary."""
    service = get_service(request)
    try:
        return await service.summarize(payload.session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
