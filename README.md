# Aethra

**Aethra** is a backend-only AI assistant for cybersecurity, networking, Linux
administration and DevOps learning & troubleshooting. It talks to a **locally
hosted LLM** through its API and keeps the model completely stateless — all
intelligence (memory, context, retrieval, prompt assembly) lives in this backend.

```
User
 ↓
Backend API (FastAPI)
 ↓
Context Builder
 ↓
Memory Manager        RAG Engine
 ↓                     ↓
Prompt Builder  ←──────┘
 ↓
Local LLM API  (Ollama / LM Studio / llama.cpp / vLLM — OpenAI-compatible)
 ↓
Response
```

## Features

- **Sessions** — each conversation owns an ID, title, objective, environment
  memory, uploaded files, OCR text, summary and recent messages.
- **Long-term memory** — remembers OS, distro, router, VMs, topology, shell,
  tools, projects, learning goals; no need to repeat yourself.
- **Short-term memory** — keeps the last N interactions; auto-summarizes
  (structured summary) when history grows, preventing context overflow.
- **Troubleshooting memory** — structured attempts list; never suggests the
  same step twice unless explicitly asked.
- **Terminal context** — store command, cwd, output, exit code; ask for an
  explanation grounded in that evidence.
- **Screenshot OCR** — extracts terminal text / errors / configs from images
  (optional: `pytesseract` + Tesseract binary).
- **File uploads** — PDF, TXT, MD, JSON, YAML, INI, CONF, LOG, XML, CSV and
  other text formats; parsed, chunked, embedded and indexed.
- **RAG** — retrieval-augmented generation over a local vector store.
  Prioritizes retrieved documentation over model general knowledge.
- **Knowledge priority**: session memory → uploaded docs → retrieved docs →
  recent conversation → LLM general knowledge.

## Quick start

```powershell
pip install -r requirements.txt

# 1. serve the local GGUF model (OpenAI-compatible API on http://localhost:8080/v1):
powershell -ExecutionPolicy Bypass -File .\serve_model.ps1

# 2. in a second terminal, start the backend (reads .env automatically):
python run.py            # serves http://127.0.0.1:8000
```

Interactive API docs: `http://127.0.0.1:8000/docs` · Model playground: `http://localhost:8080`

Quick model chat without Aethra: `powershell -ExecutionPolicy Bypass -File .\chat.ps1 "what is a reverse shell?"`

`.env` (project root) holds the configuration — `AETHRA_LLM_BASE_URL=http://localhost:8080/v1`
and `AETHRA_LLM_MODEL=WhiteRabbitNeo-V3-7B-IQ3_M` by default; existing environment
variables always take priority over `.env`.


### Optional extras

| Feature              | Install                                                     |
|----------------------|-------------------------------------------------------------|
| Semantic embeddings  | `pip install sentence-transformers` (auto-detected)          |
| PDF uploads          | `pip install pypdf`                                         |
| Screenshot OCR       | `pip install pytesseract Pillow` + [Tesseract binary](https://github.com/UB-Mannheim/tesseract/wiki) |

Without them Aethra still works: hashed TF-IDF retrieval, non-PDF uploads, and
clear 503 errors for OCR.

## Configuration (env vars, see `.env.example`)

| Variable | Default | Purpose |
|---|---|---|
| `AETHRA_LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible local LLM endpoint |
| `AETHRA_LLM_MODEL` | `qwen2.5:7b` | Model name |
| `AETHRA_LLM_API_KEY` | `not-needed` | Bearer token if your server needs one |
| `AETHRA_EMBEDDING_BACKEND` | `auto` | `auto` / `st` / `tfidf` |
| `AETHRA_TOP_K` | `5` | Chunks retrieved per query |
| `AETHRA_RECENT_WINDOW` | `12` | Recent messages kept in every prompt |
| `AETHRA_SUMMARIZE_THRESHOLD` | `24` | Messages before auto-summarization |
| `AETHRA_DATA_DIR` | `./data` | SQLite database location |

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | `{message, session_id?, use_rag?}` → answer + retrieved sources |
| POST | `/upload` | multipart `file` + `session_id?` → parse, chunk, index |
| POST | `/terminal` | `{output, command?, cwd?, exit_code?, explain?}` → store (+ explain) |
| POST | `/screenshot` | multipart `file` (+`question?`) → OCR text (+ answer) |
| GET  | `/session` | Session state: memory, summary, troubleshooting, docs, terminal |
| POST | `/memory` | `{updates: {"os": "Kali Linux", ...}}` → update environment memory |
| GET  | `/history` | Full conversation history (`limit`, `offset`) |
| POST | `/summarize` | Generate/refresh the structured session summary |
| GET  | `/health` | Service + configuration info |
| DELETE | `/memory?session_id&key` | Forget one memory key |

### Example session

```bash
# 1. teach Aethra your environment
curl -X POST localhost:8000/memory -H "Content-Type: application/json" \
  -d '{"updates":{"os":"Kali Linux","router":"TP-Link Archer AX23","wifi":"Alfa AWUS036ACH","shell":"bash"}}'

# 2. upload a failed command's output
curl -X POST localhost:8000/terminal -H "Content-Type: application/json" \
  -d '{"command":"airmon-ng start wlan0","exit_code":1,"explain":true,"output":"..."}'

# 3. index documentation
curl -X POST localhost:8000/upload -F file=@owasp-cheatsheet.pdf

# 4. chat — answers use your memory, terminal evidence and indexed docs
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"message":"why did monitor mode fail?"}'
```

## Logging

Every request is logged with timestamp, request ID, path, status and duration;
each chat additionally logs session ID, retrieved-document count and approximate
token counts. Message bodies, credentials and secrets are never logged.

## Tests

```powershell
python -m pytest tests -q
```

## Project layout

```
app/
├── api/routes.py          # HTTP endpoints
├── config.py              # env-driven settings
├── context/               # context builder + prompt builder
├── llm/client.py          # OpenAI-compatible local LLM client
├── logging_setup.py
├── memory/                # manager (short/long-term, troubleshooting) + summarizer
├── models/schemas.py      # Pydantic request/response models
├── ocr/ocr.py             # Tesseract OCR wrapper
├── rag/                   # embedder, vector store, ingest/chunking, RAG engine
├── services/orchestrator.py  # full request pipeline
└── storage/database.py    # SQLite persistence
```
