"""Aethra backend tests. Run with: python -m pytest tests -q"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag.embedder import HashedTFIDFEmbedder
from app.rag.ingest import chunk_text
from app.rag.vector_store import VectorStore
from app.storage.database import Database


# ----------------------------------------------------------------- fixtures

@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        embedding_backend="tfidf",
        summarize_threshold=100,  # keep tests deterministic
    )


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.sqlite3")
    yield database
    database.close()


class FakeLLM:
    """Deterministic stand-in for the local LLM client."""

    def __init__(self):
        self.last_messages = None

    @property
    def model(self):
        return "fake-model"

    @property
    def base_url(self):
        return "http://localhost:fake/v1"

    async def chat(self, messages, temperature=None, max_tokens=None):
        self.last_messages = messages
        return "FAKE-ANSWER"


@pytest.fixture()
def client(settings: Settings):
    app = create_app(settings)
    app.state.service.llm = FakeLLM()
    with TestClient(app) as test_client:  # context manager runs lifespan
        yield test_client


# ----------------------------------------------------------------- unit tests

def test_chunk_text_paragraph_split_and_overlap():
    text = "\n\n".join(f"paragraph {i} " + "x" * 200 for i in range(5))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) >= 4
    assert all(len(c) <= 310 for c in chunks)
    # long paragraph hard-split keeps all content
    long_para = "y" * 2500
    parts = chunk_text(long_para, size=400, overlap=100)
    assert "".join(p.strip("y") for p in parts).strip() == ""
    assert sum(len(p) for p in parts) >= 2500


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_hashed_tfidf_deterministic_and_norm():
    embedder = HashedTFIDFEmbedder(dim=4096)
    v1 = embedder.embed_one("kernel panic not syncing")
    v2 = embedder.embed_one("kernel panic not syncing")
    assert v1 == v2
    norm = sum(val * val for val in v1.values()) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_vector_store_retrieval_ordering(db: Database):
    embedder = HashedTFIDFEmbedder()
    store = VectorStore(db)
    db.create_session("s1")
    store.add_chunks("s1", None, "docker-docs", [
        "Docker containers are started with docker run and configured via docker-compose.",
        "The nl80211 subsystem governs wireless extensions in modern Linux kernels.",
        "Nginx reverse proxy configuration uses upstream blocks and proxy_pass.",
    ], embedder)
    results = store.search("start a docker container and check nginx proxy_pass", embedder, "s1", k=3)
    assert len(results) == 2  # the wireless doc has zero overlap and is filtered out
    texts = {r.text for r in results}
    assert any("Docker" in t for t in texts)
    assert any("Nginx" in t for t in texts)
    assert results[0].score > results[1].score


def test_vector_store_isolates_sessions(db: Database):
    embedder = HashedTFIDFEmbedder()
    store = VectorStore(db)
    db.create_session("s1")
    db.create_session("s2")
    store.add_chunks("s1", None, "a-doc", ["iptables rules for NAT masquerading"], embedder)
    assert store.search("iptables NAT", embedder, "s1", k=5)
    assert store.search("iptables NAT", embedder, "s2", k=5) == []


def test_troubleshooting_attempt_dedup(db: Database):
    db.create_session("s1")
    assert db.add_attempt("s1", "container fails to start", "checked logs") is True
    assert db.add_attempt("s1", "container fails to start", "checked logs") is False
    assert db.add_attempt("s1", "container fails to start", "verified ports") is True
    attempts = db.get_attempts("s1", "container fails to start")
    assert len(attempts) == 2


def test_memory_upsert(db: Database):
    db.create_session("s1")
    db.set_memory("s1", "os", "Kali Linux")
    db.set_memory("s1", "os", "Kali Linux 2026.1")
    assert db.get_memories("s1") == {"os": "Kali Linux 2026.1"}
    assert db.delete_memory("s1", "os") == 1


def test_prompt_builder_priority_order():
    from app.context.builder import ContextBundle
    from app.context.prompt_builder import build_messages, build_system_prompt

    bundle = ContextBundle(
        session={"id": "s1"},
        env_block="- os: Kali Linux",
        objective="Investigating wireless networking",
        summary="Earlier: configured monitor mode.",
        attempts_block="Problem: no signal\n  - tried: channel hop",
        terminal_block="$ iwconfig\nwlan0: no wireless extensions.",
        retrieved=[],
        recent_messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    )
    sections = bundle.to_sections()
    titles = [t for t, _ in sections]
    # Session memory first, troubleshooting before terminal output.
    assert titles.index("SESSION MEMORY (highest priority)") < titles.index(
        "RELEVANT DOCUMENTATION (retrieved via RAG)"
    )
    assert titles.index("TROUBLESHOOTING HISTORY (already attempted - do not repeat)") < titles.index(
        "RECENT TERMINAL OUTPUT"
    )

    messages = build_messages(bundle, "why is my adapter not detected?")
    assert messages[0]["role"] == "system"
    assert "Aethra" in messages[0]["content"]
    assert "Kali Linux" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "why is my adapter not detected?"}
    assert {"role": "assistant", "content": "hello"} in messages
    # Each section renders as "=== title ===" on its own line after the persona block.
    system_text = build_system_prompt(sections)
    assert system_text.count("\n=== ") == len(sections)


def test_extractive_fallback_summary(db, settings):
    import asyncio

    from app.memory.manager import MemoryManager
    from app.memory.summarizer import Summarizer

    class BrokenLLM:
        model = "x"
        base_url = "x"

        async def chat(self, messages, **kwargs):
            raise RuntimeError("LLM down")

    db.create_session("s1")
    db.add_message("s1", "user", "my container will not start")
    db.add_attempt("s1", "container fails", "checked logs")
    memory = MemoryManager(db, settings)
    summarizer = Summarizer(BrokenLLM(), memory, db)
    summary = asyncio.run(summarizer.summarize("s1"))
    assert "CURRENT OBJECTIVE" in summary
    assert "container fails" in summary
    # summary persisted, marker advanced
    session = db.get_session("s1")
    assert session["summary"] == summary
    assert session["summary_msg_id"] > 0


# ----------------------------------------------------------------- API tests

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "embedding_backend" in body


def test_chat_flow(client):
    resp = client.post("/chat", json={"message": "Why does my container exit immediately?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "FAKE-ANSWER"
    sid = body["session_id"]

    # session state exists with the exchange recorded
    state = client.get("/session", params={"session_id": sid}).json()
    roles = [m["role"] for m in state["recent_messages"]]
    assert roles == ["user", "assistant"]

    # history endpoint
    hist = client.get("/history", params={"session_id": sid}).json()
    assert hist["count"] == 2

    # second chat keeps the same session
    resp2 = client.post("/chat", json={"message": "More detail please", "session_id": sid})
    assert resp2.json()["session_id"] == sid


def test_memory_endpoint_and_prompt_injection(client):
    resp = client.post("/memory", json={
        "updates": {"os": "Kali Linux", "router": "TP-Link Archer AX23", "shell": "bash"}
    })
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    state = client.get("/session", params={"session_id": sid}).json()
    assert state["memories"]["os"] == "Kali Linux"
    assert state["memories"]["router"] == "TP-Link Archer AX23"

    chat = client.post("/chat", json={"message": "what router do I have?", "session_id": sid})
    assert chat.status_code == 200
    fake = client.app.state.service.llm
    system_prompt = fake.last_messages[0]["content"]
    assert "TP-Link Archer AX23" in system_prompt
    assert "SESSION MEMORY" in system_prompt

    # delete a memory key
    deleted = client.delete("/memory", params={"session_id": sid, "key": "shell"})
    assert deleted.status_code == 200
    assert "shell" not in deleted.json()["memories"]


def test_terminal_endpoint(client):
    resp = client.post("/terminal", json={
        "command": "systemctl start docker",
        "cwd": "/home/user/lab",
        "exit_code": 1,
        "output": "Job for docker.service failed because the control process exited with errors.",
        "explain": True,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["log_id"] > 0
    assert body["answer"] == "FAKE-ANSWER"
    fake = client.app.state.service.llm
    system = fake.last_messages[0]["content"]
    assert "systemctl start docker" in system
    assert "exit code: 1" in system


def test_upload_and_rag_retrieval(client):
    content = b"NTP synchronization on Linux is handled by systemd-timesyncd or chronyd. " * 20
    resp = client.post(
        "/upload",
        files={"file": ("notes.txt", content, "text/plain")},
        data={"session_id": "ragtest"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunks_indexed"] >= 1
    assert body["indexed"] is True

    state = client.get("/session", params={"session_id": "ragtest"}).json()
    assert state["documents"][0]["filename"] == "notes.txt"


def test_upload_rejects_binary(client):
    resp = client.post(
        "/upload",
        files={"file": ("prog.bin", bytes(range(256)), "application/octet-stream")},
    )
    assert resp.status_code == 415


def test_summarize_endpoint(client):
    client.post("/chat", json={"message": "Help me plan a Wi-Fi lab"})
    resp = client.post("/summarize", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]
    state = client.get("/session", params={"session_id": body["session_id"]}).json()
    assert state["summary"]
