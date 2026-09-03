"""Context Builder: gathers every relevant piece of state before a request.

Knowledge priority order (per spec):
1. Session memory  2. Uploaded documents  3. Retrieved documentation
4. Recent conversation  5. LLM general knowledge (implicit - lowest priority)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContextBundle:
    session: dict
    env_block: str
    objective: str
    summary: str
    attempts_block: str
    terminal_block: str
    retrieved: list = field(default_factory=list)
    recent_messages: list = field(default_factory=list)

    def retrieved_block(self) -> str:
        if not self.retrieved:
            return "(no indexed documentation matched this question)"
        parts = []
        for i, chunk in enumerate(self.retrieved, start=1):
            parts.append(f"[{i}] source: {chunk.source} (relevance: {chunk.score:.2f})\n{chunk.text}")
        return "\n\n".join(parts)

    def to_sections(self) -> list[tuple[str, str]]:
        """Ordered (title, content) sections following the knowledge priority order."""
        return [
            ("SESSION MEMORY (highest priority)", self.env_block),
            ("CURRENT OBJECTIVE", self.objective or "(not set)"),
            ("RELEVANT DOCUMENTATION (retrieved via RAG)", self.retrieved_block()),
            ("SESSION SUMMARY (earlier conversation)", self.summary or "(no summary yet)"),
            ("TROUBLESHOOTING HISTORY (already attempted - do not repeat)", self.attempts_block),
            ("RECENT TERMINAL OUTPUT", self.terminal_block),
        ]
