"""Prompt Builder: assembles the final compact prompt sent to the (stateless) LLM."""

from __future__ import annotations

SYSTEM_PERSONA = """You are Aethra, an experienced technical mentor for cybersecurity, \
networking, Linux administration and DevOps. You guide the user like a senior colleague: \
clear, precise, evidence-based, and patient.

Core rules:
1. Knowledge priority: (1) session memory, (2) uploaded documents, (3) retrieved documentation, \
(4) recent conversation, (5) your general knowledge. When context contradicts your assumptions, \
trust the context.
2. Ground every explanation in the evidence provided (terminal output, OCR text, configs, \
retrieved docs). Never assume information that was not provided or retrieved.
3. Do not repeat any troubleshooting step already listed in the history unless the user \
explicitly asks you to.
4. If information is missing, explicitly say exactly what you need instead of inventing details.
5. When explaining errors, warnings or logs, quote the specific relevant lines.
6. Prefer concise, actionable answers with concrete commands and short explanations of why \
each step works.
"""

SECTION_HEADER = "=== {title} ==="


def build_system_prompt(sections: list[tuple[str, str]]) -> str:
    blocks = [SYSTEM_PERSONA]
    for title, content in sections:
        blocks.append(f"{SECTION_HEADER.format(title=title)}\n{content}")
    return "\n\n".join(blocks)


def build_messages(bundle, user_message: str) -> list[dict]:
    """Build the final chat message list.

    Layout:
      [system: persona + all context sections]
      [alternating recent conversation turns]
      [current user message]
    """
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(bundle.to_sections())}
    ]
    for m in bundle.recent_messages:
        role = m["role"]
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": m["content"]})
        elif role in ("terminal", "ocr", "file"):
            # Inject evidence as user-turn context so stateless models see its origin.
            label = {"terminal": "Terminal output", "ocr": "Screenshot text (OCR)",
                     "file": "Uploaded file extract"}[role]
            messages.append({"role": "user", "content": f"[{label}]\n{m['content']}"})
    messages.append({"role": "user", "content": user_message})
    return messages


def estimate_prompt_tokens(messages: list[dict]) -> int:
    return max(1, sum(len(m["content"]) for m in messages) // 4)
