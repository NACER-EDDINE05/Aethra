"""OpenAI-compatible client for a locally hosted LLM."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("aethra.llm")


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Talks to any local OpenAI-compatible /chat/completions endpoint
    (Ollama, LM Studio, llama.cpp server, vLLM, ...)."""

    def __init__(self, settings):
        self._settings = settings

    @property
    def model(self) -> str:
        return self._settings.llm_model

    @property
    def base_url(self) -> str:
        return self._settings.llm_base_url

    async def chat(self, messages: list[dict], temperature: float | None = None,
                   max_tokens: int | None = None) -> str:
        """Send a fully assembled prompt to the local LLM and return its reply text."""
        payload = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": self._settings.llm_temperature if temperature is None else temperature,
            "max_tokens": self._settings.llm_max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self._settings.llm_api_key and self._settings.llm_api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self._settings.llm_api_key}"

        url = f"{self._settings.llm_base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._settings.llm_timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach the local LLM API at {url}. "
                f"Is your local model server running? ({exc})"
            ) from exc
        except ValueError as exc:
            raise LLMError(f"LLM API returned invalid JSON: {exc}") from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected LLM API response structure: {data!r}") from exc

        logger.info("LLM call ok (model=%s, prompt_msgs=%d)", self._settings.llm_model, len(messages))
        return content.strip()


def count_tokens_approx(text: str) -> int:
    return max(1, len(text) // 4)
