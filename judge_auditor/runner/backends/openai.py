"""OpenAI (and OpenAI-compatible) chat-completions backend.

Uses ``httpx`` directly rather than the OpenAI SDK so the only hard dependency
is a small async HTTP client. Pointing ``base_url`` at a vLLM / Ollama / LM Studio
server makes this double as a local-model backend.
"""

from __future__ import annotations

import asyncio
import os
import random
import time

import httpx

from ...config import JudgeConfig
from ..protocol import BackendResponse

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAIBackend:
    """Async, retrying backend for the OpenAI chat-completions API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        max_retries: int = 5,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key: pass api_key= or set the OPENAI_API_KEY env var."
            )
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def call(
        self, messages: list[dict[str, str]], config: JudgeConfig
    ) -> BackendResponse:
        payload: dict[str, object] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.response_format is not None:
            payload["response_format"] = config.response_format

        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            try:
                resp = await self._client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:  # network / timeout
                last_exc = exc
                await self._backoff(attempt, retry_after=None)
                continue

            if resp.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                await self._backoff(attempt, retry_after=resp.headers.get("retry-after"))
                continue
            resp.raise_for_status()

            latency = time.monotonic() - start
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}
            return BackendResponse(
                text=text,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                latency_s=latency,
            )

        raise RuntimeError(
            f"OpenAI request failed after {self.max_retries + 1} attempts"
        ) from last_exc

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if retry_after is not None:
            try:
                await asyncio.sleep(float(retry_after))
                return
            except ValueError:
                pass
        # Exponential backoff with full jitter.
        delay = min(2.0**attempt, 30.0) * (0.5 + random.random() / 2)
        await asyncio.sleep(delay)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
