"""Anthropic Messages API backend.

Like the OpenAI backend, this talks to the HTTP API directly via ``httpx`` (no SDK),
so it adds no heavy dependency. The Messages API differs from chat-completions in two
ways we handle here: the system prompt is a top-level ``system`` field (not a message
with ``role: system``), and ``max_tokens`` is required.
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
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicBackend:
    """Async, retrying backend for the Anthropic Messages API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        max_retries: int = 5,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key: pass api_key= or set the ANTHROPIC_API_KEY env var."
            )
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @staticmethod
    def _split_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
        """Pull system messages out into a single system string (Messages API shape)."""
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        convo = [m for m in messages if m.get("role") != "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        return system, convo

    async def call(
        self, messages: list[dict[str, str]], config: JudgeConfig
    ) -> BackendResponse:
        system, convo = self._split_system(messages)
        payload: dict[str, object] = {
            "model": config.model,
            "messages": convo,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
        if system is not None:
            payload["system"] = system

        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        url = f"{self.base_url}/v1/messages"

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
            blocks = data.get("content") or []
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            usage = data.get("usage") or {}
            return BackendResponse(
                text=text,
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                latency_s=latency,
            )

        raise RuntimeError(
            f"Anthropic request failed after {self.max_retries + 1} attempts"
        ) from last_exc

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        if retry_after is not None:
            try:
                await asyncio.sleep(float(retry_after))
                return
            except ValueError:
                pass
        delay = min(2.0**attempt, 30.0) * (0.5 + random.random() / 2)
        await asyncio.sleep(delay)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
