"""The backend contract.

A backend is anything that turns a chat message list into text. The runner is
deliberately backend-agnostic: OpenAI, Anthropic, and local models all implement
the same :class:`JudgeBackend` protocol, so the executor and analysis layers never
know which model produced a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..config import JudgeConfig


@dataclass
class BackendResponse:
    """A judge's reply plus the telemetry we need for cost and latency reporting."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None


@runtime_checkable
class JudgeBackend(Protocol):
    """Implemented by every backend (OpenAI, Anthropic, local, mock)."""

    async def call(
        self, messages: list[dict[str, str]], config: JudgeConfig
    ) -> BackendResponse: ...
