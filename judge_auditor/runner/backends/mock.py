"""In-process mock backend for tests, CI smoke runs, and validation.

A ``MockBackend`` wraps a plain function ``(messages, config) -> str``. The
``responders`` module below provides ready-made judge "model organisms" used by
the test suite and the null/known-bias validation runs: a perfectly consistent
judge, a position-biased judge, a noisy judge, etc.
"""

from __future__ import annotations

from collections.abc import Callable

from ...config import JudgeConfig
from ..protocol import BackendResponse

Responder = Callable[[list[dict[str, str]], JudgeConfig], str]


class MockBackend:
    """Deterministic-by-default backend driven by a user-supplied responder."""

    def __init__(self, responder: Responder) -> None:
        self._responder = responder
        self.calls: list[list[dict[str, str]]] = []

    async def call(
        self, messages: list[dict[str, str]], config: JudgeConfig
    ) -> BackendResponse:
        self.calls.append(messages)
        text = self._responder(messages, config)
        # Rough token accounting so cost/telemetry paths exercise real numbers.
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages)
        return BackendResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text.split()),
            latency_s=0.0,
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)
