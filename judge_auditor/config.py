"""Core configuration and input data models for Judge Auditor.

These types describe *what the user gives us*: the judge they want audited
(:class:`JudgeConfig`), the items to evaluate (:class:`EvalExample`), and how
thoroughly to run the audit (:class:`AuditConfig`). Runtime outputs (collected
judgments) live in :mod:`judge_auditor.records`.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class JudgeMode(StrEnum):
    """The two dominant LLM-as-judge patterns we support."""

    PAIRWISE = "pairwise"  # (prompt, A, B) -> winner; MT-Bench / Arena style
    SCALAR = "scalar"  # (prompt, response) -> numeric score; G-Eval style


class Winner(StrEnum):
    """Canonical pairwise verdict, resolved to the *content* (not position)."""

    A = "A"
    B = "B"
    TIE = "tie"


class PairwiseChoice(StrEnum):
    """Which *presented position* the judge picked, before mapping to A/B.

    Using position (rather than the A/B label directly) keeps the parser
    independent of how the executor laid out the two responses, which is what
    makes position-bias measurement possible.
    """

    FIRST = "first"
    SECOND = "second"
    TIE = "tie"


def _template_fields(template: str) -> set[str]:
    """Return the set of ``{placeholder}`` names referenced by a format string."""
    return {
        name for _, name, _, _ in string.Formatter().parse(template) if name
    }


@dataclass(frozen=True)
class JudgeConfig:
    """A complete description of the judge under audit.

    The ``prompt_template`` is a :meth:`str.format` template. Required
    placeholders depend on the mode:

    * pairwise: ``{prompt}``, ``{response_a}``, ``{response_b}``
    * scalar:   ``{prompt}``, ``{response}``
    """

    model: str
    prompt_template: str
    mode: JudgeMode
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    # Structured-output schema passed straight through to the backend, if set.
    response_format: dict[str, Any] | None = None
    # Scalar score bounds, used by the parser and downstream scale analysis.
    score_min: float = 1.0
    score_max: float = 10.0

    def __post_init__(self) -> None:
        fields = _template_fields(self.prompt_template)
        if self.mode is JudgeMode.PAIRWISE:
            required = {"prompt", "response_a", "response_b"}
        else:
            required = {"prompt", "response"}
        missing = required - fields
        if missing:
            raise ValueError(
                f"{self.mode.value} prompt_template is missing placeholders: "
                f"{sorted(missing)} (found {sorted(fields)})"
            )
        if self.mode is JudgeMode.SCALAR and self.score_min >= self.score_max:
            raise ValueError("score_min must be < score_max")


@dataclass(frozen=True)
class EvalExample:
    """One item to feed the judge.

    In scalar mode only ``response_a`` is used (as the single response);
    ``response_b`` must be ``None``.
    """

    id: str
    prompt: str
    response_a: str
    response_b: str | None = None
    # Optional ground-truth/expert quality score, enabling partial-correlation
    # control in the verbosity-bias analysis.
    quality_label: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditConfig:
    """Knobs controlling how thoroughly the audit runs.

    ``runs_per_example`` (K) drives the reliability estimates; K=15 is the
    default sweet spot from the plan (tight ICC/kappa CIs at manageable cost).
    """

    runs_per_example: int = 15
    max_concurrency: int = 10
    # Audit a stratified random sample of M examples instead of the full set.
    sample_size: int | None = None
    seed: int = 0
    # Checkpoint to a JSONL file and resume from it if interrupted.
    checkpoint_path: str | None = None
    # Flush completed records to the checkpoint after this many tasks.
    checkpoint_every: int = 20

    def __post_init__(self) -> None:
        if self.runs_per_example < 1:
            raise ValueError("runs_per_example must be >= 1")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
