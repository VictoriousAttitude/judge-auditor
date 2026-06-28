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


class Probe(StrEnum):
    """A prompt perturbation collected alongside the neutral judgment.

    The runner re-judges each example under a balanced *up*/*down* pair so the
    probe-bias analysis can isolate how far the verdict moves toward an injected
    suggestion (the swing between the two directions cancels per-example constants).

    * ``SYCOPHANCY_*`` injects a stated user opinion (scalar: a high/low score is
      deserved; pairwise: a named response is better — referenced by content).
    * ``ANCHOR_*`` injects an irrelevant numeric reference score (scalar only).
    """

    NEUTRAL = "neutral"
    SYCOPHANCY_UP = "sycophancy_up"
    SYCOPHANCY_DOWN = "sycophancy_down"
    ANCHOR_UP = "anchor_up"
    ANCHOR_DOWN = "anchor_down"


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
    # Optional same-intent paraphrases of ``prompt_template``. When supplied the runner
    # collects judgments under each one (variant 0 is ``prompt_template`` itself), so the
    # rubric-robustness analysis can check whether verdicts survive rephrasing.
    rubric_variants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode is JudgeMode.PAIRWISE:
            required = {"prompt", "response_a", "response_b"}
        else:
            required = {"prompt", "response"}
        for template in self.templates:
            missing = required - _template_fields(template)
            if missing:
                raise ValueError(
                    f"{self.mode.value} prompt_template is missing placeholders: "
                    f"{sorted(missing)} (found {sorted(_template_fields(template))})"
                )
        if self.mode is JudgeMode.SCALAR and self.score_min >= self.score_max:
            raise ValueError("score_min must be < score_max")

    @property
    def templates(self) -> tuple[str, ...]:
        """All rubric phrasings to audit: the base template followed by any variants."""
        return (self.prompt_template, *self.rubric_variants)


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
    # Optional ground-truth/expert quality score. Drives partial-correlation control
    # in verbosity-bias analysis and serves as the scalar validity target (does the
    # judge's score track the true quality?).
    quality_label: float | None = None
    # Optional ground-truth winner for pairwise validity: the response a human/expert
    # judged better. Lets the audit ask "does the judge agree with the truth?", not
    # just "does the judge agree with itself?".
    preferred_winner: Winner | None = None
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
    # Opt-in probe conditions (off by default => no extra calls, no behavior change).
    # When enabled the runner collects a balanced up/down probe pair per example at the
    # canonical rubric, feeding the sycophancy / anchoring detectors. Anchoring needs a
    # numeric scale, so it is ignored for pairwise judges.
    probe_sycophancy: bool = False
    probe_anchoring: bool = False

    def __post_init__(self) -> None:
        if self.runs_per_example < 1:
            raise ValueError("runs_per_example must be >= 1")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
