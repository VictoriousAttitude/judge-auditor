"""Aggregator: run every analysis module over one judgment set and compose the
results into a single :class:`ReliabilityReport`, plus an overall verdict.

This is the Phase-2 standalone milestone — given a :class:`JudgmentSet` (e.g. from
the mock backend, no API key needed) and the originating examples, ``audit`` returns
the full diagnostic the report layer renders.

The overall verdict is deliberately conservative: it starts from the headline
self-consistency band and is *downgraded* (never upgraded) by any bias or scale
problem, because an unreliable signal cannot be rescued by the absence of one bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import EvalExample, JudgeMode
from ..records import JudgmentSet
from .consistency import ConsistencyResult, consistency
from .position_bias import PositionBiasResult, position_bias
from .power_analysis import PowerAnalysisResult, power_analysis
from .scale_analysis import ScaleAnalysisResult, scale_analysis
from .verbosity_bias import VerbosityBiasResult, verbosity_bias

# Overall reliability labels, ordered worst -> best.
RELIABILITY_LEVELS = ("LOW", "MODERATE", "HIGH")


@dataclass
class ReliabilityReport:
    model: str
    mode: JudgeMode
    n_examples: int
    n_records: int
    parse_failure_rate: float

    consistency: ConsistencyResult
    verbosity: VerbosityBiasResult
    scale: ScaleAnalysisResult
    power: PowerAnalysisResult
    position: PositionBiasResult | None = None  # pairwise only

    overall: str = "LOW"
    notes: list[str] = field(default_factory=list)


def _headline_point(c: ConsistencyResult) -> float | None:
    if c.mode is JudgeMode.SCALAR:
        return c.icc_oneway.point if c.icc_oneway is not None else None
    return c.fleiss_kappa.point if c.fleiss_kappa is not None else None


def _base_level(mode: JudgeMode, point: float | None) -> str:
    """Headline self-consistency band (ICC for scalar, kappa for pairwise)."""
    if point is None:
        return "LOW"
    if mode is JudgeMode.SCALAR:
        if point >= 0.75:
            return "HIGH"
        return "MODERATE" if point >= 0.50 else "LOW"
    # Pairwise (Landis & Koch kappa bands).
    if point >= 0.60:
        return "HIGH"
    return "MODERATE" if point >= 0.40 else "LOW"


def _downgrade(level: str, to: str) -> str:
    """Return the worse (lower) of two levels."""
    return level if RELIABILITY_LEVELS.index(level) <= RELIABILITY_LEVELS.index(to) else to


def audit(
    js: JudgmentSet,
    examples: list[EvalExample],
    *,
    score_min: float = 1.0,
    score_max: float = 10.0,
    lengths: dict[str, int] | None = None,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ReliabilityReport:
    """Compose all analysis modules into a single reliability report."""
    cons = consistency(js, n_boot=n_boot, confidence=confidence, seed=seed)
    verb = verbosity_bias(js, examples, lengths=lengths)
    scale = scale_analysis(js, score_min=score_min, score_max=score_max)
    power = power_analysis(js)
    pos = position_bias(js) if js.mode is JudgeMode.PAIRWISE else None

    level = _base_level(js.mode, _headline_point(cons))
    notes: list[str] = []

    if verb.flagged:
        notes.append(
            f"Verbosity bias: score/length Spearman rho={verb.spearman_rho:.2f} "
            f"(|rho| > {verb.threshold})."
        )
        level = _downgrade(level, "MODERATE")
    if scale.mode is JudgeMode.SCALAR and scale.compressed:
        notes.append(
            f"Compressed scale: {scale.max_window_fraction:.0%} of scores in "
            f"{len(scale.compressed_values)} adjacent bins."
        )
        level = _downgrade(level, "MODERATE")
    if scale.mode is JudgeMode.PAIRWISE and scale.indistinguishable:
        notes.append(f"High tie rate: {scale.tie_rate:.0%} of comparisons are ties.")
        level = _downgrade(level, "MODERATE")
    if pos is not None and pos.favored_position != "none":
        notes.append(
            f"Position bias toward the {pos.favored_position}-presented response "
            f"(flip rate {pos.flip_rate.point:.0%})."
        )
        level = _downgrade(level, "LOW" if pos.flip_rate.point > 0.20 else "MODERATE")

    return ReliabilityReport(
        model=js.model,
        mode=js.mode,
        n_examples=len(js.example_ids),
        n_records=len(js.records),
        parse_failure_rate=js.parse_failure_rate,
        consistency=cons,
        verbosity=verb,
        scale=scale,
        power=power,
        position=pos,
        overall=level,
        notes=notes,
    )
