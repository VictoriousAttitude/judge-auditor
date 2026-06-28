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

from ..config import EvalExample, JudgeMode, Probe
from ..records import JudgmentSet
from .consistency import ConsistencyResult, consistency
from .position_bias import PositionBiasResult, position_bias
from .power_analysis import PowerAnalysisResult, power_analysis
from .probe_bias import ProbeBiasResult, probe_bias
from .rubric_robustness import RubricRobustnessResult, rubric_robustness
from .scale_analysis import ScaleAnalysisResult, scale_analysis
from .validity import ValidityResult, validity
from .verbosity_bias import VerbosityBiasResult, verbosity_bias

# Overall reliability labels, ordered worst -> best.
RELIABILITY_LEVELS = ("LOW", "MODERATE", "HIGH")


def _primary_subset(js: JudgmentSet) -> JudgmentSet:
    """The canonical-rubric (variant 0) judgments the headline metrics are computed on.

    Identical to ``js`` for a plain audit; when extra rubric variants or probe
    conditions were run they are excluded here so neither paraphrase disagreement nor
    injected probes leak into self-consistency (each is measured by its own detector)."""
    primary = [r for r in js.records if r.rubric_variant == 0 and r.probe is Probe.NEUTRAL]
    if len(primary) == len(js.records):
        return js
    return JudgmentSet(mode=js.mode, model=js.model, records=primary)


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
    validity: ValidityResult
    position: PositionBiasResult | None = None  # pairwise only
    rubric: RubricRobustnessResult = field(
        default_factory=lambda: RubricRobustnessResult(mode=JudgeMode.SCALAR)
    )
    probe: ProbeBiasResult = field(
        default_factory=lambda: ProbeBiasResult(mode=JudgeMode.SCALAR)
    )

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


def _validity_severity(val: ValidityResult) -> str:
    """How far to downgrade for poor validity: the worst bands earn a LOW."""
    if val.mode is JudgeMode.SCALAR:
        return "LOW" if val.interpretation == "poor" else "MODERATE"
    # Pairwise: a near-zero (poor/slight) Cohen's kappa is no-better-than-chance.
    return "LOW" if val.interpretation in ("poor", "slight") else "MODERATE"


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
    primary = _primary_subset(js)
    cons = consistency(primary, n_boot=n_boot, confidence=confidence, seed=seed)
    verb = verbosity_bias(primary, examples, lengths=lengths)
    scale = scale_analysis(primary, score_min=score_min, score_max=score_max)
    power = power_analysis(primary)
    valid = validity(primary, examples, n_boot=n_boot, confidence=confidence, seed=seed)
    rob = rubric_robustness(js, n_boot=n_boot, confidence=confidence, seed=seed)
    prb = probe_bias(
        js,
        score_min=score_min,
        score_max=score_max,
        n_boot=n_boot,
        confidence=confidence,
        seed=seed,
    )
    pos = position_bias(primary) if js.mode is JudgeMode.PAIRWISE else None

    level = _base_level(js.mode, _headline_point(cons))
    notes: list[str] = []

    if verb.flagged:
        v_rho = verb.spearman_rho if verb.mode is JudgeMode.SCALAR else verb.length_winrate_rho
        v_p = verb.spearman_p if verb.mode is JudgeMode.SCALAR else verb.length_winrate_p
        notes.append(
            f"Verbosity bias: length Spearman rho={v_rho:.2f} "
            f"(p={v_p:.3f}, n={verb.n_examples}; flagged at |rho|>{verb.threshold}, "
            f"p<{verb.p_threshold})."
        )
        level = _downgrade(level, "MODERATE")
    if verb.stratified_flagged and verb.strata is not None:
        worst = max(verb.strata, key=lambda s: abs(s.score_gap))
        notes.append(
            f"Verbosity bias within quality={worst.quality:g}: longer responses score "
            f"{worst.score_gap:+.1f} pts vs shorter (rho={worst.spearman_rho:.2f}, "
            f"p={worst.spearman_p:.3f}, n={worst.n}) — an interaction the global "
            "correlation masks."
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
    if valid.flagged:
        if valid.mode is JudgeMode.SCALAR and valid.pearson_r is not None:
            notes.append(
                f"Low validity: judge score tracks ground truth at Pearson r="
                f"{valid.pearson_r.point:.2f} "
                f"[{valid.pearson_r.low:.2f}, {valid.pearson_r.high:.2f}] "
                f"(Spearman rho={valid.spearman_rho:.2f}, n={valid.n_labeled}) — a "
                "self-consistent judge can still be consistently wrong."
            )
        elif valid.cohen_kappa is not None:
            notes.append(
                f"Low validity: judge agrees with ground truth at Cohen's kappa="
                f"{valid.cohen_kappa.point:.2f} "
                f"[{valid.cohen_kappa.low:.2f}, {valid.cohen_kappa.high:.2f}] "
                f"(agreement {valid.agreement_rate:.0%}, n={valid.n_labeled}) — "
                "self-consistency does not imply correctness."
            )
        level = _downgrade(level, _validity_severity(valid))
    if rob.available and rob.flagged:
        if rob.mode is JudgeMode.SCALAR and rob.icc is not None:
            notes.append(
                f"Rubric brittleness: paraphrasing the rubric moves scores "
                f"(cross-variant ICC={rob.icc.point:.2f} "
                f"[{rob.icc.low:.2f}, {rob.icc.high:.2f}], mean spread "
                f"{rob.mean_score_spread:.1f} pts across {rob.n_variants} variants) — "
                "the verdict partly reflects phrasing, not response quality."
            )
        elif rob.kappa is not None:
            notes.append(
                f"Rubric brittleness: the winner flips across rubric phrasings "
                f"({rob.winner_flip_rate:.0%} of examples; cross-variant Fleiss "
                f"kappa={rob.kappa.point:.2f} [{rob.kappa.low:.2f}, {rob.kappa.high:.2f}] "
                f"over {rob.n_variants} variants) — the verdict partly reflects phrasing."
            )
        level = _downgrade(level, "LOW" if rob.severe else "MODERATE")
    for effect in prb.effects:
        if not effect.flagged:
            continue
        if effect.mode is JudgeMode.SCALAR and effect.raw_pts is not None:
            cue = (
                "stating a desired score"
                if effect.kind == "sycophancy"
                else "an irrelevant reference score"
            )
            notes.append(
                f"{effect.kind.capitalize()} bias: {cue} moves the judge "
                f"{effect.raw_pts:+.1f} pts ({effect.effect.point:+.0%} of scale) "
                f"[{effect.effect.low:+.0%}, {effect.effect.high:+.0%}] toward the "
                f"suggestion (n={effect.n_examples})."
            )
        else:
            notes.append(
                f"{effect.kind.capitalize()} bias: stating a preferred response swings "
                f"the win rate {effect.effect.point:+.0%} "
                f"[{effect.effect.low:+.0%}, {effect.effect.high:+.0%}] toward it "
                f"(n={effect.n_examples}) — the verdict tracks the suggestion."
            )
        level = _downgrade(level, "LOW" if effect.severe else "MODERATE")

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
        validity=valid,
        position=pos,
        rubric=rob,
        probe=prb,
        overall=level,
        notes=notes,
    )
