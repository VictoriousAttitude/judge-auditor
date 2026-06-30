"""Turn a :class:`ReliabilityReport` into concrete, actionable advice.

A diagnostic that only says "your judge is unreliable" is useless; the value is in
what to *do*. Each rule below fires on a measured problem and emits the mitigation
from the plan (section 7.3). A clean report yields an empty list — the report layer
renders that as "no action needed".
"""

from __future__ import annotations

import math

from ..analysis.audit import ReliabilityReport
from ..config import JudgeMode

# Surface parse trouble once it stops being negligible.
_PARSE_FAILURE_FLOOR = 0.05


def _consistency_rec(report: ReliabilityReport) -> str | None:
    c = report.consistency
    if c.mode is JudgeMode.SCALAR:
        ci = c.icc_oneway
        if ci is None or math.isnan(ci.point) or ci.point >= 0.75:
            return None
        return (
            f"Self-consistency is {c.icc_interpretation} (ICC={ci.point:.2f}). "
            "Increase runs_per_example, simplify the rubric, or switch to a more "
            "capable judge model before trusting individual scores."
        )
    ci = c.fleiss_kappa
    if ci is None or math.isnan(ci.point) or ci.point >= 0.60:
        return None
    return (
        f"Self-consistency is {c.kappa_interpretation} (kappa={ci.point:.2f}). "
        "The judge disagrees with itself across runs; average more runs per pair or "
        "use a stronger judge model before trusting verdicts."
    )


def _position_rec(report: ReliabilityReport) -> str | None:
    p = report.position
    if p is None or p.favored_position == "none":
        return None
    return (
        f"Position bias toward the {p.favored_position}-presented response "
        f"(flip rate {p.flip_rate.point:.0%}). Consider (a) best-of-N with position "
        "swapping — run both orderings and count only the agreements; or (b) switching "
        "to scalar mode, which avoids order effects."
    )


def _verbosity_rec(report: ReliabilityReport) -> str | None:
    v = report.verbosity
    if v.flagged:
        if v.mode is JudgeMode.SCALAR and v.spearman_rho is not None:
            return (
                f"Verbosity correlation rho={v.spearman_rho:.2f}: the judge rewards length. "
                "Add an explicit 'ignore response length' instruction to the rubric, or "
                "length-normalize responses before scoring."
            )
        return (
            "The longer response wins disproportionately often. Add an 'ignore response "
            "length' instruction to the rubric, or control for length before comparing."
        )
    if v.stratified_flagged and v.strata:
        worst = max(v.strata, key=lambda s: abs(s.score_gap))
        return (
            f"Among equal-quality answers (quality={worst.quality:g}), the judge scores "
            f"longer responses {worst.score_gap:+.1f} pts differently — a length effect "
            "hidden in the overall correlation. Add an 'ignore response length' "
            "instruction to the rubric, or length-normalize before scoring."
        )
    return None


def _validity_rec(report: ReliabilityReport) -> str | None:
    v = report.validity
    if not v.available or not v.flagged:
        return None
    if v.mode is JudgeMode.SCALAR and v.pearson_r is not None:
        return (
            f"Low validity: judge scores correlate with ground truth only at "
            f"r={v.pearson_r.point:.2f} ({v.interpretation}). Self-consistency measures "
            "precision, not correctness — recalibrate the rubric against your labels, or "
            "use a stronger judge model, before trusting the scores."
        )
    if v.cohen_kappa is not None:
        return (
            f"Low validity: judge verdicts agree with ground truth at only "
            f"kappa={v.cohen_kappa.point:.2f} ({v.interpretation}, {v.agreement_rate:.0%} "
            "raw agreement). A self-consistent judge can still be wrong — revise the "
            "rubric against your labels, or use a stronger judge model."
        )
    return None  # pragma: no cover - a flagged result always carries its metric


def _rubric_rec(report: ReliabilityReport) -> str | None:
    rb = report.rubric
    if not rb.available or not rb.flagged:
        return None
    if rb.mode is JudgeMode.SCALAR and rb.icc is not None:
        return (
            f"Rubric brittleness: rephrasing the rubric moves scores (cross-variant "
            f"ICC={rb.icc.point:.2f}, mean spread {rb.mean_score_spread:.1f} pts). Pin "
            "one canonical rubric wording, make the criteria more concrete, or average "
            "over several phrasings — the verdict should not depend on phrasing."
        )
    if rb.kappa is not None:
        return (
            f"Rubric brittleness: the winner flips on {rb.winner_flip_rate:.0%} of "
            f"examples when the rubric is rephrased (cross-variant kappa={rb.kappa.point:.2f}). "
            "Pin one canonical rubric wording or sharpen the comparison criteria so the "
            "verdict is phrasing-independent."
        )
    return None  # pragma: no cover - a flagged result always carries its metric


def _probe_rec(report: ReliabilityReport) -> str | None:
    pb = report.probe
    if not pb.available or not pb.flagged:
        return None
    flagged = [e for e in pb.effects if e.flagged]
    if any(e.kind == "sycophancy" for e in flagged):
        e = next(e for e in flagged if e.kind == "sycophancy")
        if e.mode is JudgeMode.SCALAR and e.raw_pts is not None:
            return (
                f"Sycophancy: stating a desired score moves the judge {e.raw_pts:+.1f} pts "
                f"({e.effect.point:+.0%} of scale). Strip user opinions from the input, or "
                "instruct the judge to ignore stated preferences and grade only the response."
            )
        return (
            f"Sycophancy: stating a preference swings the win rate {e.effect.point:+.0%}. "
            "Withhold any stated user preference from the judge, or instruct it to grade "
            "only the responses, not the user's opinion."
        )
    e = next(e for e in flagged if e.kind == "anchoring")
    return (
        f"Anchoring: an irrelevant reference score moves the judge {e.raw_pts:+.1f} pts "
        f"({e.effect.point:+.0%} of scale). Remove prior/reference scores from the prompt "
        "so the judge grades the response on its own merits."
    )


def _scale_rec(report: ReliabilityReport) -> str | None:
    s = report.scale
    if s.mode is JudgeMode.SCALAR and s.compressed:
        bins = len(s.compressed_values)
        return (
            f"Scale is compressed ({s.max_window_fraction:.0%} of scores in {bins} "
            "adjacent values). Reduce the scale to 3-5 points, or use a comparative "
            "(pairwise) rubric instead of absolute scoring."
        )
    if s.mode is JudgeMode.PAIRWISE and s.indistinguishable:
        return (
            f"Tie rate is high ({s.tie_rate:.0%}): the judge cannot separate most "
            "pairs. Forbid ties in the rubric, or sharpen the comparison criteria."
        )
    return None


def _power_rec(report: ReliabilityReport) -> str | None:
    p = report.power
    if p.mode is JudgeMode.SCALAR:
        if p.mde is None or math.isnan(p.mde):
            return None
        return (
            f"Noise floor: the minimum reliably detectable score difference is "
            f"{p.mde:.2f} points at n={p.n_examples}. Treat smaller A/B differences as "
            "within judge noise — do not ship decisions based on them."
        )
    if p.mde_winrate is None:
        return None
    if math.isinf(p.mde_winrate):
        return (
            "The judge has no discriminating power (it flips as often as it agrees): "
            "no number of comparisons can reliably detect a winner. Fix consistency first."
        )
    return (
        f"Noise floor: the minimum reliably detectable win-rate margin is "
        f"{p.mde_winrate:.3f} at n={p.n_examples}. Treat smaller margins as within "
        "judge noise."
    )


def _parse_rec(report: ReliabilityReport) -> str | None:
    if report.parse_failure_rate <= _PARSE_FAILURE_FLOOR:
        return None
    return (
        f"Parse-failure rate is {report.parse_failure_rate:.0%}: the judge often emits "
        "an unparseable verdict. Use structured/JSON output mode or tighten the rubric's "
        "output-format instruction — unparsed verdicts are silently lost signal."
    )


def recommendations(report: ReliabilityReport) -> list[str]:
    """Ordered, actionable mitigations for every problem the audit detected."""
    rules = (
        _consistency_rec,
        _validity_rec,
        _rubric_rec,
        _position_rec,
        _verbosity_rec,
        _probe_rec,
        _scale_rec,
        _parse_rec,
        _power_rec,  # always-on noise-floor note comes last
    )
    return [rec for rule in rules if (rec := rule(report)) is not None]
