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
    if not v.flagged:
        return None
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
        _position_rec,
        _verbosity_rec,
        _scale_rec,
        _parse_rec,
        _power_rec,  # always-on noise-floor note comes last
    )
    return [rec for rule in rules if (rec := rule(report)) is not None]
