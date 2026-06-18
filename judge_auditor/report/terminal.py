"""Plain-text terminal summary of a :class:`ReliabilityReport`.

No dependencies, no colour codes — just an aligned, copy-pasteable digest suitable
for a CI log or a quick local check. The HTML/JSON renderers (later) consume the
same :class:`ReliabilityReport`.
"""

from __future__ import annotations

from ..analysis.audit import ReliabilityReport
from ..analysis.stats import CI
from ..config import JudgeMode

_RULE = "=" * 64


def _ci(ci: CI | None) -> str:
    return str(ci) if ci is not None else "n/a"


def _pct(x: float | None) -> str:
    return f"{x:.0%}" if x is not None else "n/a"


def _num(x: float | None, fmt: str = ".3f") -> str:
    return format(x, fmt) if x is not None else "n/a"


def _header(r: ReliabilityReport) -> list[str]:
    return [
        _RULE,
        f" JUDGE RELIABILITY: {r.overall}",
        _RULE,
        f" Model: {r.model}    Mode: {r.mode.value}",
        f" Examples: {r.n_examples}    Judgments: {r.n_records}"
        f"    Parse failures: {r.parse_failure_rate:.1%}",
    ]


def _consistency_lines(r: ReliabilityReport) -> list[str]:
    c = r.consistency
    if c.mode is JudgeMode.SCALAR:
        return [
            "",
            "SELF-CONSISTENCY (scalar)",
            f"  ICC(1,1):       {_ci(c.icc_oneway)}  [{c.icc_interpretation}]",
            f"  ICC(2,1):       {_ci(c.icc_twoway)}",
            f"  Within-ex SD:   {_num(c.mean_within_sd)}",
        ]
    return [
        "",
        "SELF-CONSISTENCY (pairwise)",
        f"  Fleiss' kappa:  {_ci(c.fleiss_kappa)}  [{c.kappa_interpretation}]",
        f"  Mean agreement: {_num(c.mean_agreement)}"
        f"   (min {_num(c.min_agreement)}, median {_num(c.median_agreement)})",
    ]


def _position_lines(r: ReliabilityReport) -> list[str]:
    p = r.position
    if p is None:
        return []
    return [
        "",
        "POSITION BIAS",
        f"  First-position rate: {_ci(p.first_position_rate)}"
        f"   (p={_num(p.first_preference_p_value)}, favors: {p.favored_position})",
        f"  Flip rate:           {_ci(p.flip_rate)}"
        f"   ({p.n_flipped}/{p.n_examples} examples)",
        f"  Tie rate:            {_pct(p.tie_rate)}",
    ]


def _verbosity_lines(r: ReliabilityReport) -> list[str]:
    v = r.verbosity
    if v.mode is JudgeMode.SCALAR:
        lines = [
            "",
            "VERBOSITY BIAS",
            f"  Score~length Spearman: {_num(v.spearman_rho)}"
            f"   (p={_num(v.spearman_p)})  flagged={v.flagged}",
        ]
        if v.partial_rho is not None:
            lines.append(f"  Partial (vs quality):  {_num(v.partial_rho)}")
        if v.strata is not None:
            lines.append(
                f"  Within-quality strata: stratified_flagged={v.stratified_flagged}"
                f"  (max |rho|={_num(v.max_abs_stratum_rho, '.2f')})"
            )
            for se in v.strata:
                mark = "  <-- flagged" if se.flagged else ""
                lines.append(
                    f"    q={se.quality:g}: gap={se.score_gap:+.2f} pts  "
                    f"rho={_num(se.spearman_rho, '.2f')} "
                    f"(p={_num(se.spearman_p, '.3f')}, n={se.n}){mark}"
                )
        return lines
    return [
        "",
        "VERBOSITY BIAS",
        f"  Longer-response win rate: {_num(v.longer_response_win_rate)}",
        f"  Length~win Spearman:      {_num(v.length_winrate_rho)}"
        f"   flagged={v.flagged}",
    ]


def _scale_lines(r: ReliabilityReport) -> list[str]:
    s = r.scale
    if s.mode is JudgeMode.SCALAR:
        return [
            "",
            "SCALE ANALYSIS",
            f"  Values used:      {s.distinct_values_used}/{s.num_bins}",
            f"  Effective range:  {_pct(s.effective_range)}",
            f"  Densest window:   {_pct(s.max_window_fraction)}"
            f"   compressed={s.compressed}",
        ]
    return [
        "",
        "SCALE ANALYSIS (pairwise)",
        f"  Win A / Win B / Tie: {_pct(s.win_rate_a)} / {_pct(s.win_rate_b)}"
        f" / {_pct(s.tie_rate)}",
        f"  Indistinguishable:   {s.indistinguishable}",
    ]


def _power_lines(r: ReliabilityReport) -> list[str]:
    p = r.power
    if p.mode is JudgeMode.SCALAR:
        lines = [
            "",
            "POWER / NOISE FLOOR (scalar)",
            f"  Judge noise (sigma_w): {_num(p.sigma_w)}",
            f"  MDE at n={p.n_examples}: {_num(p.mde)} points"
            "  (lower bound — judge noise only)",
        ]
        if p.power_curve:
            pts = "  ".join(f"n={n}:{mde:.2f}" for n, mde in p.power_curve)
            lines.append(f"  Power curve: {pts}")
        return lines
    lines = [
        "",
        "POWER / NOISE FLOOR (pairwise)",
        f"  Effective accuracy:   {_num(p.effective_accuracy)}"
        f"  (discriminability {_num(p.discriminability)})",
        f"  Min detectable margin at n={p.n_examples}: {_num(p.mde_winrate)}",
    ]
    if p.winrate_power_curve:
        pts = "  ".join(f"n={n}:{m:.3f}" for n, m in p.winrate_power_curve)
        lines.append(f"  Power curve: {pts}")
    return lines


def _notes_lines(r: ReliabilityReport) -> list[str]:
    if not r.notes:
        return ["", "No bias or scale problems flagged."]
    return ["", "FLAGS"] + [f"  - {n}" for n in r.notes]


def render_terminal(report: ReliabilityReport) -> str:
    """Render a reliability report as an aligned plain-text summary."""
    lines: list[str] = []
    lines += _header(report)
    lines += _consistency_lines(report)
    lines += _position_lines(report)
    lines += _verbosity_lines(report)
    lines += _scale_lines(report)
    lines += _power_lines(report)
    lines += _notes_lines(report)
    lines.append(_RULE)
    return "\n".join(lines)
