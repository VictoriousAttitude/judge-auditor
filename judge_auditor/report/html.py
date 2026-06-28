"""Self-contained static HTML report.

One file, embedded CSS, no external JavaScript or CDN dependencies — it opens in any
browser and can be attached to a PR or Slack message. Charts (the score histogram,
the win/tie split) are rendered as plain CSS bars rather than a vendored JS library,
which keeps the artifact tiny and the rendering deterministic for tests.

``render_html`` builds a flat view-model (all numbers pre-formatted into strings) and
hands it to the Jinja template, so the template stays pure layout. Autoescape is on:
the judge model name and any user-supplied text are escaped.
"""

from __future__ import annotations

import math
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..analysis.audit import ReliabilityReport
from ..analysis.stats import CI
from ..config import JudgeMode
from .recommendations import recommendations

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "report.html.j2"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _num(x: float | None, fmt: str = ".3f") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "\u221e"  # infinity sign
    return format(x, fmt)


def _pct(x: float | None) -> str:
    if x is None or math.isnan(x):
        return "n/a"
    return f"{x:.0%}"


def _ci(ci: CI | None) -> str:
    if ci is None or math.isnan(ci.point):
        return "n/a"
    return f"{ci.point:.3f} [{ci.low:.3f}, {ci.high:.3f}]"


def _consistency_rows(r: ReliabilityReport) -> list[tuple[str, str]]:
    c = r.consistency
    if c.mode is JudgeMode.SCALAR:
        return [
            ("ICC(1,1) — headline", f"{_ci(c.icc_oneway)} ({c.icc_interpretation})"),
            ("ICC(2,1)", _ci(c.icc_twoway)),
            ("Mean within-example SD", _num(c.mean_within_sd)),
            ("Runs per example", str(c.runs_per_example)),
        ]
    return [
        ("Fleiss' kappa", f"{_ci(c.fleiss_kappa)} ({c.kappa_interpretation})"),
        ("Mean agreement", _num(c.mean_agreement)),
        ("Min / median agreement", f"{_num(c.min_agreement)} / {_num(c.median_agreement)}"),
        ("Runs per example", str(c.runs_per_example)),
    ]


def _validity_rows(r: ReliabilityReport) -> list[tuple[str, str]] | None:
    v = r.validity
    if not v.available:
        return None
    if v.mode is JudgeMode.SCALAR:
        return [
            ("Score~truth Pearson", f"{_ci(v.pearson_r)} ({v.interpretation})"),
            ("Score~truth Spearman", _num(v.spearman_rho)),
            ("p-value", _num(v.spearman_p)),
            ("Labeled examples", str(v.n_labeled)),
            ("Flagged", "yes" if v.flagged else "no"),
        ]
    return [
        ("Cohen's kappa", f"{_ci(v.cohen_kappa)} ({v.interpretation})"),
        ("Agreement rate", _num(v.agreement_rate)),
        ("Accuracy (excl. ties)", f"{_num(v.accuracy_excl_ties)} (n={v.n_decisive})"),
        ("Labeled examples", str(v.n_labeled)),
        ("Flagged", "yes" if v.flagged else "no"),
    ]


def _rubric_rows(r: ReliabilityReport) -> list[tuple[str, str]] | None:
    rb = r.rubric
    if not rb.available:
        return None
    if rb.mode is JudgeMode.SCALAR:
        return [
            ("Cross-variant ICC", f"{_ci(rb.icc)} ({rb.interpretation})"),
            ("Mean score spread", f"{_num(rb.mean_score_spread)} pts"),
            ("Max score spread", f"{_num(rb.max_score_spread)} pts"),
            ("Variants", str(rb.n_variants)),
            ("Flagged", "yes" if rb.flagged else "no"),
        ]
    return [
        ("Cross-variant kappa", f"{_ci(rb.kappa)} ({rb.interpretation})"),
        ("Winner flip rate", f"{_pct(rb.winner_flip_rate)} ({rb.n_flipped}/{rb.n_examples})"),
        ("Variants", str(rb.n_variants)),
        ("Flagged", "yes" if rb.flagged else "no"),
    ]


def _probe_rows(r: ReliabilityReport) -> list[tuple[str, str]] | None:
    pb = r.probe
    if not pb.available:
        return None
    rows: list[tuple[str, str]] = []
    for e in pb.effects:
        if e.mode is JudgeMode.SCALAR:
            value = f"{_ci(e.effect)} of scale ({_num(e.raw_pts, '+.2f')} pts)"
        else:
            value = f"{_ci(e.effect)} win rate"
        rows.append((f"{e.kind.capitalize()} swing", value))
    rows.append(("Flagged", "yes" if pb.flagged else "no"))
    return rows


def _position_rows(r: ReliabilityReport) -> list[tuple[str, str]] | None:
    p = r.position
    if p is None:
        return None
    return [
        ("First-position rate", _ci(p.first_position_rate)),
        ("Binomial p (vs 0.5)", _num(p.first_preference_p_value)),
        ("Favored position", p.favored_position),
        ("Flip rate", f"{_ci(p.flip_rate)} ({p.n_flipped}/{p.n_examples})"),
        ("Tie rate", _pct(p.tie_rate)),
    ]


def _verbosity_rows(r: ReliabilityReport) -> list[tuple[str, str]]:
    v = r.verbosity
    if v.mode is JudgeMode.SCALAR:
        rows = [
            ("Score~length Spearman", _num(v.spearman_rho)),
            ("p-value", _num(v.spearman_p)),
            ("Flagged", "yes" if v.flagged else "no"),
        ]
        if v.partial_rho is not None:
            rows.append(("Partial (controlling quality)", _num(v.partial_rho)))
        if v.strata is not None:
            worst = max(v.strata, key=lambda s: abs(s.score_gap))
            verdict = "flagged" if v.stratified_flagged else "ok"
            rows.append((
                "Within-quality length effect",
                f"{verdict} (worst q={worst.quality:g}: {worst.score_gap:+.1f} pts, "
                f"rho={_num(worst.spearman_rho)})",
            ))
        return rows
    return [
        ("Longer-response win rate", _num(v.longer_response_win_rate)),
        ("Length~win Spearman", _num(v.length_winrate_rho)),
        ("Flagged", "yes" if v.flagged else "no"),
    ]


def _scale_rows(r: ReliabilityReport) -> list[tuple[str, str]]:
    s = r.scale
    if s.mode is JudgeMode.SCALAR:
        return [
            ("Distinct values used", f"{s.distinct_values_used} / {s.num_bins}"),
            ("Effective dynamic range", _pct(s.effective_range)),
            ("Densest-window share", _pct(s.max_window_fraction)),
            ("Compressed", "yes" if s.compressed else "no"),
        ]
    split = f"{_pct(s.win_rate_a)} / {_pct(s.win_rate_b)} / {_pct(s.tie_rate)}"
    return [
        ("Win A / Win B / Tie", split),
        ("Indistinguishable", "yes" if s.indistinguishable else "no"),
    ]


def _histogram(r: ReliabilityReport) -> list[tuple[str, int, float]] | None:
    """Scalar histogram bars as (label, count, width-percent)."""
    s = r.scale
    if s.mode is not JudgeMode.SCALAR or not s.histogram:
        return None
    peak = max(s.histogram) or 1
    return [
        (f"{int(v)}", c, 100.0 * c / peak)
        for v, c in zip(s.bin_values, s.histogram, strict=False)
    ]


def _pairwise_bars(r: ReliabilityReport) -> list[tuple[str, float]] | None:
    s = r.scale
    if s.mode is not JudgeMode.PAIRWISE:
        return None
    return [
        ("Win A", 100.0 * (s.win_rate_a or 0.0)),
        ("Win B", 100.0 * (s.win_rate_b or 0.0)),
        ("Tie", 100.0 * (s.tie_rate or 0.0)),
    ]


_Rows = list[tuple[str, str]]


def _power_section(r: ReliabilityReport) -> tuple[_Rows, str, _Rows]:
    """Return (rows, noise_floor_line, power_curve_rows)."""
    p = r.power
    if p.mode is JudgeMode.SCALAR:
        rows = [
            ("Judge noise (sigma_w)", _num(p.sigma_w)),
            (f"MDE at n={p.n_examples}", f"{_num(p.mde)} points"),
        ]
        floor = (
            f"Score differences smaller than {_num(p.mde, '.2f')} points are within "
            "judge noise (lower bound — counts judge noise only)."
        )
        curve = [(f"n={n}", f"{_num(mde, '.2f')} pts") for n, mde in p.power_curve]
        return rows, floor, curve
    rows = [
        ("Effective accuracy", _num(p.effective_accuracy)),
        ("Discriminability (2a-1)", _num(p.discriminability)),
        (f"Min detectable margin at n={p.n_examples}", _num(p.mde_winrate)),
    ]
    floor = (
        f"Win-rate margins smaller than {_num(p.mde_winrate)} are within judge noise."
        if p.mde_winrate is not None and math.isfinite(p.mde_winrate)
        else "Zero discriminability: no win-rate margin is reliably detectable."
    )
    curve = [(f"n={n}", _num(m)) for n, m in p.winrate_power_curve]
    return rows, floor, curve


def _headline(r: ReliabilityReport) -> list[tuple[str, str]]:
    c = r.consistency
    items: list[tuple[str, str]] = []
    if c.mode is JudgeMode.SCALAR:
        point = c.icc_oneway.point if c.icc_oneway else None
        items.append(("Self-consistency (ICC)", _num(point, ".2f")))
    else:
        point = c.fleiss_kappa.point if c.fleiss_kappa else None
        items.append(("Self-consistency (kappa)", _num(point, ".2f")))
    if r.position is not None:
        items.append(("Position-flip rate", _pct(r.position.flip_rate.point)))
    p = r.power
    if p.mode is JudgeMode.SCALAR:
        items.append(("Minimum detectable effect", f"{_num(p.mde, '.2f')} points"))
    else:
        items.append(("Min detectable win margin", _num(p.mde_winrate)))
    return items


def build_context(report: ReliabilityReport) -> dict[str, object]:
    """Assemble the flat, pre-formatted view-model passed to the template."""
    rows, floor, curve = _power_section(report)
    return {
        "overall": report.overall,
        "level": report.overall.lower(),
        "model": report.model,
        "mode": report.mode.value,
        "n_examples": report.n_examples,
        "n_records": report.n_records,
        "parse_failure_rate": f"{report.parse_failure_rate:.1%}",
        "headline": _headline(report),
        "consistency_rows": _consistency_rows(report),
        "validity_rows": _validity_rows(report),
        "rubric_rows": _rubric_rows(report),
        "probe_rows": _probe_rows(report),
        "position_rows": _position_rows(report),
        "verbosity_rows": _verbosity_rows(report),
        "scale_rows": _scale_rows(report),
        "histogram": _histogram(report),
        "pairwise_bars": _pairwise_bars(report),
        "power_rows": rows,
        "noise_floor": floor,
        "power_curve": curve,
        "recommendations": recommendations(report),
    }


def render_html(report: ReliabilityReport) -> str:
    """Render the reliability report as a single self-contained HTML document."""
    template = _env().get_template(_TEMPLATE_NAME)
    return template.render(**build_context(report))
