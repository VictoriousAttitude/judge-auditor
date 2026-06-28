"""Report diff: what changed between two audited judges?

Given a *baseline* and a *candidate* :class:`ReliabilityReport` audited on the same
eval set (e.g. before/after a rubric edit, or two model versions), this highlights
the deltas that matter: the overall verdict change and the per-metric change with a
CI-aware "changed?" mark.

A numeric metric that carries a confidence interval is marked *changed* only when the
two intervals do **not** overlap — a conservative, distribution-free signal that the
shift is larger than the audit's own noise, mirroring the rest of the tool's
confidently-rule-out philosophy. A boolean flag is marked changed when its status
flips. A metric with no CI (the noise floor, parse-failure rate) shows the raw delta
but leaves the changed mark unknown (``-``), because we cannot tell signal from noise.

Pure formatting on already-computed reports; it makes no judge calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..analysis.audit import ReliabilityReport
from ..analysis.stats import CI
from ..config import JudgeMode


@dataclass(frozen=True)
class MetricDelta:
    name: str
    before: str
    after: str
    delta: str
    changed: bool | None  # True/False from a CI/flag test; None when unknowable


@dataclass(frozen=True)
class ReportDiff:
    baseline_label: str
    candidate_label: str
    mode: JudgeMode
    verdict_before: str
    verdict_after: str
    metrics: list[MetricDelta]

    @property
    def verdict_changed(self) -> bool:
        return self.verdict_before != self.verdict_after


def _disjoint(a: CI | None, b: CI | None) -> bool | None:
    """Whether two CIs fail to overlap (None when either is missing)."""
    if a is None or b is None:
        return None
    return a.high < b.low or b.high < a.low


def _ci_metric(name: str, a: CI | None, b: CI | None, fmt: str = ".3f") -> MetricDelta:
    before = format(a.point, fmt) if a is not None else "n/a"
    after = format(b.point, fmt) if b is not None else "n/a"
    delta = format(b.point - a.point, "+" + fmt) if a is not None and b is not None else "-"
    return MetricDelta(name, before, after, delta, _disjoint(a, b))


def _num_metric(name: str, a: float | None, b: float | None, fmt: str = ".2f") -> MetricDelta:
    before = format(a, fmt) if a is not None and math.isfinite(a) else "n/a"
    after = format(b, fmt) if b is not None and math.isfinite(b) else "n/a"
    if a is not None and b is not None and math.isfinite(a) and math.isfinite(b):
        delta = format(b - a, "+" + fmt)
    else:
        delta = "-"
    return MetricDelta(name, before, after, delta, None)


def _flag_metric(name: str, a: bool, b: bool) -> MetricDelta:
    return MetricDelta(name, "yes" if a else "no", "yes" if b else "no", "-", a != b)


def _flip_rate(r: ReliabilityReport) -> CI | None:
    return r.position.flip_rate if r.position is not None else None


def _verbosity_flagged(r: ReliabilityReport) -> bool:
    return r.verbosity.flagged or r.verbosity.stratified_flagged


def diff_reports(
    baseline: ReliabilityReport,
    candidate: ReliabilityReport,
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> ReportDiff:
    """Diff two reports of the *same* mode into a list of per-metric deltas."""
    if baseline.mode is not candidate.mode:
        raise ValueError(
            f"cannot diff a {baseline.mode.value} report against a {candidate.mode.value} one"
        )
    mode = baseline.mode
    metrics: list[MetricDelta] = []

    if mode is JudgeMode.SCALAR:
        metrics.append(
            _ci_metric(
                "Self-consistency ICC(1,1)",
                baseline.consistency.icc_oneway,
                candidate.consistency.icc_oneway,
            )
        )
        metrics.append(
            _num_metric("Noise floor (MDE pts)", baseline.power.mde, candidate.power.mde)
        )
        metrics.append(
            _flag_metric(
                "Scale compressed", baseline.scale.compressed, candidate.scale.compressed
            )
        )
    else:
        metrics.append(
            _ci_metric(
                "Self-consistency kappa",
                baseline.consistency.fleiss_kappa,
                candidate.consistency.fleiss_kappa,
            )
        )
        metrics.append(
            _ci_metric("Position flip rate", _flip_rate(baseline), _flip_rate(candidate))
        )
        metrics.append(
            _num_metric(
                "Noise floor (win margin)",
                baseline.power.mde_winrate,
                candidate.power.mde_winrate,
                ".3f",
            )
        )
        metrics.append(
            _flag_metric(
                "High tie rate",
                baseline.scale.indistinguishable,
                candidate.scale.indistinguishable,
            )
        )

    metrics.append(
        _flag_metric(
            "Verbosity flagged", _verbosity_flagged(baseline), _verbosity_flagged(candidate)
        )
    )

    if baseline.validity.available or candidate.validity.available:
        if mode is JudgeMode.SCALAR:
            metrics.append(
                _ci_metric(
                    "Validity Pearson r",
                    baseline.validity.pearson_r,
                    candidate.validity.pearson_r,
                )
            )
        else:
            metrics.append(
                _ci_metric(
                    "Validity Cohen kappa",
                    baseline.validity.cohen_kappa,
                    candidate.validity.cohen_kappa,
                )
            )

    if baseline.rubric.available or candidate.rubric.available:
        if mode is JudgeMode.SCALAR:
            metrics.append(
                _ci_metric("Rubric ICC", baseline.rubric.icc, candidate.rubric.icc)
            )
        else:
            metrics.append(
                _ci_metric("Rubric kappa", baseline.rubric.kappa, candidate.rubric.kappa)
            )

    metrics.append(
        _num_metric(
            "Parse failure rate", baseline.parse_failure_rate, candidate.parse_failure_rate, ".3f"
        )
    )

    return ReportDiff(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        mode=mode,
        verdict_before=baseline.overall,
        verdict_after=candidate.overall,
        metrics=metrics,
    )


_RULE = "=" * 72
_CHANGED = {True: "yes", False: "no", None: "-"}


def render_diff_terminal(d: ReportDiff) -> str:
    """Render a report diff as an aligned plain-text table."""
    verdict = f" Verdict: {d.verdict_before} -> {d.verdict_after}"
    if d.verdict_changed:
        verdict += "   (changed)"
    rows = [("METRIC", "BEFORE", "AFTER", "DELTA", "CHANGED")]
    rows += [(m.name, m.before, m.after, m.delta, _CHANGED[m.changed]) for m in d.metrics]
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    body = [
        "  " + "  ".join(row[i].ljust(widths[i]) for i in range(5)).rstrip() for row in rows
    ]
    return "\n".join(
        [
            _RULE,
            f" REPORT DIFF: {d.baseline_label} -> {d.candidate_label}",
            _RULE,
            f" Mode: {d.mode.value}",
            verdict,
            "",
            *body,
            _RULE,
        ]
    )


def render_diff_markdown(d: ReportDiff) -> str:
    """Render a report diff as a Markdown table, ready to drop into a PR."""
    verdict = f"**Verdict:** {d.verdict_before} -> {d.verdict_after}"
    if d.verdict_changed:
        verdict += " (changed)"
    cols = ("Metric", d.baseline_label, d.candidate_label, "Delta", "Changed")
    table = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for m in d.metrics:
        table.append(
            "| " + " | ".join([m.name, m.before, m.after, m.delta, _CHANGED[m.changed]]) + " |"
        )
    return "\n".join(
        [f"### Report diff: {d.baseline_label} -> {d.candidate_label}", "", verdict, "", *table]
    )
