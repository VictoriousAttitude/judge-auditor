"""Judge comparison table: several audited judges side by side.

The hero artifact. Given a list of ``(label, ReliabilityReport)`` pairs — typically
the same eval set judged by different models — this renders a compact Markdown
table so reliability differences between judges are visible at a glance and the
table can be dropped straight into a README or PR.

Pure formatting: it reads only the already-computed reports and emits a string.
Scalar and pairwise judges can appear in the same table; mode-specific cells
(position flip, the noise-floor units) fall back to ``n/a`` where they do not apply.
"""

from __future__ import annotations

import math

from ..analysis.audit import ReliabilityReport
from ..config import JudgeMode

_COLUMNS = (
    "Judge",
    "Mode",
    "Self-consistency",
    "Position flip",
    "Verbosity",
    "Scale",
    "Noise floor",
    "Verdict",
)


def _consistency_cell(r: ReliabilityReport) -> str:
    c = r.consistency
    if c.mode is JudgeMode.SCALAR:
        if c.icc_oneway is None:
            return "n/a"
        return f"ICC {c.icc_oneway.point:.2f} ({c.icc_interpretation})"
    if c.fleiss_kappa is None:
        return "n/a"
    return f"kappa {c.fleiss_kappa.point:.2f} ({c.kappa_interpretation})"


def _flip_cell(r: ReliabilityReport) -> str:
    if r.position is None:
        return "n/a"
    return f"{r.position.flip_rate.point:.0%}"


def _scale_cell(r: ReliabilityReport) -> str:
    s = r.scale
    if s.mode is JudgeMode.SCALAR:
        return "compressed" if s.compressed else "full range"
    return "high tie" if s.indistinguishable else "ok"


def _noise_floor_cell(r: ReliabilityReport) -> str:
    p = r.power
    if p.mode is JudgeMode.SCALAR:
        return "n/a" if p.mde is None else f"{p.mde:.2f} pts"
    if p.mde_winrate is None or not math.isfinite(p.mde_winrate):
        return "no power"
    return f"{p.mde_winrate:.0%} margin"


def comparison_row(label: str, r: ReliabilityReport) -> list[str]:
    """The eight pre-formatted cells for one judge."""
    return [
        label,
        r.mode.value,
        _consistency_cell(r),
        _flip_cell(r),
        "flagged" if r.verbosity.flagged else "ok",
        _scale_cell(r),
        _noise_floor_cell(r),
        r.overall,
    ]


def render_comparison_markdown(reports: list[tuple[str, ReliabilityReport]]) -> str:
    """Render audited judges as a Markdown comparison table."""
    header = "| " + " | ".join(_COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    rows = [
        "| " + " | ".join(comparison_row(label, r)) + " |" for label, r in reports
    ]
    return "\n".join([header, divider, *rows])
