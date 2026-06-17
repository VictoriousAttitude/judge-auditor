"""Scale analysis: is the judge actually using the scoring range it was given?

A judge nominally scoring 1-10 that in practice only ever emits 7, 8, or 9 has an
*effective* scale of three points — every "fine" distinction it reports is noise.
Chen et al. (2024) documented this clustering across judge models.

* Scalar -> we bin the raw per-run scores onto the integer scale, then report:

  - **Effective dynamic range** = Shannon entropy of the score distribution divided
    by ``log(num_bins)``, i.e. the fraction of the scale's information capacity the
    judge actually uses. 1.0 == uniform across all bins; 0.0 == one bin only.
  - **Compression flag** — if more than ``compression_threshold`` (default 70%) of
    scores fall inside the best window of ``max_window`` (default 3) adjacent bins,
    the scale is flagged as compressed.

* Pairwise -> the **tie rate**. If the judge ties more than ``tie_threshold``
  (default 40%) of pairs it cannot resolve most comparisons, which is the pairwise
  analogue of a compressed scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import JudgeMode, Winner
from ..records import JudgmentSet

_EPS = 1e-12


@dataclass
class ScaleAnalysisResult:
    mode: JudgeMode
    n: int  # valid observations (per-run, not per-example)

    # Scalar.
    score_min: float | None = None
    score_max: float | None = None
    num_bins: int | None = None
    bin_values: list[float] = field(default_factory=list)
    histogram: list[int] = field(default_factory=list)
    distinct_values_used: int | None = None
    effective_range: float | None = None  # normalized entropy in [0, 1]
    max_window_fraction: float | None = None  # share in the densest adjacent window
    compressed: bool = False
    compressed_values: list[float] = field(default_factory=list)

    # Pairwise.
    n_decisions: int | None = None  # non-tie decisions
    win_rate_a: float | None = None
    win_rate_b: float | None = None
    tie_rate: float | None = None
    indistinguishable: bool = False


def _normalized_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of the distribution implied by ``counts``, in [0, 1].

    Normalized by ``log(num_bins)`` so a uniform distribution scores 1.0 and a
    single-bin distribution scores 0.0. The log base cancels in the ratio.
    """
    total = float(counts.sum())
    num_bins = counts.shape[0]
    if total <= 0 or num_bins < 2:
        return float("nan")
    p = counts / total
    nz = p[p > _EPS]
    entropy = float(-(nz * np.log(nz)).sum())
    return entropy / float(np.log(num_bins))


def _densest_window(counts: np.ndarray, width: int) -> tuple[float, list[int]]:
    """Max share of mass in any ``width`` adjacent bins; returns (share, indices)."""
    total = float(counts.sum())
    if total <= 0:
        return float("nan"), []
    w = min(width, counts.shape[0])
    best_sum, best_start = -1.0, 0
    for start in range(counts.shape[0] - w + 1):
        s = float(counts[start : start + w].sum())
        if s > best_sum:
            best_sum, best_start = s, start
    return best_sum / total, list(range(best_start, best_start + w))


def scale_analysis(
    js: JudgmentSet,
    *,
    score_min: float = 1.0,
    score_max: float = 10.0,
    compression_threshold: float = 0.70,
    max_window: int = 3,
    tie_threshold: float = 0.40,
) -> ScaleAnalysisResult:
    """Compute the scale-usage result for a judgment set."""
    if js.mode is JudgeMode.SCALAR:
        scores = [r.score for r in js.records if r.parse_ok and r.score is not None]
        lo, hi = int(round(score_min)), int(round(score_max))
        num_bins = hi - lo + 1
        bin_values = [float(lo + i) for i in range(num_bins)]
        counts = np.zeros(num_bins, dtype=float)
        for s in scores:
            idx = int(np.clip(int(round(s)) - lo, 0, num_bins - 1))
            counts[idx] += 1.0

        eff = _normalized_entropy(counts)
        window_frac, window_idx = _densest_window(counts, max_window)
        distinct = int((counts > 0).sum())
        compressed = (
            len(scores) > 0
            and not np.isnan(window_frac)
            and window_frac > compression_threshold
        )
        return ScaleAnalysisResult(
            mode=js.mode,
            n=len(scores),
            score_min=score_min,
            score_max=score_max,
            num_bins=num_bins,
            bin_values=bin_values,
            histogram=[int(c) for c in counts],
            distinct_values_used=distinct,
            effective_range=eff,
            max_window_fraction=window_frac,
            compressed=compressed,
            compressed_values=[bin_values[i] for i in window_idx] if compressed else [],
        )

    # Pairwise.
    winners = [r.winner for r in js.records if r.parse_ok and r.winner is not None]
    total = len(winners)
    a = sum(w is Winner.A for w in winners)
    b = sum(w is Winner.B for w in winners)
    ties = sum(w is Winner.TIE for w in winners)
    tie_rate = (ties / total) if total else float("nan")
    return ScaleAnalysisResult(
        mode=js.mode,
        n=total,
        n_decisions=a + b,
        win_rate_a=(a / total) if total else None,
        win_rate_b=(b / total) if total else None,
        tie_rate=tie_rate,
        indistinguishable=(total > 0 and tie_rate > tie_threshold),
    )
