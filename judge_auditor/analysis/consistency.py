"""Self-consistency: does the judge agree with itself across repeated runs?

This is the headline measurement and the source of the noise floor.

* Scalar mode -> **ICC** (intraclass correlation) computed from ANOVA mean
  squares. We report two forms:

  - ICC(1,1), one-way random effects — the *correct* model here, because the K
    repeated runs of a single judge are exchangeable (run #3 on example A shares
    no identity with run #3 on example B), so there is no crossed "rater" factor.
  - ICC(2,1), two-way random effects, absolute agreement — reported alongside for
    readers who expect the Shrout & Fleiss form; it converges with ICC(1,1) when
    there is no systematic run-position effect (which there isn't, by exchange-
    ability), so a large gap between the two is itself a diagnostic.

  Both use absolute agreement (not Pearson correlation): a judge that always
  scores 2 points high would have r=1 but ICC<1, and we want to catch that.

* Pairwise mode -> **Fleiss' kappa** on the canonical winners (A/B/tie), which
  chance-corrects the raw agreement. Kappa needs only the per-example category
  counts, not rater identity — exactly right for exchangeable runs.

All headline statistics carry bootstrap CIs (resampling examples).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..config import JudgeMode, Winner
from ..records import JudgmentSet
from .stats import CI, bootstrap_ci, interpret_icc, interpret_kappa

_EPS = 1e-12
_CATEGORIES = (Winner.A, Winner.B, Winner.TIE)


@dataclass
class ConsistencyResult:
    mode: JudgeMode
    n_examples: int
    runs_per_example: int  # the (post-truncation) balanced rating count

    # Scalar.
    icc_oneway: CI | None = None  # ICC(1,1), the headline
    icc_twoway: CI | None = None  # ICC(2,1), reported alongside
    icc_interpretation: str | None = None
    mean_within_sd: float | None = None

    # Pairwise.
    fleiss_kappa: CI | None = None
    kappa_interpretation: str | None = None
    mean_agreement: float | None = None
    min_agreement: float | None = None
    median_agreement: float | None = None


# --- ICC (scalar) ---------------------------------------------------------------


def _mean_squares(
    matrix: npt.NDArray[np.float64],
) -> tuple[float, float, float, float, int, int]:
    """Return (MSR, MSW, MSE, MSC, n, k) for an n x k targets-by-runs matrix."""
    n, k = matrix.shape
    grand = float(matrix.mean())
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)

    ss_total = float(((matrix - grand) ** 2).sum())
    ss_rows = float(k * ((row_means - grand) ** 2).sum())
    ss_cols = float(n * ((col_means - grand) ** 2).sum())
    ss_error = ss_total - ss_rows - ss_cols  # two-way residual
    ss_within = ss_total - ss_rows  # one-way within-target

    msr = ss_rows / (n - 1)
    msw = ss_within / (n * (k - 1))
    mse = ss_error / ((n - 1) * (k - 1))
    msc = ss_cols / (k - 1)
    return msr, msw, mse, msc, n, k


def icc_oneway(matrix: npt.NDArray[np.float64]) -> float:
    """ICC(1,1): one-way random effects, single rater, absolute agreement."""
    msr, msw, _mse, _msc, _n, k = _mean_squares(matrix)
    denom = msr + (k - 1) * msw
    if abs(denom) < _EPS:
        return float("nan")
    return (msr - msw) / denom


def icc_twoway(matrix: npt.NDArray[np.float64]) -> float:
    """ICC(2,1): two-way random effects, single rater, absolute agreement."""
    msr, _msw, mse, msc, n, k = _mean_squares(matrix)
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if abs(denom) < _EPS:
        return float("nan")
    return (msr - mse) / denom


def _scalar_rows(js: JudgmentSet) -> list[npt.NDArray[np.float64]]:
    """Balanced per-example score rows (truncated to the common run count)."""
    by_example: dict[str, list[float]] = {}
    for r in js.records:
        if r.parse_ok and r.score is not None:
            by_example.setdefault(r.example_id, []).append(r.score)
    kept = [v for v in by_example.values() if len(v) >= 2]
    if len(kept) < 2:
        return []
    m = min(len(v) for v in kept)
    if m < 2:
        return []
    return [np.asarray(v[:m], dtype=float) for v in kept]


# --- Fleiss' kappa (pairwise) ---------------------------------------------------


def fleiss_kappa(counts: npt.NDArray[np.float64]) -> float:
    """Fleiss' kappa from an n x c matrix of per-subject category counts.

    Each row must sum to the same number of ratings m (>= 2).
    """
    n, _c = counts.shape
    m = float(counts[0].sum())
    if m < 2:
        return float("nan")

    p_i = (np.square(counts).sum(axis=1) - m) / (m * (m - 1.0))
    p_bar = float(p_i.mean())
    p_j = counts.sum(axis=0) / (n * m)
    p_e = float(np.square(p_j).sum())

    if abs(1.0 - p_e) < _EPS:
        # All ratings fell in one category => perfect (if degenerate) agreement.
        return 1.0 if p_bar > 1.0 - _EPS else float("nan")
    return (p_bar - p_e) / (1.0 - p_e)


def _winner_counts(js: JudgmentSet) -> npt.NDArray[np.float64]:
    """Balanced per-example category-count matrix over (A, B, tie)."""
    by_example: dict[str, list[Winner]] = {}
    for r in js.records:
        if r.parse_ok and r.winner is not None:
            by_example.setdefault(r.example_id, []).append(r.winner)
    kept = [v for v in by_example.values() if len(v) >= 2]
    if len(kept) < 2:
        return np.empty((0, len(_CATEGORIES)), dtype=float)
    m = min(len(v) for v in kept)
    rows = []
    for winners in kept:
        truncated = winners[:m]
        rows.append([float(truncated.count(cat)) for cat in _CATEGORIES])
    return np.asarray(rows, dtype=float)


def _agreement_rates(counts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Per-example modal agreement rate = (max category count) / ratings."""
    m = float(counts[0].sum())
    return np.asarray(counts.max(axis=1) / m, dtype=float)


# --- Public entry point ---------------------------------------------------------


def consistency(
    js: JudgmentSet,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ConsistencyResult:
    """Compute the self-consistency result for a judgment set."""
    if js.mode is JudgeMode.SCALAR:
        rows = _scalar_rows(js)
        if not rows:
            return ConsistencyResult(mode=js.mode, n_examples=len(rows), runs_per_example=0)
        k = len(rows[0])

        def icc1(units: Sequence[npt.NDArray[np.float64]]) -> float:
            return icc_oneway(np.vstack(units))

        def icc2(units: Sequence[npt.NDArray[np.float64]]) -> float:
            return icc_twoway(np.vstack(units))

        ci1 = bootstrap_ci(rows, icc1, n_boot=n_boot, confidence=confidence, seed=seed)
        ci2 = bootstrap_ci(rows, icc2, n_boot=n_boot, confidence=confidence, seed=seed)
        mean_within_sd = float(np.mean([float(np.std(r, ddof=1)) for r in rows]))
        return ConsistencyResult(
            mode=js.mode,
            n_examples=len(rows),
            runs_per_example=k,
            icc_oneway=ci1,
            icc_twoway=ci2,
            icc_interpretation=interpret_icc(ci1.point),
            mean_within_sd=mean_within_sd,
        )

    # Pairwise.
    counts = _winner_counts(js)
    if counts.shape[0] < 2:
        return ConsistencyResult(mode=js.mode, n_examples=counts.shape[0], runs_per_example=0)
    m = int(counts[0].sum())

    def kappa(units: Sequence[npt.NDArray[np.float64]]) -> float:
        return fleiss_kappa(np.vstack(units))

    rows_list = [counts[i] for i in range(counts.shape[0])]
    ci = bootstrap_ci(rows_list, kappa, n_boot=n_boot, confidence=confidence, seed=seed)
    rates = _agreement_rates(counts)
    return ConsistencyResult(
        mode=js.mode,
        n_examples=counts.shape[0],
        runs_per_example=m,
        fleiss_kappa=ci,
        kappa_interpretation=interpret_kappa(ci.point),
        mean_agreement=float(rates.mean()),
        min_agreement=float(rates.min()),
        median_agreement=float(np.median(rates)),
    )
