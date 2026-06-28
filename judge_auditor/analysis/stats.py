"""Statistical primitives shared by the analysis modules.

Every headline metric in the report carries a confidence interval, because a
single kappa/ICC/flip-rate number is uninterpretable at the sample sizes typical
of an audit (n=50 examples). We use two CI methods:

* :func:`bootstrap_ci` — nonparametric, by resampling *examples* (the exchangeable
  unit) with replacement. This respects the hierarchical structure: a resample
  keeps all K runs of each drawn example together. Used for ICC, kappa, and any
  statistic that is a complicated function of the per-example data.
* :func:`wilson_ci` — the score interval for a binomial proportion (position
  preference, flip rate). Better small-sample coverage than the normal/Wald
  interval, and well-behaved near 0 and 1.

Interpretation thresholds follow the standard references (Landis & Koch 1977 for
kappa; Koo & Li 2016 for ICC) so the report's plain-English labels are defensible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from scipy.stats import norm

T = TypeVar("T")

_EPS = 1e-12


@dataclass(frozen=True)
class CI:
    """A point estimate with a confidence interval."""

    point: float
    low: float
    high: float
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def z_for(confidence: float) -> float:
    """Two-sided z critical value for a confidence level (e.g. 0.95 -> 1.96)."""
    return float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))


def bootstrap_ci(
    units: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap CI for ``statistic`` over a resample of ``units``.

    ``units`` is the resampling unit (typically per-example data). The point
    estimate is ``statistic`` on the full sample; the interval comes from
    resampling units with replacement ``n_boot`` times. Bootstrap replicates
    that come back NaN (degenerate resamples) are dropped before taking
    percentiles.
    """
    n = len(units)
    point = statistic(units)
    if n == 0:
        return CI(point, float("nan"), float("nan"), confidence)

    rng = np.random.default_rng(seed)
    replicates: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [units[int(i)] for i in idx]
        value = statistic(sample)
        if not math.isnan(value):
            replicates.append(value)

    if not replicates:
        return CI(point, float("nan"), float("nan"), confidence)

    arr = np.asarray(replicates, dtype=float)
    alpha = 1.0 - confidence
    low = float(np.percentile(arr, 100.0 * alpha / 2.0))
    high = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return CI(point, low, high, confidence)


def wilson_ci(successes: int, n: int, *, confidence: float = 0.95) -> CI:
    """Wilson score interval for a binomial proportion ``successes / n``."""
    if n == 0:
        return CI(float("nan"), float("nan"), float("nan"), confidence)
    z = z_for(confidence)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom
    return CI(phat, center - half, center + half, confidence)


# --- Interpretation thresholds --------------------------------------------------

# Landis & Koch (1977): agreement strength for Cohen's/Fleiss' kappa.
# (Values < 0 are handled as "poor" before this table is consulted.)
_KAPPA_BANDS = (
    (0.20, "slight"),
    (0.40, "fair"),
    (0.60, "moderate"),
    (0.80, "substantial"),
    (1.01, "almost perfect"),
)

# Koo & Li (2016): reliability bands for ICC.
_ICC_BANDS = (
    (0.50, "poor"),
    (0.75, "moderate"),
    (0.90, "good"),
    (1.01, "excellent"),
)

# Validity (judge-vs-ground-truth) correlation magnitude bands. Cohen (1988) calls
# |r|=0.5 a "large" effect; we set the bar for trusting a judge deliberately higher
# (a judge correlated only 0.5 with the truth still mislabels a lot), so "good" only
# starts at 0.70. Bands apply to |r| (a strong *negative* correlation is still poor
# validity — the judge is anti-correlated with the truth).
_VALIDITY_BANDS = (
    (0.30, "poor"),
    (0.50, "weak"),
    (0.70, "moderate"),
    (1.01, "good"),
)


def interpret_kappa(value: float) -> str:
    """Plain-English agreement strength for a kappa value (Landis & Koch 1977)."""
    if value < 0.0:
        return "poor"
    for upper, label in _KAPPA_BANDS:
        if value < upper:
            return label
    return "almost perfect"


def interpret_icc(value: float) -> str:
    """Plain-English reliability band for an ICC value (Koo & Li 2016)."""
    if value < 0.0:
        return "poor"
    for upper, label in _ICC_BANDS:
        if value < upper:
            return label
    return "excellent"


def interpret_correlation(value: float) -> str:
    """Plain-English validity band for a judge-vs-truth correlation (on |value|)."""
    mag = abs(value)
    for upper, label in _VALIDITY_BANDS:
        if mag < upper:
            return label
    return "good"


def cohen_kappa(labels_a: Sequence[T], labels_b: Sequence[T]) -> float:
    """Cohen's kappa: chance-corrected agreement between two raters' labels.

    ``labels_a`` and ``labels_b`` are paired, equal-length label sequences (here:
    the judge's per-example verdict and the ground-truth verdict). Returns NaN for
    fewer than two pairs. When both raters use a single category every pair agrees:
    that is perfect (if degenerate) agreement, reported as 1.0.
    """
    n = len(labels_a)
    if n != len(labels_b):
        raise ValueError("label sequences must be the same length")
    if n < 2:
        return float("nan")

    categories = set(labels_a) | set(labels_b)
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / n
    expected = 0.0
    for cat in categories:
        p_a = sum(a == cat for a in labels_a) / n
        p_b = sum(b == cat for b in labels_b) / n
        expected += p_a * p_b
    if abs(1.0 - expected) < _EPS:
        return 1.0 if observed > 1.0 - _EPS else float("nan")
    return (observed - expected) / (1.0 - expected)
