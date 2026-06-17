"""Power analysis: how large a real quality difference must be before the judge
can reliably detect it — the *noise floor*.

This adapts classical statistical power analysis to LLM evaluation. The judge's
own measurement noise sets a lower bound on the effects any downstream A/B test
can resolve; differences below that bound are indistinguishable from judge noise.

* Scalar -> we estimate the pooled within-example SD ``sigma_w`` (the judge's
  measurement noise) from the repeated runs, then report the minimum detectable
  effect (MDE) for a two-group comparison of ``n`` examples each at the requested
  power/alpha::

      MDE(n) = (z_alpha/2 + z_beta) * sigma_w * sqrt(2 / n)

  This is a **lower bound** (a floor): it counts only judge noise and ignores the
  between-example variance a real comparison also carries, so the true MDE is at
  least this large. We state that explicitly rather than pretend otherwise.

* Pairwise -> the judge's self-consistency gives an effective per-pair accuracy
  ``a``; its discriminability ``2a - 1`` attenuates any true win-rate margin. The
  minimum detectable *true* win-rate margin over ``n`` pairs is::

      MDE_margin(n) = (z_alpha/2 + z_beta) * 0.5 / sqrt(n) / (2a - 1)

  A judge that flips its verdict half the time (a -> 0.5) has zero discriminability
  and an infinite noise floor: no number of pairs can rescue it.

Both modes also answer the inverse question — given a target effect, how many
examples/pairs are required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from ..config import JudgeMode, Winner
from ..records import JudgmentSet
from .stats import z_for

_DEFAULT_SIZES = (10, 20, 30, 50, 100, 200, 500)
_CATEGORIES = (Winner.A, Winner.B, Winner.TIE)


@dataclass
class PowerAnalysisResult:
    mode: JudgeMode
    n_examples: int  # examples with >= 2 runs (the estimation basis)
    alpha: float = 0.05
    power: float = 0.80
    target_effect: float | None = None

    # Scalar.
    sigma_w: float | None = None  # pooled within-example SD (judge noise)
    mde: float | None = None  # noise floor at the observed sample size
    power_curve: list[tuple[int, float]] = field(default_factory=list)  # (n, MDE)
    required_n: int | None = None  # examples per group to detect target_effect

    # Pairwise.
    effective_accuracy: float | None = None  # a: prob of the modal verdict
    discriminability: float | None = None  # 2a - 1
    mde_winrate: float | None = None  # min detectable TRUE win-rate margin at n
    winrate_power_curve: list[tuple[int, float]] = field(default_factory=list)
    required_pairs: int | None = None  # pairs to detect a target_effect margin


def _z_factor(alpha: float, power: float) -> float:
    """(z_{alpha/2} + z_{1-beta}) — the two-sided power constant."""
    return z_for(1.0 - alpha) + float(norm.ppf(power))


def _scalar_within_sd(js: JudgmentSet) -> tuple[float, int]:
    """Pooled within-example SD and the count of examples with >= 2 runs."""
    by_example: dict[str, list[float]] = {}
    for r in js.records:
        if r.parse_ok and r.score is not None:
            by_example.setdefault(r.example_id, []).append(r.score)
    rows = [np.asarray(v, dtype=float) for v in by_example.values() if len(v) >= 2]
    if not rows:
        return float("nan"), 0
    num = sum((len(v) - 1) * float(np.var(v, ddof=1)) for v in rows)
    den = sum(len(v) - 1 for v in rows)
    sigma_w = math.sqrt(num / den) if den > 0 else float("nan")
    return sigma_w, len(rows)


def _effective_accuracy(js: JudgmentSet) -> tuple[float, int]:
    """Mean modal-agreement rate over examples with >= 2 winner ratings."""
    by_example: dict[str, list[Winner]] = {}
    for r in js.records:
        if r.parse_ok and r.winner is not None:
            by_example.setdefault(r.example_id, []).append(r.winner)
    rates: list[float] = []
    for winners in by_example.values():
        if len(winners) < 2:
            continue
        top = max(sum(w is c for w in winners) for c in _CATEGORIES)
        rates.append(top / len(winners))
    if not rates:
        return float("nan"), 0
    return float(np.mean(rates)), len(rates)


def power_analysis(
    js: JudgmentSet,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    sample_sizes: tuple[int, ...] = _DEFAULT_SIZES,
    target_effect: float | None = None,
) -> PowerAnalysisResult:
    """Compute the noise-floor / power result for a judgment set."""
    factor = _z_factor(alpha, power)

    if js.mode is JudgeMode.SCALAR:
        sigma_w, n_ex = _scalar_within_sd(js)
        if n_ex == 0 or math.isnan(sigma_w):
            return PowerAnalysisResult(
                mode=js.mode, n_examples=n_ex, alpha=alpha, power=power,
                target_effect=target_effect, sigma_w=sigma_w,
            )

        def mde_at(n: int) -> float:
            return factor * sigma_w * math.sqrt(2.0 / n)

        curve = [(int(n), mde_at(int(n))) for n in sample_sizes if n >= 1]
        required = None
        if target_effect is not None and target_effect > 0:
            required = math.ceil(2.0 * (factor * sigma_w / target_effect) ** 2)
        return PowerAnalysisResult(
            mode=js.mode,
            n_examples=n_ex,
            alpha=alpha,
            power=power,
            target_effect=target_effect,
            sigma_w=sigma_w,
            mde=mde_at(n_ex),
            power_curve=curve,
            required_n=required,
        )

    # Pairwise.
    acc, n_ex = _effective_accuracy(js)
    if n_ex == 0 or math.isnan(acc):
        return PowerAnalysisResult(
            mode=js.mode, n_examples=n_ex, alpha=alpha, power=power,
            target_effect=target_effect, effective_accuracy=acc,
        )
    discrim = 2.0 * acc - 1.0

    def margin_at(n: int) -> float:
        if discrim <= 0:
            return float("inf")
        return factor * 0.5 / math.sqrt(n) / discrim

    curve = [(int(n), margin_at(int(n))) for n in sample_sizes if n >= 1]
    required = None
    if target_effect is not None and target_effect > 0 and discrim > 0:
        required = math.ceil((factor * 0.5 / (target_effect * discrim)) ** 2)
    return PowerAnalysisResult(
        mode=js.mode,
        n_examples=n_ex,
        alpha=alpha,
        power=power,
        target_effect=target_effect,
        effective_accuracy=acc,
        discriminability=discrim,
        mde_winrate=margin_at(n_ex),
        winrate_power_curve=curve,
        required_pairs=required,
    )
