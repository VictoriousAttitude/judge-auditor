"""Rubric robustness: does the verdict survive paraphrasing the rubric?

A judge's verdict should reflect the response, not the incidental wording of the
rubric. If you keep the rubric's *intent* but rephrase it and the verdicts move, the
judge is **brittle**: part of its signal is an artifact of prompt phrasing rather than
response quality. This module measures that by comparing the judge's per-example
aggregate verdict across the rubric variants the runner collected (the
``rubric_variant`` stamped on each record).

It activates only when the audit ran at least two rubric variants; with a single
rubric there is nothing to compare and ``available`` is False (no verdict impact),
exactly like validity without ground truth.

* Scalar -> cross-variant **ICC(2,1)** on the per-example mean score under each
  variant. Variants are a genuine crossed factor here (unlike the exchangeable repeat
  runs), so the two-way absolute-agreement form is the correct one. Also reports the
  mean per-example score spread (max-min of the per-variant means).
* Pairwise -> cross-variant **Fleiss' kappa** on the per-example majority winner under
  each variant, plus the fraction of examples whose majority winner is not unanimous
  across variants.

Comparing per-example *aggregates* (mean score / majority winner per variant) first
averages out within-variant run noise, so what remains is the systematic effect of
phrasing. The flag (and the downgrade it drives) fires only when the bootstrap CI
rules out good / substantial cross-variant agreement on a sufficient sample
(``n_examples >= min_n``) — the same confidently-rule-out logic as the validity gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..config import JudgeMode
from ..records import JudgmentSet
from .consistency import _CATEGORIES, fleiss_kappa, icc_twoway
from .stats import CI, bootstrap_ci, interpret_icc, interpret_kappa
from .validity import _majority_winner

# Cross-variant agreement at or above these bars is "robust"; the flag fires when the
# CI upper bound stays below them (we can rule robustness out). The LOW thresholds set
# how far to downgrade: below them the verdict is barely phrasing-independent at all.
_ROBUST_ICC = 0.75
_ROBUST_ICC_LOW = 0.50
_ROBUST_KAPPA = 0.60
_ROBUST_KAPPA_LOW = 0.40


@dataclass
class RubricRobustnessResult:
    mode: JudgeMode
    available: bool = False
    n_variants: int = 0
    n_examples: int = 0
    min_n: int = 8
    interpretation: str | None = None
    flagged: bool = False  # confidently brittle on a sufficient sample

    # Scalar: cross-variant agreement of per-example mean scores.
    icc: CI | None = None
    mean_score_spread: float | None = None
    max_score_spread: float | None = None

    # Pairwise: cross-variant agreement of per-example majority winners.
    kappa: CI | None = None
    winner_flip_rate: float | None = None
    n_flipped: int | None = None

    @property
    def severe(self) -> bool:
        """True when cross-variant agreement is so low the downgrade should be LOW."""
        if self.mode is JudgeMode.SCALAR:
            return self.icc is not None and self.icc.point < _ROBUST_ICC_LOW
        return self.kappa is not None and self.kappa.point < _ROBUST_KAPPA_LOW


def _variants(js: JudgmentSet) -> list[int]:
    return sorted({r.rubric_variant for r in js.records})


def _scalar_matrix(js: JudgmentSet, variants: list[int]) -> list[npt.NDArray[np.float64]]:
    """Per-example rows of mean score under each variant (examples complete over all)."""
    rows: list[npt.NDArray[np.float64]] = []
    for eid in js.example_ids:
        means: list[float] = []
        complete = True
        for v in variants:
            vals = [r.score for r in js.for_example(eid, v) if r.parse_ok and r.score is not None]
            if not vals:
                complete = False
                break
            means.append(float(np.mean(vals)))
        if complete:
            rows.append(np.asarray(means, dtype=float))
    return rows


def _scalar_robustness(
    js: JudgmentSet,
    variants: list[int],
    *,
    n_boot: int,
    confidence: float,
    seed: int,
    min_n: int,
) -> RubricRobustnessResult:
    rows = _scalar_matrix(js, variants)
    if len(rows) < 2:
        return RubricRobustnessResult(
            mode=JudgeMode.SCALAR,
            available=False,
            n_variants=len(variants),
            n_examples=len(rows),
        )

    def icc(units: Sequence[npt.NDArray[np.float64]]) -> float:
        return icc_twoway(np.vstack(units))

    icc_ci = bootstrap_ci(rows, icc, n_boot=n_boot, confidence=confidence, seed=seed)
    spreads = [float(r.max() - r.min()) for r in rows]
    flagged = (
        len(rows) >= min_n and not np.isnan(icc_ci.high) and icc_ci.high < _ROBUST_ICC
    )
    interp = None if np.isnan(icc_ci.point) else interpret_icc(icc_ci.point)
    return RubricRobustnessResult(
        mode=JudgeMode.SCALAR,
        available=True,
        n_variants=len(variants),
        n_examples=len(rows),
        min_n=min_n,
        interpretation=interp,
        flagged=flagged,
        icc=icc_ci,
        mean_score_spread=float(np.mean(spreads)),
        max_score_spread=float(np.max(spreads)),
    )


def _pairwise_robustness(
    js: JudgmentSet,
    variants: list[int],
    *,
    n_boot: int,
    confidence: float,
    seed: int,
    min_n: int,
) -> RubricRobustnessResult:
    rows: list[npt.NDArray[np.float64]] = []  # per-example category counts across variants
    n_flipped = 0
    for eid in js.example_ids:
        majorities = []
        complete = True
        for v in variants:
            winners = [
                r.winner for r in js.for_example(eid, v) if r.parse_ok and r.winner is not None
            ]
            if not winners:
                complete = False
                break
            majorities.append(_majority_winner(winners))
        if not complete:
            continue
        rows.append(np.asarray([float(majorities.count(c)) for c in _CATEGORIES], dtype=float))
        if len(set(majorities)) > 1:
            n_flipped += 1

    if len(rows) < 2:
        return RubricRobustnessResult(
            mode=JudgeMode.PAIRWISE,
            available=False,
            n_variants=len(variants),
            n_examples=len(rows),
        )

    def kappa(units: Sequence[npt.NDArray[np.float64]]) -> float:
        return fleiss_kappa(np.vstack(units))

    kappa_ci = bootstrap_ci(rows, kappa, n_boot=n_boot, confidence=confidence, seed=seed)
    flag = (
        len(rows) >= min_n and not np.isnan(kappa_ci.high) and kappa_ci.high < _ROBUST_KAPPA
    )
    interp = None if np.isnan(kappa_ci.point) else interpret_kappa(kappa_ci.point)
    return RubricRobustnessResult(
        mode=JudgeMode.PAIRWISE,
        available=True,
        n_variants=len(variants),
        n_examples=len(rows),
        min_n=min_n,
        interpretation=interp,
        flagged=flag,
        kappa=kappa_ci,
        winner_flip_rate=n_flipped / len(rows),
        n_flipped=n_flipped,
    )


def rubric_robustness(
    js: JudgmentSet,
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    min_n: int = 8,
) -> RubricRobustnessResult:
    """Cross-rubric stability (silent unless the audit ran >= 2 rubric variants)."""
    variants = _variants(js)
    if len(variants) < 2:
        mode = js.mode
        return RubricRobustnessResult(mode=mode, available=False, n_variants=len(variants))
    if js.mode is JudgeMode.SCALAR:
        return _scalar_robustness(
            js, variants, n_boot=n_boot, confidence=confidence, seed=seed, min_n=min_n
        )
    return _pairwise_robustness(
        js, variants, n_boot=n_boot, confidence=confidence, seed=seed, min_n=min_n
    )
