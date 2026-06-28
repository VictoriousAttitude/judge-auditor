"""Validity: does the judge agree with the *ground truth*, not just with itself?

Self-consistency (:mod:`consistency`) measures *reliability* — precision. It cannot
tell a judge that is consistently right from one that is consistently wrong: a judge
with excellent ICC and near-zero validity is precise and useless. This module measures
*validity* — accuracy — by comparing the judge's verdict to expert/human ground truth,
when the caller supplies it (``quality_label`` for scalar, ``preferred_winner`` for
pairwise). The audit downgrades a precise-but-invalid judge on this evidence.

* Scalar -> Pearson (linear) and Spearman (rank) correlation between the judge's mean
  score and the ground-truth ``quality_label``, with a bootstrap CI on Pearson.
  Both are invariant to a quality/score scale mismatch (e.g. quality 1-5 vs score
  1-10): Pearson to any affine rescaling, Spearman to any monotone one.
* Pairwise -> raw agreement rate, Cohen's kappa (chance-corrected) and accuracy
  excluding ties between the judge's per-example majority winner and the
  ``preferred_winner``, with a bootstrap CI on kappa.

Validity is only computed when ground truth is present on at least two examples;
otherwise ``available`` is False and the audit ignores it. The flag (and the verdict
downgrade it drives) fires only when we can *confidently rule out acceptable
validity* — the upper end of the bootstrap CI is still below the "moderate" bar — on
a sufficient sample (``n_labeled >= min_n``). Crucially this is the opposite of the
significance gate the bias detectors use: the worst judges have a correlation
*indistinguishable from zero*, so a "p < 0.05 nonzero" test would wrongly clear them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr

from ..config import EvalExample, JudgeMode, Winner
from ..records import JudgmentSet
from .stats import CI, bootstrap_ci, cohen_kappa, interpret_correlation, interpret_kappa

# Below this |correlation| upper bound we can rule out usable scalar validity.
_CORR_MODERATE = 0.50
# Below this kappa upper bound we can rule out usable pairwise validity.
_KAPPA_MODERATE = 0.40


@dataclass
class ValidityResult:
    mode: JudgeMode
    available: bool = False
    n_labeled: int = 0
    min_n: int = 8
    interpretation: str | None = None
    flagged: bool = False  # confidently poor validity on a sufficient sample

    # Scalar: judge mean score vs quality_label.
    pearson_r: CI | None = None
    spearman_rho: float | None = None
    spearman_p: float | None = None

    # Pairwise: judge majority winner vs preferred_winner.
    agreement_rate: float | None = None
    cohen_kappa: CI | None = None
    accuracy_excl_ties: float | None = None
    n_decisive: int | None = None  # labeled examples where truth AND judge are A/B


def _is_constant(values: Sequence[float]) -> bool:
    return float(np.std(np.asarray(values, dtype=float))) == 0.0


def _safe_corr(
    func: Callable[[Sequence[float], Sequence[float]], tuple[float, float]],
    x: Sequence[float],
    y: Sequence[float],
) -> tuple[float, float]:
    """Correlation + p-value, guarding the undefined (too-few / constant) cases."""
    if len(x) < 3 or _is_constant(x) or _is_constant(y):
        return float("nan"), float("nan")
    res = func(x, y)
    return float(res[0]), float(res[1])


def _majority_winner(winners: Sequence[Winner]) -> Winner:
    """The judge's modal verdict; an A/B split with no unique mode is undecided (tie)."""
    counts = Counter(winners)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return Winner.TIE
    return top[0][0]


def _scalar_validity(
    js: JudgmentSet,
    examples: list[EvalExample],
    *,
    n_boot: int,
    confidence: float,
    seed: int,
    min_n: int,
) -> ValidityResult:
    by_id = {ex.id: ex for ex in examples}
    pairs: list[tuple[float, float]] = []
    for eid in js.example_ids:
        ex = by_id.get(eid)
        if ex is None or ex.quality_label is None:
            continue
        vals = [r.score for r in js.for_example(eid) if r.parse_ok and r.score is not None]
        if not vals:
            continue
        pairs.append((float(np.mean(vals)), float(ex.quality_label)))

    if len(pairs) < 2:
        return ValidityResult(mode=JudgeMode.SCALAR, available=False, n_labeled=len(pairs))

    scores = [p[0] for p in pairs]
    quals = [p[1] for p in pairs]

    def pearson_point(units: Sequence[tuple[float, float]]) -> float:
        r, _ = _safe_corr(pearsonr, [u[0] for u in units], [u[1] for u in units])
        return r

    pearson_ci = bootstrap_ci(pairs, pearson_point, n_boot=n_boot, confidence=confidence, seed=seed)
    rho, p = _safe_corr(spearmanr, scores, quals)

    flagged = (
        len(pairs) >= min_n
        and not np.isnan(pearson_ci.high)
        and pearson_ci.high < _CORR_MODERATE
    )
    interp = None if np.isnan(pearson_ci.point) else interpret_correlation(pearson_ci.point)
    return ValidityResult(
        mode=JudgeMode.SCALAR,
        available=True,
        n_labeled=len(pairs),
        min_n=min_n,
        interpretation=interp,
        flagged=flagged,
        pearson_r=pearson_ci,
        spearman_rho=rho,
        spearman_p=p,
    )


def _pairwise_validity(
    js: JudgmentSet,
    examples: list[EvalExample],
    *,
    n_boot: int,
    confidence: float,
    seed: int,
    min_n: int,
) -> ValidityResult:
    by_id = {ex.id: ex for ex in examples}
    units: list[tuple[Winner, Winner]] = []  # (judge majority, truth)
    for eid in js.example_ids:
        ex = by_id.get(eid)
        if ex is None or ex.preferred_winner is None:
            continue
        winners = [r.winner for r in js.for_example(eid) if r.parse_ok and r.winner is not None]
        if not winners:
            continue
        units.append((_majority_winner(winners), ex.preferred_winner))

    if len(units) < 2:
        return ValidityResult(mode=JudgeMode.PAIRWISE, available=False, n_labeled=len(units))

    agreement = float(np.mean([j == t for j, t in units]))
    decisive = [(j, t) for j, t in units if j is not Winner.TIE and t is not Winner.TIE]
    accuracy = float(np.mean([j == t for j, t in decisive])) if decisive else None

    def kappa_point(rows: Sequence[tuple[Winner, Winner]]) -> float:
        return cohen_kappa([r[0] for r in rows], [r[1] for r in rows])

    kappa_ci = bootstrap_ci(units, kappa_point, n_boot=n_boot, confidence=confidence, seed=seed)

    flagged = (
        len(units) >= min_n
        and not np.isnan(kappa_ci.high)
        and kappa_ci.high < _KAPPA_MODERATE
    )
    interp = None if np.isnan(kappa_ci.point) else interpret_kappa(kappa_ci.point)
    return ValidityResult(
        mode=JudgeMode.PAIRWISE,
        available=True,
        n_labeled=len(units),
        min_n=min_n,
        interpretation=interp,
        flagged=flagged,
        agreement_rate=agreement,
        cohen_kappa=kappa_ci,
        accuracy_excl_ties=accuracy,
        n_decisive=len(decisive),
    )


def validity(
    js: JudgmentSet,
    examples: list[EvalExample],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    min_n: int = 8,
) -> ValidityResult:
    """Compute judge-vs-ground-truth validity (silent when no ground truth is given)."""
    if js.mode is JudgeMode.SCALAR:
        return _scalar_validity(
            js, examples, n_boot=n_boot, confidence=confidence, seed=seed, min_n=min_n
        )
    return _pairwise_validity(
        js, examples, n_boot=n_boot, confidence=confidence, seed=seed, min_n=min_n
    )
