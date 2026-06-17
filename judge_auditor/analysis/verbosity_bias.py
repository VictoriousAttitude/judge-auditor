"""Verbosity bias: does the judge reward length instead of quality?

We correlate the judge's verdict with response length (a whitespace word-count
proxy by default; callers may pass precomputed lengths for true token counts).

* Scalar -> Spearman rank correlation between per-example mean score and response
  length. If the caller supplied ``quality_label`` on every example, we also
  report the **partial** Spearman correlation controlling for quality — this
  separates "longer answers are genuinely better in this dataset" from "the judge
  prefers length regardless of quality."
* Pairwise -> the win rate of the *longer* response, plus the Spearman correlation
  between the length difference (len_A - len_B) and the win margin (P(A) - P(B)).

|rho| > threshold (default 0.3) flags the judge as potentially length-biased; the
correlation is always reported regardless of the flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from ..config import EvalExample, JudgeMode, Winner
from ..records import JudgmentSet


@dataclass
class VerbosityBiasResult:
    mode: JudgeMode
    n_examples: int
    threshold: float = 0.3
    flagged: bool = False

    # Scalar.
    spearman_rho: float | None = None
    spearman_p: float | None = None
    partial_rho: float | None = None  # score~length controlling for quality_label

    # Pairwise.
    longer_response_win_rate: float | None = None
    length_winrate_rho: float | None = None
    length_winrate_p: float | None = None


def word_count(text: str) -> int:
    """Default length proxy: number of whitespace-delimited tokens."""
    return len(text.split())


def _is_constant(values: list[float]) -> bool:
    return float(np.std(np.asarray(values, dtype=float))) == 0.0


def _spearman(x: list[float], y: list[float]) -> tuple[float, float]:
    # Correlation is undefined (and scipy warns) when an input is constant.
    if len(x) < 3 or _is_constant(x) or _is_constant(y):
        return float("nan"), float("nan")
    rho, p = spearmanr(x, y)
    return float(rho), float(p)


def _partial_spearman(x: list[float], y: list[float], z: list[float]) -> float:
    """Spearman correlation of x and y controlling for z (rank partial corr)."""
    if len(x) < 4 or _is_constant(x) or _is_constant(y) or _is_constant(z):
        return float("nan")
    rxy, _ = spearmanr(x, y)
    rxz, _ = spearmanr(x, z)
    ryz, _ = spearmanr(y, z)
    denom = np.sqrt((1.0 - rxz**2) * (1.0 - ryz**2))
    if denom < 1e-12:
        return float("nan")
    return float((rxy - rxz * ryz) / denom)


def verbosity_bias(
    js: JudgmentSet,
    examples: list[EvalExample],
    *,
    lengths: dict[str, int] | None = None,
    threshold: float = 0.3,
) -> VerbosityBiasResult:
    """Compute verbosity-bias statistics.

    ``lengths`` (example_id -> length) overrides the word-count proxy for the
    scalar response / the first pairwise response; the proxy is otherwise used.
    """
    by_id = {ex.id: ex for ex in examples}

    def length_of(text: str, example_id: str) -> int:
        if lengths is not None and example_id in lengths:
            return lengths[example_id]
        return word_count(text)

    if js.mode is JudgeMode.SCALAR:
        scores: list[float] = []
        lens: list[float] = []
        quals: list[float] = []
        for eid in js.example_ids:
            ex = by_id.get(eid)
            if ex is None:
                continue
            vals = [r.score for r in js.for_example(eid) if r.parse_ok and r.score is not None]
            if not vals:
                continue
            scores.append(float(np.mean(vals)))
            lens.append(float(length_of(ex.response_a, eid)))
            quals.append(ex.quality_label if ex.quality_label is not None else float("nan"))

        rho, p = _spearman(scores, lens)
        partial = None
        if scores and not any(np.isnan(quals)):
            partial = _partial_spearman(scores, lens, quals)
        return VerbosityBiasResult(
            mode=js.mode,
            n_examples=len(scores),
            threshold=threshold,
            flagged=(not np.isnan(rho)) and abs(rho) > threshold,
            spearman_rho=rho,
            spearman_p=p,
            partial_rho=partial,
        )

    # Pairwise.
    len_diffs: list[float] = []
    win_margins: list[float] = []
    longer_wins: list[float] = []
    for eid in js.example_ids:
        ex = by_id.get(eid)
        if ex is None or ex.response_b is None:
            continue
        winners = [r.winner for r in js.for_example(eid) if r.parse_ok and r.winner is not None]
        a = sum(w is Winner.A for w in winners)
        b = sum(w is Winner.B for w in winners)
        decisive = a + b
        if decisive == 0:
            continue
        p_a = a / decisive
        len_a = length_of(ex.response_a, eid)
        len_b = word_count(ex.response_b)
        len_diffs.append(float(len_a - len_b))
        win_margins.append(p_a - (b / decisive))
        if len_a != len_b:
            longer_wins.append(p_a if len_a > len_b else (b / decisive))

    rho, p = _spearman(len_diffs, win_margins)
    longer_rate = float(np.mean(longer_wins)) if longer_wins else None
    return VerbosityBiasResult(
        mode=js.mode,
        n_examples=len(len_diffs),
        threshold=threshold,
        flagged=(not np.isnan(rho)) and abs(rho) > threshold,
        longer_response_win_rate=longer_rate,
        length_winrate_rho=rho,
        length_winrate_p=p,
    )
