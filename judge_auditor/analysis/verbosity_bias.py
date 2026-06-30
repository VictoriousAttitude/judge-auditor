"""Verbosity bias: does the judge reward length instead of quality?

We correlate the judge's verdict with response length (a whitespace word-count
proxy by default; callers may pass precomputed lengths for true token counts).

* Scalar -> Spearman rank correlation between per-example mean score and response
  length. If the caller supplied ``quality_label`` on every example, we also
  report the **partial** Spearman correlation controlling for quality — this
  separates "longer answers are genuinely better in this dataset" from "the judge
  prefers length regardless of quality."
* Scalar (interaction) -> when ``quality_label`` is *discrete* (a graded rubric,
  e.g. 1-5), we additionally split examples into quality strata and measure the
  score~length effect *within* each stratum, where quality is held ~constant. This
  catches a length effect that lives in a single quality band — e.g. the judge docks
  verbose-but-correct answers while ignoring length on wrong ones — which a single
  global (or even partial) correlation averages away to nearly nothing.
* Pairwise -> the win rate of the *longer* response, plus the Spearman correlation
  between the length difference (len_A - len_B) and the win margin (P(A) - P(B)).

The judge is flagged as potentially length-biased only when the correlation is
**both** practically large (|rho| > threshold, default 0.3) **and** statistically
significant (p < p_threshold, default 0.05) on a minimum sample (n >= min_n,
default 8). This guards against a large-but-noisy correlation on a handful of
examples spuriously tripping the flag. The correlation and its p-value are always
reported regardless of the flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from ..config import EvalExample, JudgeMode, Winner
from ..records import JudgmentSet


@dataclass
class StratumLengthEffect:
    """Score~length effect within one (near-)constant-quality stratum.

    Quality is held fixed inside a stratum, so any score difference between shorter
    and longer responses is a pure style/verbosity effect. ``score_gap`` is the mean
    score of the longer half minus that of the shorter half (negative == the judge
    penalises length here).
    """

    quality: float
    n: int
    spearman_rho: float
    spearman_p: float
    mean_short: float
    mean_long: float
    score_gap: float
    flagged: bool


@dataclass
class VerbosityBiasResult:
    mode: JudgeMode
    n_examples: int
    threshold: float = 0.3
    p_threshold: float = 0.05
    min_n: int = 8
    flagged: bool = False

    # Scalar.
    spearman_rho: float | None = None
    spearman_p: float | None = None
    partial_rho: float | None = None  # score~length controlling for quality_label

    # Scalar interaction: per-quality-stratum length effect (discrete labels only).
    strata: list[StratumLengthEffect] | None = None
    stratified_flagged: bool = False
    max_abs_stratum_rho: float | None = None

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


def _is_flagged(
    rho: float,
    p: float,
    n: int,
    *,
    threshold: float,
    p_threshold: float,
    min_n: int,
) -> bool:
    """Flag only a large (|rho| > threshold) AND significant (p < p_threshold)
    correlation on at least ``min_n`` examples, so a noisy spike on a few points
    no longer trips the flag."""
    return (
        n >= min_n
        and not np.isnan(rho)
        and not np.isnan(p)
        and abs(rho) > threshold
        and p < p_threshold
    )


def _stratify_length_effect(
    scores: list[float],
    lens: list[float],
    quals: list[float],
    *,
    max_strata: int,
    min_stratum_n: int,
    threshold: float,
    p_threshold: float,
) -> list[StratumLengthEffect]:
    """Measure the score~length effect within each discrete quality stratum.

    Only meaningful when ``quality_label`` is discrete (so quality is ~constant
    inside a stratum). For continuous labels — more than ``max_strata`` distinct
    values — we return nothing and leave verbosity to the global/partial correlation,
    because coarse quantile bins would not hold quality fixed.
    """
    distinct = sorted(set(quals))
    if len(distinct) < 2 or len(distinct) > max_strata:
        return []
    out: list[StratumLengthEffect] = []
    for q in distinct:
        idx = [i for i, v in enumerate(quals) if v == q]
        if len(idx) < min_stratum_n:
            continue
        s = [scores[i] for i in idx]
        ln = [lens[i] for i in idx]
        rho, p = _spearman(s, ln)
        if np.isnan(rho):
            continue  # length or score constant within the stratum: no signal
        med = float(np.median(ln))
        short = [s[k] for k in range(len(ln)) if ln[k] < med]
        long = [s[k] for k in range(len(ln)) if ln[k] >= med]
        if not short or not long:  # median sat at an extreme; split the other way
            short = [s[k] for k in range(len(ln)) if ln[k] <= med]
            long = [s[k] for k in range(len(ln)) if ln[k] > med]
        if not short or not long:  # pragma: no cover - non-constant lengths always split
            continue
        mean_short = float(np.mean(short))
        mean_long = float(np.mean(long))
        out.append(
            StratumLengthEffect(
                quality=float(q),
                n=len(idx),
                spearman_rho=rho,
                spearman_p=p,
                mean_short=mean_short,
                mean_long=mean_long,
                score_gap=mean_long - mean_short,
                flagged=abs(rho) > threshold and p < p_threshold,
            )
        )
    return out


def verbosity_bias(
    js: JudgmentSet,
    examples: list[EvalExample],
    *,
    lengths: dict[str, int] | None = None,
    threshold: float = 0.3,
    p_threshold: float = 0.05,
    min_n: int = 8,
    max_strata: int = 6,
    min_stratum_n: int = 8,
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
        strata: list[StratumLengthEffect] | None = None
        stratified_flagged = False
        max_abs_stratum_rho: float | None = None
        if scores and not any(np.isnan(quals)):
            partial = _partial_spearman(scores, lens, quals)
            found = _stratify_length_effect(
                scores,
                lens,
                quals,
                max_strata=max_strata,
                min_stratum_n=min_stratum_n,
                threshold=threshold,
                p_threshold=p_threshold,
            )
            if found:
                strata = found
                stratified_flagged = any(se.flagged for se in found)
                max_abs_stratum_rho = max(abs(se.spearman_rho) for se in found)
        return VerbosityBiasResult(
            mode=js.mode,
            n_examples=len(scores),
            threshold=threshold,
            p_threshold=p_threshold,
            min_n=min_n,
            flagged=_is_flagged(
                rho,
                p,
                len(scores),
                threshold=threshold,
                p_threshold=p_threshold,
                min_n=min_n,
            ),
            spearman_rho=rho,
            spearman_p=p,
            partial_rho=partial,
            strata=strata,
            stratified_flagged=stratified_flagged,
            max_abs_stratum_rho=max_abs_stratum_rho,
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
        p_threshold=p_threshold,
        min_n=min_n,
        flagged=_is_flagged(
            rho,
            p,
            len(len_diffs),
            threshold=threshold,
            p_threshold=p_threshold,
            min_n=min_n,
        ),
        longer_response_win_rate=longer_rate,
        length_winrate_rho=rho,
        length_winrate_p=p,
    )
