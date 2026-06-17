"""Synthetic judges with known, controllable reliability.

These generators build :class:`JudgmentSet`\\ s (plus the matching
:class:`EvalExample`\\ s) whose statistical properties are fixed *by construction*,
so the analysis can be validated against a ground truth:

* calibration — generate scores with a known intraclass correlation, or pairwise
  verdicts with a known majority-flip rate, and check the tool recovers them;
* known-bias / null — assemble archetypal judges (position-biased, noisy,
  compressed, perfectly reliable) and check the verdict matches expectation.

They take no API key and are fully seeded, so they double as the data behind the
reproducible judge-comparison demo. This is measurement-grade test apparatus, not
a model of any real judge: it injects exactly the effect under test and nothing
else (e.g. responses are constant-length so verbosity never confounds ICC).
"""

from __future__ import annotations

import json

import numpy as np

from .config import EvalExample, JudgeMode, PairwiseChoice, Winner
from .records import JudgmentRecord, JudgmentSet

# Constant-length filler so verbosity bias never spuriously confounds a generator
# whose only injected effect is in the scores/verdicts.
_RESPONSE = "filler " * 12

Generated = tuple[JudgmentSet, list[EvalExample]]


def _orderings(runs: int) -> list[str]:
    """Balanced AB/BA ordering schedule for ``runs`` pairwise calls."""
    n_ab = (runs + 1) // 2
    return ["AB"] * n_ab + ["BA"] * (runs - n_ab)


def _winner_for(choice: PairwiseChoice, ordering: str) -> Winner:
    """Map a presented-position choice to the canonical (content) winner."""
    if choice is PairwiseChoice.TIE:
        return Winner.TIE
    first_is_a = ordering == "AB"
    picked_first = choice is PairwiseChoice.FIRST
    return Winner.A if (picked_first == first_is_a) else Winner.B


def _choice_for(winner: Winner, ordering: str) -> PairwiseChoice:
    """Inverse of :func:`_winner_for`: which position holds the winning content."""
    if winner is Winner.TIE:
        return PairwiseChoice.TIE
    a_is_first = ordering == "AB"
    winner_is_a = winner is Winner.A
    return PairwiseChoice.FIRST if (winner_is_a == a_is_first) else PairwiseChoice.SECOND


def _scalar_examples(n: int) -> list[EvalExample]:
    return [
        EvalExample(id=f"ex{i}", prompt=f"prompt {i}", response_a=_RESPONSE)
        for i in range(n)
    ]


def _pairwise_examples(n: int) -> list[EvalExample]:
    return [
        EvalExample(id=f"ex{i}", prompt=f"prompt {i}", response_a=_RESPONSE, response_b=_RESPONSE)
        for i in range(n)
    ]


def scalar_judge(
    *,
    icc: float,
    n_examples: int = 120,
    runs: int = 20,
    sigma_w: float = 1.5,
    score_min: float = 1.0,
    score_max: float = 10.0,
    center: float | None = None,
    quantize: bool = False,
    seed: int = 0,
    model: str = "synthetic-scalar",
) -> Generated:
    """A scalar judge whose theoretical ICC(1,1) equals ``icc``.

    Per-example true means are drawn ``N(center, sigma_b^2)`` and each run adds
    independent noise ``N(0, sigma_w^2)``, so the one-way ICC is
    ``sigma_b^2 / (sigma_b^2 + sigma_w^2) == icc`` by construction. Leave
    ``quantize`` off for calibration (continuous scores recover ``icc`` exactly);
    turn it on for realistic integer scores in ``[score_min, score_max]``.
    """
    if not 0.0 <= icc < 1.0:
        raise ValueError("icc must be in [0, 1)")
    mid = center if center is not None else (score_min + score_max) / 2.0
    sigma_b = sigma_w * float(np.sqrt(icc / (1.0 - icc))) if icc > 0.0 else 0.0

    rng = np.random.default_rng(seed)
    examples = _scalar_examples(n_examples)
    records: list[JudgmentRecord] = []
    for ex in examples:
        true_mean = mid + rng.normal(0.0, sigma_b)
        for j in range(runs):
            score = true_mean + rng.normal(0.0, sigma_w)
            if quantize:
                score = float(np.clip(round(score), score_min, score_max))
            records.append(
                JudgmentRecord(
                    example_id=ex.id,
                    run_index=j,
                    rubric_variant=0,
                    ordering=None,
                    raw_response=json.dumps({"score": score}),
                    parse_ok=True,
                    score=score,
                    model=model,
                )
            )
    return JudgmentSet(mode=JudgeMode.SCALAR, model=model, records=records), examples


def compressed_scalar_judge(
    *,
    levels: tuple[int, ...] = (5, 6),
    sigma_w: float = 0.3,
    n_examples: int = 120,
    runs: int = 20,
    score_min: float = 1.0,
    score_max: float = 10.0,
    seed: int = 0,
    model: str = "synthetic-compressed",
) -> Generated:
    """A judge that is self-consistent yet uses only a few adjacent score bins.

    Per-example means are pinned to ``levels`` (e.g. only 5s and 6s) with small
    within-example noise, so consistency is decent but the score scale is
    compressed — exercising the scale-compression downgrade path.
    """
    rng = np.random.default_rng(seed)
    examples = _scalar_examples(n_examples)
    records: list[JudgmentRecord] = []
    for i, ex in enumerate(examples):
        base = float(levels[i % len(levels)])
        for j in range(runs):
            score = float(np.clip(round(base + rng.normal(0.0, sigma_w)), score_min, score_max))
            records.append(
                JudgmentRecord(
                    example_id=ex.id,
                    run_index=j,
                    rubric_variant=0,
                    ordering=None,
                    raw_response=json.dumps({"score": score}),
                    parse_ok=True,
                    score=score,
                    model=model,
                )
            )
    return JudgmentSet(mode=JudgeMode.SCALAR, model=model, records=records), examples


def pairwise_judge_with_flip_rate(
    flip_rate: float,
    *,
    n_examples: int = 200,
    runs: int = 16,
    seed: int = 0,
    model: str = "synthetic-flip",
) -> Generated:
    """Pairwise verdicts with an exact majority-flip rate.

    A ``round(flip_rate * n_examples)`` prefix of examples is made to flip: their
    AB-ordering majority winner is A while their BA-ordering majority winner is B.
    The rest are stable (A under both orderings). The measured flip rate equals
    ``flip_rate`` up to integer rounding.
    """
    if not 0.0 <= flip_rate <= 1.0:
        raise ValueError("flip_rate must be in [0, 1]")
    n_flip = round(flip_rate * n_examples)
    orderings = _orderings(runs)
    examples = _pairwise_examples(n_examples)
    records: list[JudgmentRecord] = []
    for i, ex in enumerate(examples):
        flips = i < n_flip
        for j, ordering in enumerate(orderings):
            winner = Winner.B if (flips and ordering == "BA") else Winner.A
            records.append(
                JudgmentRecord(
                    example_id=ex.id,
                    run_index=j,
                    rubric_variant=0,
                    ordering=ordering,
                    raw_response="[[A]]" if winner is Winner.A else "[[B]]",
                    parse_ok=True,
                    choice=_choice_for(winner, ordering),
                    winner=winner,
                    model=model,
                )
            )
    return JudgmentSet(mode=JudgeMode.PAIRWISE, model=model, records=records), examples


def pairwise_judge_with_first_rate(
    first_rate: float,
    *,
    tie_rate: float = 0.0,
    n_examples: int = 120,
    runs: int = 16,
    seed: int = 0,
    model: str = "synthetic-position",
) -> Generated:
    """A position-biased judge that picks the first-presented response w.p. ``first_rate``.

    Each run independently chooses FIRST (prob ``first_rate``), TIE (prob
    ``tie_rate``), else SECOND. ``first_rate == 1.0`` is the maximally
    position-biased judge (Zheng et al. style): it flips on every example.
    """
    if not 0.0 <= first_rate <= 1.0 or not 0.0 <= tie_rate <= 1.0 or first_rate + tie_rate > 1.0:
        raise ValueError("first_rate and tie_rate must be valid probabilities")
    rng = np.random.default_rng(seed)
    orderings = _orderings(runs)
    examples = _pairwise_examples(n_examples)
    records: list[JudgmentRecord] = []
    for ex in examples:
        for j, ordering in enumerate(orderings):
            u = float(rng.random())
            if u < first_rate:
                choice = PairwiseChoice.FIRST
            elif u < first_rate + tie_rate:
                choice = PairwiseChoice.TIE
            else:
                choice = PairwiseChoice.SECOND
            winner = _winner_for(choice, ordering)
            records.append(
                JudgmentRecord(
                    example_id=ex.id,
                    run_index=j,
                    rubric_variant=0,
                    ordering=ordering,
                    raw_response=choice.value,
                    parse_ok=True,
                    choice=choice,
                    winner=winner,
                    model=model,
                )
            )
    return JudgmentSet(mode=JudgeMode.PAIRWISE, model=model, records=records), examples


def consistent_pairwise_judge(
    *,
    n_examples: int = 120,
    runs: int = 16,
    seed: int = 0,
    model: str = "synthetic-consistent",
) -> Generated:
    """A content-driven pairwise judge: a fixed true winner per example, no flips.

    Each example has a stable winner (A or B) returned on every run regardless of
    ordering, so kappa is ~1, there are no flips, and the first-position rate sits
    near 0.5 (no position preference) — the ideal "null" pairwise judge.
    """
    rng = np.random.default_rng(seed)
    orderings = _orderings(runs)
    examples = _pairwise_examples(n_examples)
    records: list[JudgmentRecord] = []
    for ex in examples:
        true_winner = Winner.A if rng.random() < 0.5 else Winner.B
        for j, ordering in enumerate(orderings):
            records.append(
                JudgmentRecord(
                    example_id=ex.id,
                    run_index=j,
                    rubric_variant=0,
                    ordering=ordering,
                    raw_response="[[A]]" if true_winner is Winner.A else "[[B]]",
                    parse_ok=True,
                    choice=_choice_for(true_winner, ordering),
                    winner=true_winner,
                    model=model,
                )
            )
    return JudgmentSet(mode=JudgeMode.PAIRWISE, model=model, records=records), examples
