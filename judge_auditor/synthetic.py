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

from .config import EvalExample, JudgeMode, PairwiseChoice, Probe, Winner
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


def scalar_judge_with_validity(
    *,
    rho: float,
    n_examples: int = 150,
    runs: int = 20,
    sigma_w: float = 0.5,
    score_min: float = 1.0,
    score_max: float = 10.0,
    quantize: bool = False,
    seed: int = 0,
    model: str = "synthetic-validity",
) -> Generated:
    """A self-consistent scalar judge whose score~quality correlation equals ``rho``.

    Each example gets a ground-truth ``quality_label`` ``q ~ N(0, 1)``; the judge's
    per-example true mean is ``rho*q + sqrt(1 - rho^2)*e`` (``e`` independent), so the
    judge mean score correlates with the truth at ``rho`` by construction. Small
    within-example noise (``sigma_w``) keeps the judge *reliable* (high ICC) regardless
    of ``rho`` — so ``rho=0`` is the archetypal "precise but useless" judge: perfectly
    self-consistent yet uncorrelated with the truth.
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    rng = np.random.default_rng(seed)
    mid = (score_min + score_max) / 2.0
    scale = (score_max - score_min) / 6.0  # map +/-3 SD of N(0,1) across the range
    resid = float(np.sqrt(1.0 - rho**2))
    examples: list[EvalExample] = []
    records: list[JudgmentRecord] = []
    for i in range(n_examples):
        q = float(rng.normal())
        true_mean = rho * q + resid * float(rng.normal())
        examples.append(
            EvalExample(id=f"ex{i}", prompt=f"prompt {i}", response_a=_RESPONSE, quality_label=q)
        )
        for j in range(runs):
            score = mid + scale * true_mean + float(rng.normal(0.0, sigma_w))
            if quantize:
                score = float(np.clip(round(score), score_min, score_max))
            records.append(
                JudgmentRecord(
                    example_id=f"ex{i}",
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


def pairwise_judge_with_accuracy(
    accuracy: float,
    *,
    n_examples: int = 150,
    runs: int = 16,
    seed: int = 0,
    model: str = "synthetic-accuracy",
) -> Generated:
    """A self-consistent pairwise judge that agrees with the truth w.p. ``accuracy``.

    Each example has a ground-truth ``preferred_winner`` (A or B, balanced). The judge
    commits to one winner — the true one with probability ``accuracy``, otherwise the
    wrong one — and returns it on *every* run regardless of ordering. So the judge is
    perfectly self-consistent (Fleiss' kappa ~ 1, no position flips) while its
    *validity* (agreement with the truth) is exactly ``accuracy``. ``accuracy=0.5`` is
    the precise-but-useless judge: it agrees with itself completely and with the truth
    no better than a coin.
    """
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be in [0, 1]")
    rng = np.random.default_rng(seed)
    orderings = _orderings(runs)
    examples: list[EvalExample] = []
    records: list[JudgmentRecord] = []
    for i in range(n_examples):
        truth = Winner.A if rng.random() < 0.5 else Winner.B
        correct = rng.random() < accuracy
        judged = truth if correct else (Winner.B if truth is Winner.A else Winner.A)
        examples.append(
            EvalExample(
                id=f"ex{i}",
                prompt=f"prompt {i}",
                response_a=_RESPONSE,
                response_b=_RESPONSE,
                preferred_winner=truth,
            )
        )
        for j, ordering in enumerate(orderings):
            records.append(
                JudgmentRecord(
                    example_id=f"ex{i}",
                    run_index=j,
                    rubric_variant=0,
                    ordering=ordering,
                    raw_response="[[A]]" if judged is Winner.A else "[[B]]",
                    parse_ok=True,
                    choice=_choice_for(judged, ordering),
                    winner=judged,
                    model=model,
                )
            )
    return JudgmentSet(mode=JudgeMode.PAIRWISE, model=model, records=records), examples


def scalar_judge_with_rubric_sensitivity(
    *,
    sensitivity: float,
    n_variants: int = 3,
    n_examples: int = 120,
    runs: int = 20,
    sigma_w: float = 0.3,
    score_min: float = 1.0,
    score_max: float = 10.0,
    quantize: bool = False,
    seed: int = 0,
    model: str = "synthetic-rubric-scalar",
) -> Generated:
    """A scalar judge whose dependence on rubric *phrasing* is set by ``sensitivity``.

    Each example has a base true quality; under variant ``v`` its true mean is
    ``(1 - s)*base + s*offset_v`` (``offset_v`` independent per variant). Within each
    variant the judge is self-consistent (small ``sigma_w``), so ``s=0`` is *robust*
    (all variants share the per-example mean, cross-variant ICC ~ 1) and ``s=1`` is
    *brittle* (each variant scores independently, cross-variant ICC ~ 0) while the
    headline variant-0 self-consistency stays high either way.
    """
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError("sensitivity must be in [0, 1]")
    if n_variants < 2:
        raise ValueError("need at least 2 rubric variants")
    rng = np.random.default_rng(seed)
    mid = (score_min + score_max) / 2.0
    scale = (score_max - score_min) / 6.0
    s = sensitivity
    examples = _scalar_examples(n_examples)
    records: list[JudgmentRecord] = []
    for ex in examples:
        base = float(rng.normal())
        for v in range(n_variants):
            variant_mean = (1.0 - s) * base + s * float(rng.normal())
            for j in range(runs):
                score = mid + scale * variant_mean + float(rng.normal(0.0, sigma_w))
                if quantize:
                    score = float(np.clip(round(score), score_min, score_max))
                records.append(
                    JudgmentRecord(
                        example_id=ex.id,
                        run_index=j,
                        rubric_variant=v,
                        ordering=None,
                        raw_response=json.dumps({"score": score}),
                        parse_ok=True,
                        score=score,
                        model=model,
                    )
                )
    return JudgmentSet(mode=JudgeMode.SCALAR, model=model, records=records), examples


def pairwise_judge_with_rubric_sensitivity(
    *,
    flip_fraction: float,
    n_variants: int = 3,
    n_examples: int = 120,
    runs: int = 16,
    seed: int = 0,
    model: str = "synthetic-rubric-pairwise",
) -> Generated:
    """A pairwise judge whose winner depends on rubric phrasing for ``flip_fraction`` of examples.

    Each example has a base true winner returned stably across runs (so variant-0
    self-consistency is ~ perfect and there are no position flips). For a
    ``flip_fraction`` prefix of examples the non-canonical variants return the opposite
    winner, so the cross-variant winner disagrees; the rest agree across all variants.
    The measured cross-variant winner-flip rate equals ``flip_fraction`` (up to integer
    rounding): ``flip_fraction=0`` is robust (cross-variant kappa ~ 1) and ``=1`` is
    fully brittle.
    """
    if not 0.0 <= flip_fraction <= 1.0:
        raise ValueError("flip_fraction must be in [0, 1]")
    if n_variants < 2:
        raise ValueError("need at least 2 rubric variants")
    n_flip = round(flip_fraction * n_examples)
    rng = np.random.default_rng(seed)
    orderings = _orderings(runs)
    examples = _pairwise_examples(n_examples)
    records: list[JudgmentRecord] = []
    for i, ex in enumerate(examples):
        base = Winner.A if rng.random() < 0.5 else Winner.B
        opposite = Winner.B if base is Winner.A else Winner.A
        brittle = i < n_flip
        for v in range(n_variants):
            winner = opposite if (brittle and v > 0) else base
            for j, ordering in enumerate(orderings):
                records.append(
                    JudgmentRecord(
                        example_id=ex.id,
                        run_index=j,
                        rubric_variant=v,
                        ordering=ordering,
                        raw_response="[[A]]" if winner is Winner.A else "[[B]]",
                        parse_ok=True,
                        choice=_choice_for(winner, ordering),
                        winner=winner,
                        model=model,
                    )
                )
    return JudgmentSet(mode=JudgeMode.PAIRWISE, model=model, records=records), examples


def _scalar_probe_judge(
    up: Probe,
    down: Probe,
    *,
    strength: float,
    n_examples: int,
    runs: int,
    sigma_w: float,
    score_min: float,
    score_max: float,
    seed: int,
    model: str,
) -> Generated:
    """A self-consistent scalar judge whose score swings by ``strength`` of the scale
    between the up and down probe directions (and is unmoved under NEUTRAL).

    Each example has a stable per-example baseline shared by all three probe
    conditions; the up probe shifts its mean by ``+strength*range/2`` and the down probe
    by ``-strength*range/2``, so the measured ``mean(up) - mean(down)`` swing is exactly
    ``strength`` of the score range by construction. The shared baseline cancels in the
    difference, so the recovered effect is confound-free.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    span = score_max - score_min
    mid = (score_min + score_max) / 2.0
    half = strength * span / 2.0
    rng = np.random.default_rng(seed)
    examples = _scalar_examples(n_examples)
    records: list[JudgmentRecord] = []
    for ex in examples:
        base = mid + float(rng.normal(0.0, span / 8.0))
        for probe, shift in ((Probe.NEUTRAL, 0.0), (up, half), (down, -half)):
            for j in range(runs):
                score = base + shift + float(rng.normal(0.0, sigma_w))
                records.append(
                    JudgmentRecord(
                        example_id=ex.id,
                        run_index=j,
                        rubric_variant=0,
                        probe=probe,
                        ordering=None,
                        raw_response=json.dumps({"score": score}),
                        parse_ok=True,
                        score=score,
                        model=model,
                    )
                )
    return JudgmentSet(mode=JudgeMode.SCALAR, model=model, records=records), examples


def scalar_judge_with_sycophancy(
    *,
    strength: float,
    n_examples: int = 120,
    runs: int = 20,
    sigma_w: float = 0.3,
    score_min: float = 1.0,
    score_max: float = 10.0,
    seed: int = 0,
    model: str = "synthetic-sycophancy",
) -> Generated:
    """A scalar judge that caves to a stated score opinion by ``strength`` of the scale."""
    return _scalar_probe_judge(
        Probe.SYCOPHANCY_UP,
        Probe.SYCOPHANCY_DOWN,
        strength=strength,
        n_examples=n_examples,
        runs=runs,
        sigma_w=sigma_w,
        score_min=score_min,
        score_max=score_max,
        seed=seed,
        model=model,
    )


def scalar_judge_with_anchoring(
    *,
    strength: float,
    n_examples: int = 120,
    runs: int = 20,
    sigma_w: float = 0.3,
    score_min: float = 1.0,
    score_max: float = 10.0,
    seed: int = 0,
    model: str = "synthetic-anchoring",
) -> Generated:
    """A scalar judge that drifts toward an irrelevant reference score by ``strength``."""
    return _scalar_probe_judge(
        Probe.ANCHOR_UP,
        Probe.ANCHOR_DOWN,
        strength=strength,
        n_examples=n_examples,
        runs=runs,
        sigma_w=sigma_w,
        score_min=score_min,
        score_max=score_max,
        seed=seed,
        model=model,
    )


def pairwise_judge_with_sycophancy(
    *,
    strength: float,
    n_examples: int = 150,
    runs: int = 16,
    seed: int = 0,
    model: str = "synthetic-sycophancy-pairwise",
) -> Generated:
    """A pairwise judge that complies with a stated preference on ``strength`` of examples.

    Each example has a stable base winner returned under NEUTRAL (so headline
    self-consistency is ~perfect, no position flips). For a ``strength`` prefix of
    examples the judge fully caves: it returns the suggested content (A under the up
    probe, B under the down probe); the rest ignore the probe and return the base
    winner. So the win-rate-for-A swing between the two probe directions equals
    ``strength`` by construction.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    n_comply = round(strength * n_examples)
    rng = np.random.default_rng(seed)
    orderings = _orderings(runs)
    examples = _pairwise_examples(n_examples)
    records: list[JudgmentRecord] = []
    for i, ex in enumerate(examples):
        base = Winner.A if rng.random() < 0.5 else Winner.B
        if i < n_comply:
            conditions = [
                (Probe.NEUTRAL, base),
                (Probe.SYCOPHANCY_UP, Winner.A),
                (Probe.SYCOPHANCY_DOWN, Winner.B),
            ]
        else:
            conditions = [
                (Probe.NEUTRAL, base),
                (Probe.SYCOPHANCY_UP, base),
                (Probe.SYCOPHANCY_DOWN, base),
            ]
        for probe, winner in conditions:
            for j, ordering in enumerate(orderings):
                records.append(
                    JudgmentRecord(
                        example_id=ex.id,
                        run_index=j,
                        rubric_variant=0,
                        probe=probe,
                        ordering=ordering,
                        raw_response="[[A]]" if winner is Winner.A else "[[B]]",
                        parse_ok=True,
                        choice=_choice_for(winner, ordering),
                        winner=winner,
                        model=model,
                    )
                )
    return JudgmentSet(mode=JudgeMode.PAIRWISE, model=model, records=records), examples


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
