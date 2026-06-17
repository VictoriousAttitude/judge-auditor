"""Position bias (pairwise only): does swapping response order change the verdict?

Two complementary views, both computed from the same AB/BA runs the executor
already collected (no extra experiment):

* **First-position preference rate** — among non-tie decisions, how often the
  judge picks the *first-presented* response. A fair judge sits at 0.5; we test
  the deviation with an exact binomial test and report a Wilson CI. This is the
  most statistically powerful position-bias signal because it aggregates every
  decision, not just the contradictory ones.
* **Flip rate** — the fraction of examples whose *majority* canonical winner
  differs between the A,B ordering and the B,A ordering. This is the practical
  "how often is my verdict unstable under swapping" number from Zheng et al.

A judge with a high first-position rate AND a high flip rate is systematically
order-biased; a high flip rate with a ~0.5 first rate is just noisy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binomtest

from ..config import PairwiseChoice, Winner
from ..records import JudgmentSet
from .stats import CI, wilson_ci

_CATEGORIES = (Winner.A, Winner.B, Winner.TIE)


@dataclass
class PositionBiasResult:
    # First-position preference (direction + magnitude).
    n_decisions: int  # non-tie decisions
    first_position_rate: CI  # Wilson CI; point = observed rate
    first_preference_p_value: float  # exact binomial test vs 0.5
    favored_position: str  # "first" | "second" | "none"

    # Verdict instability under swapping.
    n_examples: int  # examples with both orderings present
    n_flipped: int
    flip_rate: CI  # Wilson CI on the majority-flip rate
    flipped_examples: list[str] = field(default_factory=list)

    tie_rate: float = 0.0  # fraction of all valid decisions that were ties


def _modal_winner(winners: list[Winner]) -> Winner:
    """Most frequent canonical winner, tie-broken by (A, B, tie) priority."""
    counts = [sum(w is c for w in winners) for c in _CATEGORIES]
    return _CATEGORIES[int(np.argmax(counts))]


def position_bias(js: JudgmentSet, *, alpha: float = 0.05) -> PositionBiasResult:
    """Compute the position-bias result for a pairwise judgment set."""
    first = second = ties = 0
    ab: dict[str, list[Winner]] = defaultdict(list)
    ba: dict[str, list[Winner]] = defaultdict(list)

    for r in js.records:
        if not r.parse_ok or r.choice is None:
            continue
        if r.choice is PairwiseChoice.FIRST:
            first += 1
        elif r.choice is PairwiseChoice.SECOND:
            second += 1
        else:
            ties += 1
        if r.winner is not None:
            if r.ordering == "AB":
                ab[r.example_id].append(r.winner)
            elif r.ordering == "BA":
                ba[r.example_id].append(r.winner)

    total = first + second + ties
    n_decisions = first + second

    # First-position preference.
    rate_ci = wilson_ci(first, n_decisions)
    if n_decisions > 0:
        p_value = float(binomtest(first, n_decisions, 0.5, alternative="two-sided").pvalue)
    else:
        p_value = float("nan")
    favored = "none"
    if n_decisions > 0 and p_value < alpha:
        favored = "first" if first / n_decisions > 0.5 else "second"

    # Flip rate over examples evaluated in both orderings.
    both = sorted(set(ab) & set(ba))
    flipped = [
        eid for eid in both if _modal_winner(ab[eid]) is not _modal_winner(ba[eid])
    ]
    flip_ci = wilson_ci(len(flipped), len(both))

    return PositionBiasResult(
        n_decisions=n_decisions,
        first_position_rate=rate_ci,
        first_preference_p_value=p_value,
        favored_position=favored,
        n_examples=len(both),
        n_flipped=len(flipped),
        flip_rate=flip_ci,
        flipped_examples=flipped,
        tie_rate=(ties / total) if total else 0.0,
    )
