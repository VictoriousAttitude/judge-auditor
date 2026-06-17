from __future__ import annotations

from judge_auditor.analysis.position_bias import position_bias
from judge_auditor.config import JudgeMode, PairwiseChoice
from judge_auditor.records import JudgmentRecord, JudgmentSet
from judge_auditor.runner.executor import _canonical_winner


def pairwise_set(per_example: list[list[tuple[str, PairwiseChoice]]]) -> JudgmentSet:
    records = []
    for i, runs in enumerate(per_example):
        for j, (ordering, choice) in enumerate(runs):
            winner = _canonical_winner(choice, ordering)
            records.append(
                JudgmentRecord(f"ex{i}", j, 0, ordering, "x", True, choice=choice, winner=winner)
            )
    return JudgmentSet(mode=JudgeMode.PAIRWISE, model="m", records=records)


def always_first_runs(k: int = 10) -> list[tuple[str, PairwiseChoice]]:
    half = k // 2
    return [("AB", PairwiseChoice.FIRST)] * half + [("BA", PairwiseChoice.FIRST)] * half


def content_consistent_runs(k: int = 10) -> list[tuple[str, PairwiseChoice]]:
    # Prefers content A regardless of order: FIRST when A is shown first (AB),
    # SECOND when A is shown second (BA). Canonical winner is always A.
    half = k // 2
    return [("AB", PairwiseChoice.FIRST)] * half + [("BA", PairwiseChoice.SECOND)] * half


def test_always_first_judge_is_maximally_biased():
    res = position_bias(pairwise_set([always_first_runs() for _ in range(40)]))
    assert res.first_position_rate.point == 1.0
    assert res.first_preference_p_value < 0.05
    assert res.favored_position == "first"
    assert res.flip_rate.point == 1.0  # winner flips A<->B with order every time
    assert res.n_flipped == 40


def test_content_consistent_judge_has_no_position_bias():
    res = position_bias(pairwise_set([content_consistent_runs() for _ in range(40)]))
    assert abs(res.first_position_rate.point - 0.5) < 1e-9
    assert res.first_preference_p_value > 0.05
    assert res.favored_position == "none"
    assert res.flip_rate.point == 0.0
    assert res.n_flipped == 0


def test_recovers_injected_flip_rate():
    n, flips = 200, 30  # 15% injected flip rate
    per_example = [always_first_runs() for _ in range(flips)] + [
        content_consistent_runs() for _ in range(n - flips)
    ]
    res = position_bias(pairwise_set(per_example))
    assert res.n_examples == n
    assert res.n_flipped == flips
    assert abs(res.flip_rate.point - 0.15) < 1e-9
    assert res.flip_rate.low < 0.15 < res.flip_rate.high


def test_exact_half_preference_is_not_significant():
    # 20 FIRST + 20 SECOND decisions => rate 0.5, binomial p == 1.0.
    runs = [("AB", PairwiseChoice.FIRST)] * 20 + [("AB", PairwiseChoice.SECOND)] * 20
    res = position_bias(pairwise_set([runs]))
    assert res.first_position_rate.point == 0.5
    assert res.first_preference_p_value == 1.0
    assert res.favored_position == "none"


def test_tie_rate_and_decisions_exclude_ties():
    runs = (
        [("AB", PairwiseChoice.FIRST)] * 4
        + [("AB", PairwiseChoice.SECOND)] * 4
        + [("AB", PairwiseChoice.TIE)] * 2
    )
    res = position_bias(pairwise_set([runs]))
    assert res.tie_rate == 0.2  # 2 of 10
    assert res.n_decisions == 8  # ties excluded from the preference test


def test_second_position_bias_detected():
    runs = [("AB", PairwiseChoice.SECOND)] * 10  # always second-presented
    res = position_bias(pairwise_set([runs for _ in range(30)]))
    assert res.first_position_rate.point == 0.0
    assert res.first_preference_p_value < 0.05
    assert res.favored_position == "second"
