from __future__ import annotations

from judge_auditor.analysis.scale_analysis import scale_analysis
from judge_auditor.config import JudgeMode, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet


def scalar_set(scores: list[float]) -> JudgmentSet:
    records = [
        JudgmentRecord(f"ex{i}", 0, 0, None, str(s), True, score=s)
        for i, s in enumerate(scores)
    ]
    return JudgmentSet(JudgeMode.SCALAR, "m", records)


def pairwise_set(winners: list[Winner]) -> JudgmentSet:
    records = [
        JudgmentRecord(f"ex{i}", 0, 0, "AB", "x", True, winner=w)
        for i, w in enumerate(winners)
    ]
    return JudgmentSet(JudgeMode.PAIRWISE, "m", records)


def test_uniform_distribution_high_effective_range():
    # Every integer 1..10 used equally => normalized entropy == 1.0.
    scores = [float(v) for v in range(1, 11)] * 10
    res = scale_analysis(scores_set := scalar_set(scores))
    assert res.num_bins == 10
    assert res.distinct_values_used == 10
    assert abs(res.effective_range - 1.0) < 1e-9
    assert not res.compressed
    assert scores_set.mode is JudgeMode.SCALAR


def test_all_same_value_minimal_range_and_compressed():
    res = scale_analysis(scalar_set([7.0] * 50))
    assert res.distinct_values_used == 1
    assert res.effective_range == 0.0
    assert res.compressed
    assert res.max_window_fraction == 1.0
    assert 7.0 in res.compressed_values


def test_compression_flag_three_adjacent_bins():
    # 80% of scores in {7,8,9}, 20% spread elsewhere => compressed.
    scores = [7.0] * 30 + [8.0] * 30 + [9.0] * 20 + [2.0] * 10 + [4.0] * 10
    res = scale_analysis(scalar_set(scores))
    assert res.compressed
    assert res.max_window_fraction > 0.70
    assert sorted(res.compressed_values) == [7.0, 8.0, 9.0]


def test_well_spread_not_compressed():
    # Even split across 1,4,7,10 => no 3-adjacent window exceeds 70%.
    scores = ([1.0] * 25) + ([4.0] * 25) + ([7.0] * 25) + ([10.0] * 25)
    res = scale_analysis(scalar_set(scores))
    assert not res.compressed
    assert res.max_window_fraction < 0.70


def test_scores_clamped_into_range():
    # Out-of-range integers (if ever parsed) clamp to the edge bins.
    res = scale_analysis(scalar_set([0.0, 11.0, 5.0]), score_min=1.0, score_max=10.0)
    assert res.n == 3
    assert res.histogram[0] == 1  # 0 -> bin for 1
    assert res.histogram[-1] == 1  # 11 -> bin for 10


def test_custom_scale_range():
    res = scale_analysis(scalar_set([1.0, 2.0, 3.0, 4.0, 5.0]), score_min=1.0, score_max=5.0)
    assert res.num_bins == 5
    assert res.bin_values == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert abs(res.effective_range - 1.0) < 1e-9


def test_pairwise_low_tie_rate_distinguishable():
    winners = [Winner.A] * 45 + [Winner.B] * 45 + [Winner.TIE] * 10
    res = scale_analysis(pairwise_set(winners))
    assert res.tie_rate == 0.1
    assert res.n_decisions == 90
    assert not res.indistinguishable


def test_pairwise_high_tie_rate_indistinguishable():
    winners = [Winner.A] * 25 + [Winner.B] * 25 + [Winner.TIE] * 50
    res = scale_analysis(pairwise_set(winners))
    assert res.tie_rate == 0.5
    assert res.indistinguishable


def test_empty_scalar_set_is_safe():
    res = scale_analysis(scalar_set([]))
    assert res.n == 0
    assert not res.compressed
