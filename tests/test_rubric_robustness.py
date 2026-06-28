"""Unit + calibration tests for the rubric-robustness analysis.

The detector asks whether the judge's verdict survives paraphrasing the rubric. We
drive it with synthetic judges of known cross-variant character (robust vs brittle)
and assert it recovers the constructed agreement, stays silent without variants, and
respects the small-sample gate.
"""

from __future__ import annotations

import numpy as np

from judge_auditor import synthetic as S
from judge_auditor.analysis.audit import audit
from judge_auditor.analysis.rubric_robustness import _variants, rubric_robustness
from judge_auditor.config import JudgeMode, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet
from judge_auditor.report.recommendations import recommendations


def _scalar_record(eid: str, run: int, variant: int, score: float) -> JudgmentRecord:
    return JudgmentRecord(
        example_id=eid,
        run_index=run,
        rubric_variant=variant,
        ordering=None,
        raw_response="{}",
        parse_ok=True,
        score=score,
    )


def _pairwise_record(eid: str, run: int, variant: int, winner: Winner) -> JudgmentRecord:
    return JudgmentRecord(
        example_id=eid,
        run_index=run,
        rubric_variant=variant,
        ordering="AB",
        raw_response="x",
        parse_ok=True,
        winner=winner,
    )

# --- Availability gate ----------------------------------------------------------


def test_single_rubric_is_unavailable():
    # The ordinary generators emit only variant 0: nothing to compare.
    js, _ = S.scalar_judge(icc=0.90, n_examples=40, runs=10, seed=1)
    r = rubric_robustness(js)
    assert not r.available
    assert r.n_variants == 1
    assert not r.flagged


def test_variants_helper_lists_distinct_variants():
    js, _ = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=0.0, n_variants=3, n_examples=10, runs=4, seed=0
    )
    assert _variants(js) == [0, 1, 2]


# --- Scalar ---------------------------------------------------------------------


def test_scalar_robust_judge_has_high_cross_variant_icc():
    js, _ = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=0.0, n_variants=3, n_examples=120, runs=20, seed=2
    )
    r = rubric_robustness(js)
    assert r.available and r.n_variants == 3
    assert r.icc is not None and r.icc.point >= 0.90
    assert r.interpretation in ("good", "excellent")
    assert not r.flagged
    assert not r.severe


def test_scalar_brittle_judge_is_flagged_and_severe():
    js, _ = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=1.0, n_variants=3, n_examples=120, runs=20, seed=3
    )
    r = rubric_robustness(js)
    assert r.available
    assert r.icc is not None and r.icc.point < 0.30
    assert r.flagged
    assert r.severe


def test_scalar_cross_variant_icc_calibration():
    # Cross-variant ICC ~ (1-s)^2 / ((1-s)^2 + s^2); at s=0.5 that is ~0.5.
    js, _ = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=0.5, n_variants=3, n_examples=200, runs=20, seed=4
    )
    r = rubric_robustness(js)
    assert r.icc is not None
    assert abs(r.icc.point - 0.5) < 0.12
    assert r.icc.low <= 0.5 <= r.icc.high


def test_scalar_score_spread_grows_with_sensitivity():
    robust, _ = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=0.0, n_variants=3, n_examples=120, runs=20, seed=5
    )
    brittle, _ = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=1.0, n_variants=3, n_examples=120, runs=20, seed=5
    )
    rr = rubric_robustness(robust)
    rb = rubric_robustness(brittle)
    assert rr.mean_score_spread is not None and rb.mean_score_spread is not None
    assert rb.mean_score_spread > rr.mean_score_spread


# --- Pairwise -------------------------------------------------------------------


def test_pairwise_robust_judge_agrees_across_variants():
    js, _ = S.pairwise_judge_with_rubric_sensitivity(
        flip_fraction=0.0, n_variants=3, n_examples=120, runs=16, seed=6
    )
    r = rubric_robustness(js)
    assert r.available and r.mode is JudgeMode.PAIRWISE
    assert r.kappa is not None and r.kappa.point >= 0.99
    assert r.winner_flip_rate == 0.0
    assert not r.flagged


def test_pairwise_brittle_judge_is_flagged():
    js, _ = S.pairwise_judge_with_rubric_sensitivity(
        flip_fraction=1.0, n_variants=3, n_examples=120, runs=16, seed=7
    )
    r = rubric_robustness(js)
    assert r.kappa is not None and r.kappa.point < 0.40
    assert r.flagged
    assert r.severe


def test_pairwise_winner_flip_rate_recovers_flip_fraction():
    js, _ = S.pairwise_judge_with_rubric_sensitivity(
        flip_fraction=0.3, n_variants=2, n_examples=120, runs=16, seed=8
    )
    r = rubric_robustness(js)
    assert r.winner_flip_rate is not None
    assert abs(r.winner_flip_rate - 0.3) < 0.02
    assert r.n_flipped == 36


# --- Small-sample gate ----------------------------------------------------------


def test_small_sample_brittle_judge_is_not_flagged():
    # Below min_n the detector reports the estimate but refuses to flag.
    js, _ = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=1.0, n_variants=3, n_examples=5, runs=20, seed=9
    )
    r = rubric_robustness(js, min_n=8)
    assert r.available
    assert r.n_examples < 8
    assert not r.flagged


# --- Degenerate / incomplete data ----------------------------------------------


def test_scalar_unavailable_with_too_few_complete_examples():
    # Two variants but only one example has data for both: nothing to correlate.
    records = [
        _scalar_record("ex0", 0, 0, 5.0),
        _scalar_record("ex0", 0, 1, 6.0),
        _scalar_record("ex1", 0, 0, 4.0),  # ex1 missing variant 1 -> dropped
    ]
    js = JudgmentSet(mode=JudgeMode.SCALAR, model="m", records=records)
    r = rubric_robustness(js)
    assert not r.available
    assert r.n_examples == 1


def test_scalar_degenerate_data_has_no_interpretation():
    # All scores identical => ICC is undefined (NaN) => no interpretation, no flag.
    records = [
        _scalar_record(f"ex{i}", run, v, 5.0)
        for i in range(4)
        for v in (0, 1)
        for run in range(3)
    ]
    js = JudgmentSet(mode=JudgeMode.SCALAR, model="m", records=records)
    r = rubric_robustness(js)
    assert r.available
    assert r.interpretation is None
    assert not r.flagged


def test_pairwise_skips_incomplete_examples():
    records = [
        _pairwise_record("ex0", 0, 0, Winner.A),
        _pairwise_record("ex0", 0, 1, Winner.A),
        _pairwise_record("ex1", 0, 0, Winner.B),  # ex1 missing variant 1 -> dropped
    ]
    js = JudgmentSet(mode=JudgeMode.PAIRWISE, model="m", records=records)
    r = rubric_robustness(js)
    assert not r.available
    assert r.n_examples == 1


# --- Recommendations ------------------------------------------------------------


def test_scalar_brittle_judge_yields_recommendation():
    js, exs = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=1.0, n_variants=3, n_examples=120, runs=20, quantize=True, seed=31
    )
    recs = recommendations(audit(js, exs))
    assert any("brittleness" in r.lower() for r in recs)


def test_pairwise_brittle_judge_yields_recommendation():
    js, exs = S.pairwise_judge_with_rubric_sensitivity(
        flip_fraction=1.0, n_variants=3, n_examples=120, runs=16, seed=32
    )
    recs = recommendations(audit(js, exs))
    assert any("brittleness" in r.lower() for r in recs)


def test_robust_judge_yields_no_rubric_recommendation():
    js, exs = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=0.0, n_variants=3, n_examples=120, runs=20, quantize=True, seed=33
    )
    recs = recommendations(audit(js, exs))
    assert not any("brittleness" in r.lower() for r in recs)


def test_seeded_runs_are_deterministic():
    a, _ = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=0.5, n_variants=3, n_examples=60, runs=12, seed=42
    )
    b, _ = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=0.5, n_variants=3, n_examples=60, runs=12, seed=42
    )
    ra, rb = rubric_robustness(a, seed=1), rubric_robustness(b, seed=1)
    assert ra.icc is not None and rb.icc is not None
    assert np.isclose(ra.icc.point, rb.icc.point)
    assert np.isclose(ra.icc.low, rb.icc.low)
