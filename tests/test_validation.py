"""End-to-end validation (plan S.8): the tool detects what we inject, stays quiet
when there is nothing to find, and recovers known statistical parameters.

These are the falsifiability tests. They run the full ``audit`` pipeline over
synthetic judges whose properties are fixed by construction (no API key, fully
seeded) and assert the report matches the ground truth.
"""

from __future__ import annotations

from judge_auditor import synthetic as S
from judge_auditor.analysis.audit import audit

# --- 8.3 Calibration: known statistics are recovered ----------------------------


def test_known_icc_is_recovered_within_ci():
    js, exs = S.scalar_judge(icc=0.80, n_examples=150, runs=20, seed=1)
    r = audit(js, exs)
    icc = r.consistency.icc_oneway
    assert icc is not None
    assert abs(icc.point - 0.80) < 0.07
    assert icc.low <= 0.80 <= icc.high


def test_known_flip_rate_is_recovered():
    js, exs = S.pairwise_judge_with_flip_rate(0.15, n_examples=200, runs=16, seed=5)
    r = audit(js, exs)
    assert r.position is not None
    flip = r.position.flip_rate
    assert abs(flip.point - 0.15) < 0.02
    assert flip.low <= 0.15 <= flip.high


def test_known_first_position_rate_is_recovered():
    js, exs = S.pairwise_judge_with_first_rate(0.75, n_examples=200, runs=16, seed=7)
    r = audit(js, exs)
    assert r.position is not None
    rate = r.position.first_position_rate
    assert abs(rate.point - 0.75) < 0.05
    assert r.position.favored_position == "first"
    assert r.position.first_preference_p_value < 0.01


# --- 8.1 Known-bias: an injected bias is detected with confidence ---------------


def test_position_biased_judge_is_flagged_low():
    js, exs = S.pairwise_judge_with_first_rate(1.0, n_examples=120, runs=16, seed=6)
    r = audit(js, exs)
    assert r.position is not None
    assert r.position.favored_position == "first"
    assert r.position.flip_rate.point > 0.9
    assert r.overall == "LOW"


def test_noisy_scalar_judge_is_flagged_low():
    js, exs = S.scalar_judge(icc=0.20, n_examples=120, runs=20, quantize=True, seed=3)
    r = audit(js, exs)
    assert r.consistency.icc_oneway is not None
    assert r.consistency.icc_oneway.point < 0.50
    assert r.overall == "LOW"


# --- 8.2 Null: a clean judge produces a clean report ----------------------------


def test_reliable_scalar_judge_passes_clean():
    js, exs = S.scalar_judge(icc=0.90, n_examples=120, runs=20, quantize=True, seed=2)
    r = audit(js, exs)
    assert r.consistency.icc_oneway is not None
    assert r.consistency.icc_oneway.point >= 0.75
    assert not r.scale.compressed
    assert not r.verbosity.flagged
    assert r.power.mde is not None and r.power.mde < 1.0
    assert r.overall == "HIGH"
    assert r.notes == []


def test_consistent_pairwise_judge_passes_clean():
    js, exs = S.consistent_pairwise_judge(n_examples=120, runs=16, seed=8)
    r = audit(js, exs)
    assert r.consistency.fleiss_kappa is not None
    assert r.consistency.fleiss_kappa.point >= 0.80
    assert r.position is not None
    assert r.position.favored_position == "none"
    assert r.position.flip_rate.point < 0.05
    assert r.overall == "HIGH"
    assert r.notes == []
