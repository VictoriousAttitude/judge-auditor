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


# --- Validity: a precise-but-wrong judge is caught (reliability != validity) -----


def test_consistent_but_invalid_scalar_judge_is_downgraded():
    # Reliable (high ICC) yet uncorrelated with the ground truth: the audit must
    # downgrade it on validity even though every reliability signal is clean.
    js, exs = S.scalar_judge_with_validity(rho=0.0, n_examples=150, runs=20, seed=11)
    r = audit(js, exs)
    assert r.consistency.icc_oneway is not None and r.consistency.icc_oneway.point >= 0.75
    assert r.validity.available and r.validity.flagged
    assert r.overall == "LOW"
    assert any("validity" in n.lower() for n in r.notes)


def test_consistent_but_invalid_pairwise_judge_is_downgraded():
    js, exs = S.pairwise_judge_with_accuracy(0.5, n_examples=150, runs=16, seed=12)
    r = audit(js, exs)
    assert r.consistency.fleiss_kappa is not None and r.consistency.fleiss_kappa.point >= 0.95
    assert r.validity.available and r.validity.flagged
    assert r.overall == "LOW"
    assert any("validity" in n.lower() for n in r.notes)


def test_valid_scalar_judge_keeps_high_verdict():
    js, exs = S.scalar_judge_with_validity(rho=0.9, n_examples=150, runs=20, seed=13)
    r = audit(js, exs)
    assert r.validity.available
    assert not r.validity.flagged
    assert r.validity.interpretation in ("good", "moderate")
    assert r.overall == "HIGH"


def test_validity_silent_without_ground_truth():
    # The existing generators attach no labels: validity must stay unavailable and
    # never touch the verdict (no regression for users without ground truth).
    js, exs = S.scalar_judge(icc=0.90, n_examples=120, runs=20, quantize=True, seed=2)
    r = audit(js, exs)
    assert not r.validity.available
    assert not r.validity.flagged
    assert r.overall == "HIGH"


# --- Rubric robustness: a verdict that depends on rubric phrasing is caught ------


def test_brittle_scalar_rubric_judge_is_downgraded():
    # Reliable under each rubric phrasing, yet the score depends on which phrasing was
    # used: the audit must downgrade on rubric brittleness even though variant-0
    # self-consistency is clean.
    js, exs = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=1.0, n_variants=3, n_examples=120, runs=20, quantize=True, seed=21
    )
    r = audit(js, exs)
    assert r.consistency.icc_oneway is not None and r.consistency.icc_oneway.point >= 0.75
    assert r.rubric.available and r.rubric.flagged
    assert r.overall == "LOW"
    assert any("rubric" in n.lower() for n in r.notes)


def test_brittle_pairwise_rubric_judge_is_downgraded():
    js, exs = S.pairwise_judge_with_rubric_sensitivity(
        flip_fraction=1.0, n_variants=3, n_examples=120, runs=16, seed=22
    )
    r = audit(js, exs)
    assert r.consistency.fleiss_kappa is not None and r.consistency.fleiss_kappa.point >= 0.95
    assert r.rubric.available and r.rubric.flagged
    assert r.overall == "LOW"
    assert any("rubric" in n.lower() for n in r.notes)


def test_robust_rubric_judge_keeps_high_verdict():
    js, exs = S.scalar_judge_with_rubric_sensitivity(
        sensitivity=0.0, n_variants=3, n_examples=120, runs=20, quantize=True, seed=23
    )
    r = audit(js, exs)
    assert r.rubric.available and not r.rubric.flagged
    assert r.overall == "HIGH"
    assert r.notes == []


def test_rubric_robustness_silent_with_single_rubric():
    # A single-rubric audit (the common case) must not gain a rubric flag and the
    # headline metrics are unchanged (no regression).
    js, exs = S.scalar_judge(icc=0.90, n_examples=120, runs=20, quantize=True, seed=2)
    r = audit(js, exs)
    assert not r.rubric.available
    assert not r.rubric.flagged
    assert r.overall == "HIGH"
