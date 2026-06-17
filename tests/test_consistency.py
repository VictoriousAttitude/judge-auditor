from __future__ import annotations

import numpy as np

from judge_auditor.analysis.consistency import (
    consistency,
    fleiss_kappa,
    icc_oneway,
    icc_twoway,
)
from judge_auditor.config import JudgeMode, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet


def scalar_set(matrix: np.ndarray) -> JudgmentSet:
    records = []
    for i, row in enumerate(matrix):
        for j, s in enumerate(row):
            records.append(
                JudgmentRecord(f"ex{i}", j, 0, None, str(s), True, score=float(s))
            )
    return JudgmentSet(mode=JudgeMode.SCALAR, model="m", records=records)


def pairwise_set(rows: list[list[Winner]]) -> JudgmentSet:
    records = []
    for i, row in enumerate(rows):
        for j, w in enumerate(row):
            records.append(JudgmentRecord(f"ex{i}", j, 0, "AB", "x", True, winner=w))
    return JudgmentSet(mode=JudgeMode.PAIRWISE, model="m", records=records)


def variance_component_matrix(true_icc: float, n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sb = np.sqrt(true_icc)
    sw = np.sqrt(1.0 - true_icc)
    row = rng.normal(0.0, sb, size=(n, 1))
    noise = rng.normal(0.0, sw, size=(n, k))
    return 5.0 + row + noise


# --- ICC math -------------------------------------------------------------------


def test_icc_recovers_known_value():
    m = variance_component_matrix(0.6, n=150, k=15, seed=0)
    assert abs(icc_oneway(m) - 0.6) < 0.08


def test_icc_oneway_and_twoway_converge_for_exchangeable_runs():
    # No crossed run-position effect => the two ICC forms should nearly coincide.
    m = variance_component_matrix(0.7, n=150, k=15, seed=1)
    assert abs(icc_oneway(m) - icc_twoway(m)) < 0.02


def test_icc_perfect_agreement():
    m = np.tile(np.arange(40, dtype=float).reshape(-1, 1), (1, 10))
    assert abs(icc_oneway(m) - 1.0) < 1e-9


def test_icc_pure_noise_near_zero():
    rng = np.random.default_rng(3)
    m = rng.normal(0.0, 1.0, size=(200, 15))  # no between-example signal
    assert abs(icc_oneway(m)) < 0.1


# --- Fleiss' kappa --------------------------------------------------------------


def test_fleiss_perfect_agreement():
    # Every example unanimous; winners differ across examples.
    rows = [[Winner.A] * 12 for _ in range(20)] + [[Winner.B] * 12 for _ in range(20)]
    counts = np.asarray(
        [[row.count(c) for c in (Winner.A, Winner.B, Winner.TIE)] for row in rows],
        dtype=float,
    )
    assert abs(fleiss_kappa(counts) - 1.0) < 1e-9


def test_fleiss_random_near_zero():
    rng = np.random.default_rng(4)
    cats = [Winner.A, Winner.B, Winner.TIE]
    rows = [[cats[i] for i in rng.integers(0, 3, 15)] for _ in range(200)]
    counts = np.asarray(
        [[row.count(c) for c in cats] for row in rows], dtype=float
    )
    assert abs(fleiss_kappa(counts)) < 0.1


def _kappa_for_split(per_example_majority: int, seed: int) -> float:
    """Build 80 examples, each with `per_example_majority`/(15-...) agreement."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(80):
        modal = Winner.A if i % 2 == 0 else Winner.B
        other = Winner.B if modal is Winner.A else Winner.A
        row = [modal] * per_example_majority + [other] * (15 - per_example_majority)
        rng.shuffle(row)
        rows.append(row)
    counts = np.asarray(
        [[row.count(c) for c in (Winner.A, Winner.B, Winner.TIE)] for row in rows],
        dtype=float,
    )
    return fleiss_kappa(counts)


def test_fleiss_kappa_increases_with_agreement():
    # Monotone in per-example agreement. (With balanced 50/50 marginals, chance
    # agreement is high, so a 13/2 split lands in the "moderate" band, not high.)
    k_unanimous = _kappa_for_split(15, seed=5)
    k_strong = _kappa_for_split(13, seed=5)
    k_weak = _kappa_for_split(9, seed=5)
    assert abs(k_unanimous - 1.0) < 1e-9
    assert 0.4 < k_strong < 0.65
    assert k_unanimous > k_strong > k_weak
    assert k_weak < 0.2


# --- consistency() dispatch -----------------------------------------------------


def test_consistency_scalar_result():
    m = variance_component_matrix(0.6, n=150, k=15, seed=7)
    res = consistency(scalar_set(m), n_boot=300, seed=0)
    assert res.mode is JudgeMode.SCALAR
    assert res.icc_oneway is not None and res.icc_twoway is not None
    assert abs(res.icc_oneway.point - 0.6) < 0.1
    assert res.icc_oneway.low < 0.6 < res.icc_oneway.high
    assert res.icc_interpretation == "moderate"
    assert res.mean_within_sd is not None and abs(res.mean_within_sd - np.sqrt(0.4)) < 0.1


def test_consistency_pairwise_result_unanimous():
    rows = [[Winner.A] * 10 for _ in range(15)] + [[Winner.B] * 10 for _ in range(15)]
    res = consistency(pairwise_set(rows), n_boot=300, seed=0)
    assert res.mode is JudgeMode.PAIRWISE
    assert res.fleiss_kappa is not None
    assert abs(res.fleiss_kappa.point - 1.0) < 1e-9
    assert res.kappa_interpretation == "almost perfect"
    assert res.mean_agreement == 1.0
    assert res.min_agreement == 1.0


def test_consistency_handles_too_few_examples():
    res = consistency(scalar_set(np.array([[5.0, 5.0, 5.0]])), n_boot=50)
    assert res.icc_oneway is None
    assert res.runs_per_example == 0
