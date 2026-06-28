from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from judge_auditor import synthetic as S
from judge_auditor.analysis.consistency import consistency
from judge_auditor.analysis.stats import cohen_kappa, interpret_correlation
from judge_auditor.analysis.validity import _majority_winner, validity
from judge_auditor.config import EvalExample, JudgeMode, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet


def scalar_set(rows, runs: int = 4):
    """rows: list of (quality_label, constant_score) -> a scalar JudgmentSet."""
    examples, records = [], []
    for i, (q, score) in enumerate(rows):
        eid = f"e{i}"
        ql = None if q is None else float(q)
        examples.append(EvalExample(id=eid, prompt="p", response_a="r", quality_label=ql))
        for j in range(runs):
            records.append(JudgmentRecord(eid, j, 0, None, str(score), True, score=float(score)))
    return JudgmentSet(JudgeMode.SCALAR, "m", records), examples


def pairwise_set(rows, runs: int = 6):
    """rows: list of (truth_winner, judge_winner) -> a pairwise JudgmentSet."""
    examples, records = [], []
    for i, (truth, judge) in enumerate(rows):
        eid = f"e{i}"
        examples.append(
            EvalExample(id=eid, prompt="p", response_a="a", response_b="b", preferred_winner=truth)
        )
        for j in range(runs):
            ordering = "AB" if j < runs // 2 else "BA"
            records.append(JudgmentRecord(eid, j, 0, ordering, "x", True, winner=judge))
    return JudgmentSet(JudgeMode.PAIRWISE, "m", records), examples


# --- stats primitives -----------------------------------------------------------


def test_cohen_kappa_perfect_and_degenerate():
    assert cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == 1.0
    # Both raters use only one category: every pair agrees -> 1.0 (degenerate).
    assert cohen_kappa(["A", "A", "A"], ["A", "A", "A"]) == 1.0
    assert np.isnan(cohen_kappa(["A"], ["A"]))


def test_cohen_kappa_anticorrelated_is_negative():
    assert cohen_kappa(["A", "B", "A", "B"], ["B", "A", "B", "A"]) < 0.0


def test_interpret_correlation_bands():
    assert interpret_correlation(0.1) == "poor"
    assert interpret_correlation(0.4) == "weak"
    assert interpret_correlation(0.6) == "moderate"
    assert interpret_correlation(0.85) == "good"
    # Bands act on magnitude: a strong negative correlation is still poor validity.
    assert interpret_correlation(-0.9) == "good"  # |.9| -> good band, sign handled by caller
    assert interpret_correlation(-0.1) == "poor"


def test_majority_winner():
    assert _majority_winner([Winner.A, Winner.A, Winner.B]) is Winner.A
    assert _majority_winner([Winner.A, Winner.B]) is Winner.TIE  # no unique mode
    assert _majority_winner([Winner.TIE, Winner.TIE, Winner.A]) is Winner.TIE


# --- scalar validity ------------------------------------------------------------


def test_scalar_perfect_validity_not_flagged():
    rows = [(q, 2 * q) for q in range(1, 13)]  # score is a monotone function of quality
    js, exs = scalar_set(rows)
    res = validity(js, exs, n_boot=500)
    assert res.available
    assert res.n_labeled == 12
    assert res.pearson_r is not None and res.pearson_r.point > 0.99
    assert res.interpretation == "good"
    assert not res.flagged


def test_scalar_validity_is_scale_invariant():
    # Quality on a 1-5 scale, score on a ~1-10 scale: Pearson still recovers ~1.
    rows = [(q, 2 * q + 1) for q in [1, 2, 3, 4, 5] * 3]
    js, exs = scalar_set(rows)
    res = validity(js, exs, n_boot=300)
    assert res.pearson_r is not None and res.pearson_r.point > 0.99


def test_scalar_zero_validity_flagged():
    rng = np.random.default_rng(0)
    rows = [(float(q), float(rng.integers(1, 11))) for q in range(40)]  # score independent of q
    js, exs = scalar_set(rows)
    res = validity(js, exs, n_boot=800)
    assert res.available
    assert res.pearson_r is not None and res.pearson_r.high < 0.5
    assert res.flagged
    assert res.interpretation == "poor"


def test_scalar_pearson_matches_scipy():
    rng = np.random.default_rng(3)
    rows = [(float(q), float(q) + float(rng.normal(0, 2))) for q in range(30)]
    js, exs = scalar_set(rows)
    res = validity(js, exs, n_boot=200)
    scores = [s for _, s in rows]
    quals = [q for q, _ in rows]
    assert res.pearson_r is not None
    assert abs(res.pearson_r.point - float(pearsonr(scores, quals)[0])) < 1e-9


def test_scalar_absent_labels_unavailable():
    rows = [(None, float(s)) for s in range(10)]
    js, exs = scalar_set(rows)
    res = validity(js, exs)
    assert not res.available
    assert res.flagged is False
    assert res.pearson_r is None


def test_scalar_small_n_not_flagged():
    # Only 4 labeled examples: even zero correlation must not flag (below min_n and
    # the CI is too wide to rule out good validity).
    rows = [(1.0, 9.0), (2.0, 1.0), (3.0, 7.0), (4.0, 3.0)]
    js, exs = scalar_set(rows)
    res = validity(js, exs, n_boot=300)
    assert res.available
    assert res.n_labeled == 4
    assert not res.flagged


# --- pairwise validity ----------------------------------------------------------


def test_pairwise_perfect_validity_not_flagged():
    rows = [(Winner.A if i % 2 else Winner.B, Winner.A if i % 2 else Winner.B) for i in range(12)]
    js, exs = pairwise_set(rows)
    res = validity(js, exs, n_boot=500)
    assert res.available
    assert res.agreement_rate == 1.0
    assert res.accuracy_excl_ties == 1.0
    assert res.cohen_kappa is not None and res.cohen_kappa.point > 0.99
    assert not res.flagged


def test_pairwise_coin_validity_flagged():
    # Judge always says A; truth is balanced -> agreement ~0.5, kappa ~0 -> flagged.
    rows = [(Winner.A if i % 2 else Winner.B, Winner.A) for i in range(40)]
    js, exs = pairwise_set(rows)
    res = validity(js, exs, n_boot=800)
    assert res.agreement_rate is not None and abs(res.agreement_rate - 0.5) < 0.1
    assert res.cohen_kappa is not None and res.cohen_kappa.high < 0.4
    assert res.flagged


def test_pairwise_accuracy_excludes_ties():
    # Two decisive-correct, one where the judge ties (undecided): accuracy_excl_ties
    # counts only the two decisive pairs, agreement_rate counts all three.
    rows = [(Winner.A, Winner.A), (Winner.B, Winner.B), (Winner.A, Winner.TIE)]
    js, exs = pairwise_set(rows)
    res = validity(js, exs)
    assert res.n_decisive == 2
    assert res.accuracy_excl_ties == 1.0
    assert res.agreement_rate is not None and abs(res.agreement_rate - 2 / 3) < 1e-9


def test_pairwise_absent_labels_unavailable():
    examples, records = [], []
    for i in range(6):
        eid = f"e{i}"
        examples.append(EvalExample(id=eid, prompt="p", response_a="a", response_b="b"))
        records.append(JudgmentRecord(eid, 0, 0, "AB", "x", True, winner=Winner.A))
    js = JudgmentSet(JudgeMode.PAIRWISE, "m", records)
    res = validity(js, examples)
    assert not res.available
    assert res.cohen_kappa is None


# --- calibration: reliability vs validity are separated -------------------------


def test_scalar_known_validity_is_recovered():
    js, exs = S.scalar_judge_with_validity(rho=0.8, n_examples=200, runs=20, seed=1)
    res = validity(js, exs, n_boot=1000)
    assert res.pearson_r is not None
    assert abs(res.pearson_r.point - 0.8) < 0.1
    assert res.pearson_r.low <= 0.8 <= res.pearson_r.high


def test_consistent_but_invalid_scalar_judge_is_flagged():
    # rho=0: the judge is perfectly self-consistent yet uncorrelated with the truth.
    js, exs = S.scalar_judge_with_validity(rho=0.0, n_examples=200, runs=20, seed=2)
    cons = consistency(js, n_boot=400)
    val = validity(js, exs, n_boot=800)
    assert cons.icc_oneway is not None and cons.icc_oneway.point >= 0.75  # reliable
    assert val.flagged  # but invalid
    assert val.interpretation == "poor"


def test_pairwise_known_accuracy_is_recovered():
    js, exs = S.pairwise_judge_with_accuracy(0.9, n_examples=200, runs=16, seed=4)
    res = validity(js, exs, n_boot=800)
    assert res.agreement_rate is not None and abs(res.agreement_rate - 0.9) < 0.05
    assert res.accuracy_excl_ties is not None and abs(res.accuracy_excl_ties - 0.9) < 0.05
    assert not res.flagged


def test_consistent_but_invalid_pairwise_judge_is_flagged():
    js, exs = S.pairwise_judge_with_accuracy(0.5, n_examples=200, runs=16, seed=5)
    cons = consistency(js, n_boot=400)
    val = validity(js, exs, n_boot=800)
    assert cons.fleiss_kappa is not None and cons.fleiss_kappa.point >= 0.95  # reliable
    assert val.flagged  # but no better than a coin vs the truth
