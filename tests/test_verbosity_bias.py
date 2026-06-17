from __future__ import annotations

import numpy as np

from judge_auditor.analysis.verbosity_bias import verbosity_bias, word_count
from judge_auditor.config import EvalExample, JudgeMode, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet


def words(n: int) -> str:
    return " ".join(["w"] * n)


def scalar_case(scores_by_length: dict[int, float], runs: int = 3):
    examples, records = [], []
    for i, (length, score) in enumerate(scores_by_length.items()):
        eid = f"ex{i}"
        examples.append(EvalExample(id=eid, prompt="q", response_a=words(length)))
        for j in range(runs):
            records.append(JudgmentRecord(eid, j, 0, None, str(score), True, score=score))
    return JudgmentSet(JudgeMode.SCALAR, "m", records), examples


def test_scalar_length_correlated_scores_flag():
    # Score increases monotonically with length => Spearman rho ~ 1.
    data = {5: 2.0, 10: 4.0, 20: 6.0, 40: 8.0, 80: 9.0}
    js, examples = scalar_case(data)
    res = verbosity_bias(js, examples)
    assert res.spearman_rho is not None and res.spearman_rho > 0.95
    assert res.flagged


def test_scalar_length_independent_scores_not_flagged():
    rng = np.random.default_rng(0)
    examples, records = [], []
    for i in range(40):
        eid = f"ex{i}"
        length = int(rng.integers(5, 100))
        score = float(rng.integers(1, 11))  # independent of length
        examples.append(EvalExample(id=eid, prompt="q", response_a=words(length)))
        for j in range(3):
            records.append(JudgmentRecord(eid, j, 0, None, str(score), True, score=score))
    js = JudgmentSet(JudgeMode.SCALAR, "m", records)
    res = verbosity_bias(js, examples)
    assert abs(res.spearman_rho) < 0.4
    assert not res.flagged


def test_scalar_partial_correlation_removes_quality_confound():
    # Length and score are correlated only THROUGH quality: longer == higher
    # quality == higher score. Controlling for quality should null the partial rho.
    # Length and quality are correlated but NOT perfectly collinear (noise added),
    # otherwise the partial-correlation denominator degenerates to zero.
    rng = np.random.default_rng(0)
    examples, records = [], []
    for i in range(40):
        quality = float(i)
        # length tracks quality with independent noise (avoids rank collinearity)
        length = max(1, int(round(5 + 2 * i + rng.normal(0, 6))))
        # score tracks quality (not length) with its own independent noise, so the
        # score~quality rank correlation is high but < 1 (denominator stays nonzero)
        score = quality + rng.normal(0, 1.5)
        eid = f"ex{i}"
        examples.append(
            EvalExample(id=eid, prompt="q", response_a=words(length), quality_label=quality)
        )
        records.append(JudgmentRecord(eid, 0, 0, None, str(score), True, score=score))
        records.append(JudgmentRecord(eid, 1, 0, None, str(score), True, score=score))
    js = JudgmentSet(JudgeMode.SCALAR, "m", records)
    res = verbosity_bias(js, examples)
    assert res.spearman_rho > 0.9  # raw correlation is strong (via quality)
    assert res.partial_rho is not None and abs(res.partial_rho) < 0.2  # vanishes


def test_pairwise_longer_response_wins():
    # response_a is always longer AND always wins => longer-win-rate ~ 1, rho > 0.
    examples, records = [], []
    for i in range(20):
        eid = f"ex{i}"
        examples.append(
            EvalExample(id=eid, prompt="q", response_a=words(50 + i), response_b=words(5))
        )
        for j in range(6):
            ordering = "AB" if j < 3 else "BA"
            records.append(
                JudgmentRecord(eid, j, 0, ordering, "x", True, winner=Winner.A)
            )
    js = JudgmentSet(JudgeMode.PAIRWISE, "m", records)
    res = verbosity_bias(js, examples)
    assert res.longer_response_win_rate is not None
    assert res.longer_response_win_rate > 0.95


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("  spaced   out  ") == 2
    assert word_count("") == 0
