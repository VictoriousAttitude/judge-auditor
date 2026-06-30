from __future__ import annotations

import numpy as np

from judge_auditor import synthetic as S
from judge_auditor.analysis.audit import audit
from judge_auditor.config import EvalExample, JudgeMode, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet
from judge_auditor.report.recommendations import recommendations


def words(n: int) -> str:
    return " ".join(["w"] * n)


def test_low_icc_scalar_recommends_more_runs():
    recs = recommendations(audit(*S.scalar_judge(icc=0.1, quantize=True, seed=3)))
    assert any("Self-consistency is" in r and "ICC=" in r for r in recs)


def test_scalar_verbosity_flag_recommends_length_instruction():
    rng = np.random.default_rng(0)
    examples, records = [], []
    lengths = [5, 12, 20, 30, 45, 60, 80, 100, 130, 170]
    for i, length in enumerate(lengths):
        eid = f"ex{i}"
        base = 1.0 + i * 0.9
        examples.append(EvalExample(id=eid, prompt="q", response_a=words(length)))
        for j in range(4):
            s = base + rng.normal(0, 0.1)
            records.append(JudgmentRecord(eid, j, 0, None, str(s), True, score=s))
    recs = recommendations(audit(JudgmentSet(JudgeMode.SCALAR, "m", records), examples))
    assert any("Verbosity correlation rho=" in r for r in recs)


def test_pairwise_verbosity_flag_recommends_length_control():
    examples, records = [], []
    lengths_a = [3, 6, 10, 15, 25, 40, 60, 90, 130, 180]
    for i, la in enumerate(lengths_a):
        eid = f"ex{i}"
        examples.append(
            EvalExample(id=eid, prompt="q", response_a=words(la), response_b=words(20))
        )
        winner = Winner.A if la > 20 else Winner.B  # the longer response wins
        for j in range(6):
            ordering = "AB" if j < 3 else "BA"
            records.append(JudgmentRecord(eid, j, 0, ordering, "x", True, winner=winner))
    recs = recommendations(audit(JudgmentSet(JudgeMode.PAIRWISE, "m", records), examples))
    assert any("longer response wins disproportionately" in r for r in recs)


def test_stratified_only_verbosity_recommends_length_instruction():
    # Within each quality stratum length moves the score, but the two strata move in
    # opposite directions so the global correlation cancels: only the stratified flag.
    rng = np.random.default_rng(0)
    examples, records = [], []
    lengths = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    idx = 0
    for q, sign in ((1.0, +1.0), (2.0, -1.0)):
        for k, length in enumerate(lengths):
            eid = f"ex{idx}"
            idx += 1
            base = 5.0 + sign * (k * 0.5)
            examples.append(
                EvalExample(id=eid, prompt="q", response_a=words(length), quality_label=q)
            )
            for j in range(4):
                s = base + rng.normal(0, 0.05)
                records.append(JudgmentRecord(eid, j, 0, None, str(s), True, score=s))
    rep = audit(JudgmentSet(JudgeMode.SCALAR, "m", records), examples)
    assert not rep.verbosity.flagged
    assert rep.verbosity.stratified_flagged
    assert any("equal-quality answers" in r for r in recommendations(rep))


def test_single_run_scalar_has_no_noise_floor_rec():
    examples, records = [], []
    for i in range(5):
        eid = f"ex{i}"
        examples.append(EvalExample(id=eid, prompt="q", response_a=words(10)))
        records.append(JudgmentRecord(eid, 0, 0, None, "5", True, score=5.0))
    recs = recommendations(audit(JudgmentSet(JudgeMode.SCALAR, "m", records), examples))
    assert not any("Noise floor" in r for r in recs)


def test_single_run_pairwise_has_no_noise_floor_rec():
    examples, records = [], []
    for i in range(5):
        eid = f"ex{i}"
        examples.append(
            EvalExample(id=eid, prompt="q", response_a=words(10), response_b=words(10))
        )
        records.append(JudgmentRecord(eid, 0, 0, "AB", "x", True, winner=Winner.A))
    recs = recommendations(audit(JudgmentSet(JudgeMode.PAIRWISE, "m", records), examples))
    assert not any("Noise floor" in r or "discriminating power" in r for r in recs)


def test_high_parse_failure_rate_recommends_structured_output():
    examples, records = [], []
    for i in range(10):
        eid = f"ex{i}"
        examples.append(EvalExample(id=eid, prompt="q", response_a=words(10)))
        for j in range(4):
            if j == 0:
                records.append(JudgmentRecord(eid, j, 0, None, "junk", False, parse_error="x"))
            else:
                records.append(JudgmentRecord(eid, j, 0, None, "5", True, score=5.0))
    recs = recommendations(audit(JudgmentSet(JudgeMode.SCALAR, "m", records), examples))
    assert any("Parse-failure rate" in r for r in recs)
