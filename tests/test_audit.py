from __future__ import annotations

import numpy as np

from judge_auditor.analysis.audit import audit
from judge_auditor.config import EvalExample, JudgeMode, PairwiseChoice, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet
from judge_auditor.report.terminal import render_terminal


def words(n: int) -> str:
    return " ".join(["w"] * n)


def reliable_scalar() -> tuple[JudgmentSet, list[EvalExample]]:
    # Tight within-example scores spread across the full 1..10 scale, length held
    # constant => consistent, no verbosity correlation, no scale compression.
    rng = np.random.default_rng(0)
    examples, records = [], []
    for i in range(20):
        base = float(1 + (i % 10))  # spread across the scale
        eid = f"ex{i}"
        examples.append(EvalExample(id=eid, prompt="q", response_a=words(50)))
        for j in range(8):
            s = base + rng.normal(0, 0.15)
            records.append(JudgmentRecord(eid, j, 0, None, str(s), True, score=s))
    return JudgmentSet(JudgeMode.SCALAR, "judge-x", records), examples


def compressed_scalar() -> tuple[JudgmentSet, list[EvalExample]]:
    # Everything clustered at 8 => compressed scale (and degenerate consistency).
    examples, records = [], []
    for i in range(20):
        eid = f"ex{i}"
        examples.append(EvalExample(id=eid, prompt="q", response_a=words(10)))
        for j in range(8):
            records.append(JudgmentRecord(eid, j, 0, None, "8", True, score=8.0))
    return JudgmentSet(JudgeMode.SCALAR, "judge-x", records), examples


def position_biased_pairwise() -> tuple[JudgmentSet, list[EvalExample]]:
    # Always picks the first-presented response => flips with order.
    examples, records = [], []
    for i in range(30):
        eid = f"ex{i}"
        examples.append(
            EvalExample(id=eid, prompt="q", response_a=words(20), response_b=words(20))
        )
        for j in range(6):
            ordering = "AB" if j % 2 == 0 else "BA"
            winner = Winner.A if ordering == "AB" else Winner.B  # first-presented wins
            records.append(
                JudgmentRecord(
                    eid, j, 0, ordering, "x", True,
                    choice=PairwiseChoice.FIRST, winner=winner,
                )
            )
    return JudgmentSet(JudgeMode.PAIRWISE, "judge-x", records), examples


def consistent_pairwise() -> tuple[JudgmentSet, list[EvalExample]]:
    # Always picks content A regardless of order => no position bias, perfect agreement.
    examples, records = [], []
    for i in range(30):
        eid = f"ex{i}"
        examples.append(
            EvalExample(id=eid, prompt="q", response_a=words(20), response_b=words(20))
        )
        for j in range(6):
            ordering = "AB" if j % 2 == 0 else "BA"
            # Prefers content A: FIRST when A shown first (AB), SECOND when (BA).
            choice = PairwiseChoice.FIRST if ordering == "AB" else PairwiseChoice.SECOND
            records.append(
                JudgmentRecord(eid, j, 0, ordering, "x", True, choice=choice, winner=Winner.A)
            )
    return JudgmentSet(JudgeMode.PAIRWISE, "judge-x", records), examples


def test_reliable_scalar_audit_is_high_and_clean():
    js, examples = reliable_scalar()
    rep = audit(js, examples)
    assert rep.mode is JudgeMode.SCALAR
    assert rep.overall == "HIGH"
    assert rep.notes == []
    assert rep.position is None
    assert rep.consistency.icc_oneway is not None
    assert rep.power.sigma_w is not None and rep.power.mde is not None


def test_compressed_scalar_audit_flags_scale():
    js, examples = compressed_scalar()
    rep = audit(js, examples)
    assert rep.scale.compressed
    assert any("Compressed scale" in n for n in rep.notes)
    assert rep.overall in ("LOW", "MODERATE")


def test_position_biased_pairwise_audit_is_low():
    js, examples = position_biased_pairwise()
    rep = audit(js, examples)
    assert rep.position is not None
    assert rep.position.favored_position == "first"
    assert rep.overall == "LOW"
    assert any("Position bias" in n for n in rep.notes)


def test_consistent_pairwise_audit_is_high_and_clean():
    js, examples = consistent_pairwise()
    rep = audit(js, examples)
    assert rep.position is not None
    assert rep.position.favored_position == "none"
    assert rep.overall == "HIGH"
    assert rep.notes == []


def test_render_terminal_scalar_contains_sections():
    js, examples = reliable_scalar()
    out = render_terminal(audit(js, examples))
    assert "JUDGE RELIABILITY: HIGH" in out
    assert "SELF-CONSISTENCY (scalar)" in out
    assert "SCALE ANALYSIS" in out
    assert "POWER / NOISE FLOOR (scalar)" in out
    assert "judge-x" in out


def test_render_terminal_pairwise_contains_position_section():
    js, examples = position_biased_pairwise()
    out = render_terminal(audit(js, examples))
    assert "POSITION BIAS" in out
    assert "FLAGS" in out
    assert "JUDGE RELIABILITY: LOW" in out
