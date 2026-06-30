from __future__ import annotations

from judge_auditor.config import JudgeMode, PairwiseChoice, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet


def test_record_roundtrip_pairwise():
    rec = JudgmentRecord(
        example_id="ex1",
        run_index=2,
        rubric_variant=0,
        ordering="BA",
        raw_response="[[A]]",
        parse_ok=True,
        choice=PairwiseChoice.FIRST,
        winner=Winner.B,
        model="mock",
    )
    back = JudgmentRecord.from_dict(rec.to_dict())
    assert back == rec
    assert back.choice is PairwiseChoice.FIRST
    assert back.winner is Winner.B


def test_set_roundtrip_and_helpers():
    records = [
        JudgmentRecord("ex1", 0, 0, None, "7", True, score=7.0),
        JudgmentRecord("ex1", 1, 0, None, "bad", False, parse_error="x"),
        JudgmentRecord("ex2", 0, 0, None, "5", True, score=5.0),
    ]
    js = JudgmentSet(mode=JudgeMode.SCALAR, model="mock", records=records)
    assert js.example_ids == ["ex1", "ex2"]
    assert len(js.for_example("ex1")) == 2
    assert js.parse_failure_rate == 1 / 3

    back = JudgmentSet.from_dict(js.to_dict())
    assert back.mode is JudgeMode.SCALAR
    assert len(back) == 3
    assert back.records[0].score == 7.0


def test_set_save_load_json(tmp_path):
    records = [JudgmentRecord("ex1", 0, 0, "AB", "[[A]]", True, winner=Winner.A)]
    js = JudgmentSet(mode=JudgeMode.PAIRWISE, model="mock", records=records)
    path = str(tmp_path / "set.json")
    js.save_json(path)
    loaded = JudgmentSet.load_json(path)
    assert loaded.records[0].winner is Winner.A


def test_empty_set_parse_failure_rate_is_zero():
    js = JudgmentSet(mode=JudgeMode.SCALAR, model="mock", records=[])
    assert js.parse_failure_rate == 0.0
