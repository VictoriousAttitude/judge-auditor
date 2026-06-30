from __future__ import annotations

from judge_auditor import synthetic as S
from judge_auditor.analysis.audit import audit
from judge_auditor.config import EvalExample, JudgeMode, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet
from judge_auditor.report.comparison import comparison_row, render_comparison_markdown


def _reports():
    reliable = audit(*S.scalar_judge(icc=0.90, quantize=True, seed=2))
    biased = audit(*S.pairwise_judge_with_first_rate(1.0, seed=6))
    return [("Reliable", reliable), ("Biased", biased)]


def test_markdown_has_header_divider_and_one_row_per_judge():
    table = render_comparison_markdown(_reports())
    lines = table.splitlines()
    assert lines[0].startswith("| Judge | Mode |")
    assert set(lines[1].replace(" ", "")) == {"|", "-"}
    assert len(lines) == 2 + 2  # header + divider + two judges
    assert lines[2].startswith("| Reliable |")
    assert lines[3].startswith("| Biased |")


def test_scalar_row_reports_icc_and_pairwise_reports_kappa():
    reliable, biased = _reports()
    assert "ICC" in comparison_row(*reliable)[2]
    assert "kappa" in comparison_row(*biased)[2]


def test_position_biased_judge_row_shows_full_flip_and_no_power():
    _, biased = _reports()
    row = comparison_row(*biased)
    assert row[3] == "100%"  # position flip
    assert row[6] == "no power"  # zero discriminability noise floor
    assert row[7] == "LOW"  # verdict


def test_scalar_judge_has_no_position_cell():
    reliable, _ = _reports()
    assert comparison_row(*reliable)[3] == "n/a"


def test_consistent_pairwise_row_shows_finite_margin():
    rep = audit(*S.consistent_pairwise_judge(seed=1))
    assert comparison_row("C", rep)[6].endswith("margin")


def test_degenerate_scalar_consistency_cell_is_na():
    # A single-run scalar judge has no ICC basis, so the cell falls back to "n/a".
    exs = [EvalExample(id="ex0", prompt="q", response_a="r")]
    rec = JudgmentRecord("ex0", 0, 0, None, "5", True, score=5.0)
    js = JudgmentSet(JudgeMode.SCALAR, "m", [rec])
    assert comparison_row("S", audit(js, exs))[2] == "n/a"


def test_degenerate_pairwise_consistency_cell_is_na():
    exs = [EvalExample(id="ex0", prompt="q", response_a="a", response_b="b")]
    js = JudgmentSet(
        JudgeMode.PAIRWISE, "m", [JudgmentRecord("ex0", 0, 0, "AB", "x", True, winner=Winner.A)]
    )
    assert comparison_row("P", audit(js, exs))[2] == "n/a"
