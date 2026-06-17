from __future__ import annotations

from judge_auditor import synthetic as S
from judge_auditor.analysis.audit import audit
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
