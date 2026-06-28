from __future__ import annotations

import pytest

from judge_auditor.analysis.audit import audit
from judge_auditor.analysis.stats import CI
from judge_auditor.config import JudgeMode
from judge_auditor.report.diff import (
    _ci_metric,
    _disjoint,
    _flag_metric,
    _num_metric,
    diff_reports,
    render_diff_markdown,
    render_diff_terminal,
)
from judge_auditor.synthetic import (
    pairwise_judge_with_accuracy,
    pairwise_judge_with_flip_rate,
    pairwise_judge_with_rubric_sensitivity,
    scalar_judge,
    scalar_judge_with_rubric_sensitivity,
    scalar_judge_with_validity,
)


def _scalar_report(icc: float, *, seed: int = 0):
    js, exs = scalar_judge(icc=icc, n_examples=60, runs=12, seed=seed)
    return audit(js, exs)


def _pairwise_report(flip_rate: float, *, seed: int = 0):
    js, exs = pairwise_judge_with_flip_rate(flip_rate, n_examples=60, runs=12, seed=seed)
    return audit(js, exs)


# --- CI overlap helper ----------------------------------------------------------


def test_disjoint_true_when_intervals_separate():
    assert _disjoint(CI(0.2, 0.1, 0.3), CI(0.8, 0.7, 0.9)) is True


def test_disjoint_false_when_intervals_overlap():
    assert _disjoint(CI(0.4, 0.2, 0.6), CI(0.5, 0.3, 0.7)) is False


def test_disjoint_none_when_a_ci_missing():
    assert _disjoint(None, CI(0.5, 0.3, 0.7)) is None
    assert _disjoint(CI(0.5, 0.3, 0.7), None) is None


# --- metric builders ------------------------------------------------------------


def test_ci_metric_formats_signed_delta_and_changed():
    m = _ci_metric("x", CI(0.20, 0.10, 0.30), CI(0.80, 0.70, 0.90))
    assert m.before == "0.200" and m.after == "0.800"
    assert m.delta == "+0.600"
    assert m.changed is True


def test_ci_metric_handles_missing_side():
    m = _ci_metric("x", None, CI(0.80, 0.70, 0.90))
    assert m.before == "n/a" and m.after == "0.800"
    assert m.delta == "-" and m.changed is None


def test_num_metric_handles_infinite_value():
    m = _num_metric("noise", float("inf"), 0.25, ".3f")
    assert m.before == "n/a" and m.after == "0.250"
    assert m.delta == "-" and m.changed is None


def test_num_metric_signed_delta():
    m = _num_metric("noise", 0.50, 0.30, ".2f")
    assert m.delta == "-0.20" and m.changed is None


def test_flag_metric_changed_on_flip():
    assert _flag_metric("f", True, False).changed is True
    assert _flag_metric("f", True, True).changed is False


# --- diff_reports ---------------------------------------------------------------


def test_diff_rejects_mode_mismatch():
    scalar = _scalar_report(0.8)
    pairwise_like = _scalar_report(0.8)
    pairwise_like.mode = JudgeMode.PAIRWISE
    with pytest.raises(ValueError, match="cannot diff"):
        diff_reports(scalar, pairwise_like)


def test_diff_detects_consistency_improvement():
    before = _scalar_report(0.20, seed=1)
    after = _scalar_report(0.90, seed=2)
    d = diff_reports(before, after, baseline_label="v1", candidate_label="v2")
    assert d.baseline_label == "v1" and d.candidate_label == "v2"
    icc_row = next(m for m in d.metrics if m.name.startswith("Self-consistency"))
    assert float(icc_row.after) > float(icc_row.before)
    assert icc_row.changed is True  # well-separated ICCs -> disjoint CIs


def test_diff_verdict_change_tracked():
    before = _scalar_report(0.20)
    after = _scalar_report(0.90)
    d = diff_reports(before, after)
    assert d.verdict_before in ("LOW", "MODERATE")
    assert d.verdict_after == "HIGH"
    assert d.verdict_changed is True


def test_diff_no_verdict_change_for_similar_reports():
    d = diff_reports(_scalar_report(0.85, seed=3), _scalar_report(0.85, seed=4))
    assert d.verdict_changed is False


def test_diff_omits_validity_and_rubric_when_unavailable():
    d = diff_reports(_scalar_report(0.8), _scalar_report(0.8))
    names = {m.name for m in d.metrics}
    assert not any("Validity" in n for n in names)
    assert not any("Rubric" in n for n in names)


# --- rendering ------------------------------------------------------------------


def test_render_terminal_contains_header_and_metrics():
    d = diff_reports(
        _scalar_report(0.2), _scalar_report(0.9), baseline_label="A", candidate_label="B"
    )
    text = render_diff_terminal(d)
    assert "REPORT DIFF: A -> B" in text
    assert "Self-consistency ICC(1,1)" in text
    assert "CHANGED" in text


def test_render_markdown_is_a_table_with_labels():
    d = diff_reports(
        _scalar_report(0.2), _scalar_report(0.9), baseline_label="A", candidate_label="B"
    )
    md = render_diff_markdown(d)
    assert md.startswith("### Report diff: A -> B")
    assert "| Metric | A | B | Delta | Changed |" in md
    assert "| --- |" in md


def test_render_terminal_marks_verdict_change():
    d = diff_reports(_scalar_report(0.2), _scalar_report(0.9))
    assert "(changed)" in render_diff_terminal(d)


# --- pairwise mode --------------------------------------------------------------


def test_diff_pairwise_includes_position_flip_metric():
    d = diff_reports(_pairwise_report(0.0), _pairwise_report(1.0))
    names = {m.name for m in d.metrics}
    assert "Self-consistency kappa" in names
    assert "Position flip rate" in names
    assert "Noise floor (win margin)" in names
    flip = next(m for m in d.metrics if m.name == "Position flip rate")
    assert float(flip.after) > float(flip.before)


def test_diff_pairwise_renders_markdown():
    d = diff_reports(
        _pairwise_report(0.0), _pairwise_report(1.0), baseline_label="old", candidate_label="new"
    )
    md = render_diff_markdown(d)
    assert "| Metric | old | new | Delta | Changed |" in md
    assert "Position flip rate" in md


# --- validity / rubric rows (only present when available) ------------------------


def test_diff_scalar_includes_validity_row_when_available():
    j0, e0 = scalar_judge_with_validity(rho=0.0, n_examples=40, runs=10, seed=1)
    j1, e1 = scalar_judge_with_validity(rho=0.9, n_examples=40, runs=10, seed=2)
    d = diff_reports(audit(j0, e0), audit(j1, e1))
    assert any(m.name == "Validity Pearson r" for m in d.metrics)


def test_diff_scalar_includes_rubric_row_when_available():
    j0, e0 = scalar_judge_with_rubric_sensitivity(sensitivity=1.0, n_examples=40, runs=10, seed=1)
    j1, e1 = scalar_judge_with_rubric_sensitivity(sensitivity=0.0, n_examples=40, runs=10, seed=2)
    d = diff_reports(audit(j0, e0), audit(j1, e1))
    assert any(m.name == "Rubric ICC" for m in d.metrics)


def test_diff_pairwise_includes_validity_row_when_available():
    j0, e0 = pairwise_judge_with_accuracy(0.5, n_examples=40, runs=10, seed=1)
    j1, e1 = pairwise_judge_with_accuracy(0.95, n_examples=40, runs=10, seed=2)
    d = diff_reports(audit(j0, e0), audit(j1, e1))
    assert any(m.name == "Validity Cohen kappa" for m in d.metrics)


def test_diff_pairwise_includes_rubric_row_when_available():
    j0, e0 = pairwise_judge_with_rubric_sensitivity(
        flip_fraction=1.0, n_examples=40, runs=10, seed=1
    )
    j1, e1 = pairwise_judge_with_rubric_sensitivity(
        flip_fraction=0.0, n_examples=40, runs=10, seed=2
    )
    d = diff_reports(audit(j0, e0), audit(j1, e1))
    assert any(m.name == "Rubric kappa" for m in d.metrics)
