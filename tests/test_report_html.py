from __future__ import annotations

import numpy as np

from judge_auditor.analysis.audit import audit
from judge_auditor.analysis.stats import CI
from judge_auditor.config import EvalExample, JudgeMode
from judge_auditor.records import JudgmentRecord, JudgmentSet
from judge_auditor.report.html import _ci, _pct, render_html

from .test_audit import position_biased_pairwise, reliable_scalar


def test_html_scalar_is_self_contained_document():
    js, examples = reliable_scalar()
    html = render_html(audit(js, examples))
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html.strip().splitlines()[-1] or html.rstrip().endswith("</html>")
    # No external resources => truly self-contained.
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html


def test_html_scalar_contains_verdict_and_sections():
    js, examples = reliable_scalar()
    html = render_html(audit(js, examples))
    assert "HIGH" in html
    assert "judge-x" in html
    assert "SELF-CONSISTENCY" in html
    assert "SCALE ANALYSIS" in html
    assert "POWER / NOISE FLOOR" in html
    assert "RECOMMENDED ACTIONS" in html
    # Histogram bars rendered for scalar mode.
    assert 'class="bar"' in html


def test_html_pairwise_has_position_card_and_recommendations():
    js, examples = position_biased_pairwise()
    html = render_html(audit(js, examples))
    assert "POSITION BIAS" in html
    assert "headline low" in html  # verdict colour class
    assert "Position bias" in html  # recommendation text


def test_html_escapes_model_name():
    js, examples = reliable_scalar()
    js.model = "<script>evil</script>"
    html = render_html(audit(js, examples))
    assert "<script>evil" not in html
    assert "&lt;script&gt;evil" in html


def test_pct_and_ci_format_missing_values_as_na():
    assert _pct(None) == "n/a"
    assert _pct(float("nan")) == "n/a"
    assert _ci(None) == "n/a"
    assert _ci(CI(float("nan"), float("nan"), float("nan"))) == "n/a"


def test_html_renders_within_quality_length_effect_row():
    rng = np.random.default_rng(0)
    examples, records = [], []
    idx = 0
    for q in (1.0, 2.0):
        for k in range(10):
            eid = f"ex{idx}"
            idx += 1
            base = q * 0.4 + k * 0.8 + 0.5
            examples.append(
                EvalExample(
                    id=eid, prompt="q", response_a=" ".join(["w"] * (5 + k * 10)), quality_label=q
                )
            )
            for j in range(8):
                s = base + rng.normal(0, 0.1)
                records.append(JudgmentRecord(eid, j, 0, None, str(s), True, score=s))
    html = render_html(audit(JudgmentSet(JudgeMode.SCALAR, "judge-x", records), examples))
    assert "Within-quality length effect" in html
