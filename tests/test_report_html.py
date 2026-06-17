from __future__ import annotations

from judge_auditor.analysis.audit import audit
from judge_auditor.report.html import render_html

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
