"""Calibration and integration tests for the sycophancy / anchoring probe detectors.

The synthetic probe judges inject an exactly-known swing between the up and down
probe directions, so the detector must recover that swing (scalar: fraction of the
score range; pairwise: win-rate-for-A swing) and the audit must downgrade on it —
while a probe-free audit stays silent.
"""

from __future__ import annotations

import pytest

from judge_auditor.analysis.audit import audit
from judge_auditor.analysis.probe_bias import probe_bias
from judge_auditor.config import JudgeMode, Probe
from judge_auditor.report.recommendations import recommendations
from judge_auditor.runner.probes import probe_prefix
from judge_auditor.synthetic import (
    pairwise_judge_with_sycophancy,
    scalar_judge,
    scalar_judge_with_anchoring,
    scalar_judge_with_sycophancy,
)


def test_probe_prefix_neutral_is_empty() -> None:
    out = probe_prefix(Probe.NEUTRAL, JudgeMode.SCALAR, ordering=None, score_min=1, score_max=10)
    assert out == ""


@pytest.mark.parametrize(
    "probe,ordering,label",
    [
        (Probe.SYCOPHANCY_UP, "AB", "A"),  # up targets content A, presented first
        (Probe.SYCOPHANCY_UP, "BA", "B"),  # content A now in the second slot
        (Probe.SYCOPHANCY_DOWN, "AB", "B"),  # down targets content B
        (Probe.SYCOPHANCY_DOWN, "BA", "A"),
    ],
)
def test_pairwise_sycophancy_label_follows_content(probe, ordering, label) -> None:
    out = probe_prefix(probe, JudgeMode.PAIRWISE, ordering=ordering, score_min=1, score_max=10)
    assert f"Response {label}" in out


def test_pairwise_probe_requires_ordering() -> None:
    with pytest.raises(ValueError, match="needs an ordering"):
        probe_prefix(
            Probe.SYCOPHANCY_UP, JudgeMode.PAIRWISE, ordering=None, score_min=1, score_max=10
        )


def test_anchoring_not_applicable_to_pairwise() -> None:
    with pytest.raises(ValueError, match="not applicable to pairwise"):
        probe_prefix(Probe.ANCHOR_UP, JudgeMode.PAIRWISE, ordering="AB", score_min=1, score_max=10)


def test_scalar_sycophancy_recovers_injected_swing() -> None:
    js, _ = scalar_judge_with_sycophancy(strength=0.30, seed=1)
    res = probe_bias(js, score_min=1.0, score_max=10.0, seed=1)
    assert res.available
    syc = res.sycophancy
    assert syc is not None
    assert syc.flagged
    assert abs(syc.effect.point - 0.30) < 0.03
    # raw_pts is the swing in score units: 0.30 of a 9-point range ~ 2.7 pts.
    assert syc.raw_pts is not None
    assert abs(syc.raw_pts - 2.7) < 0.3


def test_scalar_anchoring_recovers_injected_swing() -> None:
    js, _ = scalar_judge_with_anchoring(strength=0.20, seed=2)
    res = probe_bias(js, score_min=1.0, score_max=10.0, seed=2)
    anc = res.anchoring
    assert anc is not None
    assert anc.kind == "anchoring"
    assert anc.flagged
    assert abs(anc.effect.point - 0.20) < 0.03


def test_pairwise_sycophancy_recovers_winrate_swing() -> None:
    js, _ = pairwise_judge_with_sycophancy(strength=0.30, seed=3)
    res = probe_bias(js, seed=3)
    syc = res.sycophancy
    assert syc is not None
    assert syc.mode is JudgeMode.PAIRWISE
    assert syc.raw_pts is None
    assert syc.flagged
    assert abs(syc.effect.point - 0.30) < 0.04
    # Anchoring is scalar-only.
    assert res.anchoring is None


def test_zero_strength_is_not_flagged() -> None:
    js, _ = scalar_judge_with_sycophancy(strength=0.0, seed=4)
    res = probe_bias(js, score_min=1.0, score_max=10.0, seed=4)
    syc = res.sycophancy
    assert syc is not None
    assert not syc.flagged
    assert abs(syc.effect.point) < 0.03


def test_probe_bias_unavailable_without_probes() -> None:
    js, _ = scalar_judge(icc=0.8, seed=5)
    res = probe_bias(js, score_min=1.0, score_max=10.0, seed=5)
    assert not res.available
    assert res.sycophancy is None
    assert res.anchoring is None
    assert res.effects == []


def test_small_sample_below_min_n_not_flagged() -> None:
    js, _ = scalar_judge_with_sycophancy(strength=0.40, n_examples=5, seed=6)
    res = probe_bias(js, score_min=1.0, score_max=10.0, seed=6)
    syc = res.sycophancy
    assert syc is not None
    assert syc.n_examples == 5
    assert not syc.flagged  # large effect, but n < min_n
    assert not syc.severe or not syc.flagged


def test_audit_downgrades_on_severe_sycophancy() -> None:
    js, examples = scalar_judge_with_sycophancy(strength=0.40, seed=7)
    report = audit(js, examples)
    # Baseline self-consistency is high, so only the probe can move the verdict.
    assert report.consistency.icc_oneway is not None
    assert report.consistency.icc_oneway.point > 0.7
    assert report.probe.flagged
    assert report.probe.severe
    assert report.overall == "LOW"
    assert any("ycophancy bias" in n for n in report.notes)


def test_audit_moderate_downgrade_on_mild_sycophancy() -> None:
    js, examples = scalar_judge_with_sycophancy(strength=0.10, seed=8)
    report = audit(js, examples)
    syc = report.probe.sycophancy
    assert syc is not None and syc.flagged and not syc.severe
    assert report.overall == "MODERATE"


def test_scalar_sycophancy_recommendation() -> None:
    js, examples = scalar_judge_with_sycophancy(strength=0.30, seed=10)
    recs = recommendations(audit(js, examples))
    assert any("Sycophancy" in r and "ignore stated preferences" in r for r in recs)


def test_scalar_anchoring_recommendation() -> None:
    js, examples = scalar_judge_with_anchoring(strength=0.30, seed=11)
    recs = recommendations(audit(js, examples))
    assert any("Anchoring" in r and "reference score" in r for r in recs)


def test_pairwise_sycophancy_recommendation() -> None:
    js, examples = pairwise_judge_with_sycophancy(strength=0.40, seed=12)
    recs = recommendations(audit(js, examples))
    assert any("Sycophancy" in r and "win rate" in r for r in recs)


def test_clean_judge_yields_no_probe_recommendation() -> None:
    js, examples = scalar_judge_with_sycophancy(strength=0.0, seed=13)
    recs = recommendations(audit(js, examples))
    assert not any("Sycophancy" in r or "Anchoring" in r for r in recs)


def test_audit_headline_excludes_probe_records() -> None:
    js, examples = scalar_judge_with_sycophancy(strength=0.30, seed=9)
    report = audit(js, examples)
    # Headline self-consistency is computed on the NEUTRAL subset only, so the
    # injected probe runs do not corrupt the ICC sample.
    assert report.consistency.n_examples == 120
