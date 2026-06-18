"""Monte-Carlo calibration: the estimators are unbiased, their CIs cover, and the
detectors neither cry wolf nor sleep through a real bias.

The single-seed checks in ``test_validation.py`` prove the pipeline gets the right
answer *once*; a lucky seed could hide a small bias or an over-wide interval. Here
we average over many seeds and across a grid of true parameter values so the
guarantees are statistical, not anecdotal:

* **Calibration** — the mean recovered ICC / flip rate sits on the truth across a
  grid, with low seed-to-seed spread.
* **CI coverage** — a nominal 95% bootstrap interval contains the true ICC about
  95% of the time (percentile bootstrap typically under-covers slightly, so we
  assert a band around the nominal rate rather than an exact value).
* **Detector operating characteristics** — a clean judge is essentially never
  flagged (false-alarm control) and an injected bias is essentially always caught
  (sensitivity).

Seed counts and ``n_boot`` are kept modest so the file stays fast; the tolerances
were set from larger offline runs and carry margin.
"""

from __future__ import annotations

import numpy as np
import pytest

from judge_auditor import synthetic as S
from judge_auditor.analysis.audit import audit

# --- Calibration over a grid of true values -------------------------------------


@pytest.mark.parametrize("true_icc", [0.0, 0.3, 0.6, 0.9])
def test_icc_point_estimate_is_unbiased(true_icc: float):
    """Continuous (unquantized) scores recover the theoretical ICC on average."""
    points = [
        audit(*S.scalar_judge(icc=true_icc, n_examples=120, runs=20, seed=s), n_boot=100)
        .consistency.icc_oneway.point  # type: ignore[union-attr]
        for s in range(30)
    ]
    arr = np.asarray(points)
    assert abs(float(arr.mean()) - true_icc) < 0.03  # unbiased
    assert float(arr.std()) < 0.05  # tight across seeds


@pytest.mark.parametrize("true_flip", [0.0, 0.15, 0.5, 1.0])
def test_flip_rate_is_recovered_exactly(true_flip: float):
    """The deterministic majority-flip construction is recovered to rounding."""
    points = [
        audit(
            *S.pairwise_judge_with_flip_rate(true_flip, n_examples=200, runs=16, seed=s),
            n_boot=100,
        ).position.flip_rate.point  # type: ignore[union-attr]
        for s in range(20)
    ]
    assert abs(float(np.mean(points)) - true_flip) < 0.005


# --- Confidence-interval coverage -----------------------------------------------


def test_icc_confidence_interval_covers_near_nominal():
    """A 95% ICC interval should contain the truth ~95% of the time."""
    true_icc = 0.6
    covered = sum(
        ci.low <= true_icc <= ci.high
        for s in range(60)
        if (
            ci := audit(
                *S.scalar_judge(icc=true_icc, n_examples=120, runs=20, seed=s),
                n_boot=300,
            ).consistency.icc_oneway
        )
        is not None
    )
    coverage = covered / 60
    assert 0.85 <= coverage <= 1.0  # percentile bootstrap, slack around nominal 0.95


# --- Detector operating characteristics -----------------------------------------


def test_clean_judges_are_rarely_flagged():
    """False-alarm control: clean judges should almost never trip a flag."""
    scalar_alarms = 0
    pairwise_alarms = 0
    for s in range(30):
        rs = audit(
            *S.scalar_judge(icc=0.90, n_examples=120, runs=20, quantize=True, seed=s),
            n_boot=100,
        )
        if rs.overall != "HIGH" or rs.verbosity.flagged or rs.scale.compressed or rs.notes:
            scalar_alarms += 1
        rp = audit(*S.consistent_pairwise_judge(n_examples=120, runs=16, seed=s), n_boot=100)
        if rp.overall != "HIGH" or rp.position.favored_position != "none" or rp.notes:  # type: ignore[union-attr]
            pairwise_alarms += 1
    assert scalar_alarms / 30 <= 0.10
    assert pairwise_alarms / 30 <= 0.10


def test_injected_biases_are_almost_always_caught():
    """Sensitivity: a maximally biased / noisy judge is caught nearly every time."""
    position_hits = 0
    noisy_hits = 0
    for s in range(30):
        rb = audit(
            *S.pairwise_judge_with_first_rate(1.0, n_examples=120, runs=16, seed=s), n_boot=100
        )
        if rb.overall == "LOW" and rb.position.favored_position == "first":  # type: ignore[union-attr]
            position_hits += 1
        rn = audit(
            *S.scalar_judge(icc=0.20, n_examples=120, runs=20, quantize=True, seed=s),
            n_boot=100,
        )
        if rn.overall == "LOW":
            noisy_hits += 1
    assert position_hits / 30 >= 0.95
    assert noisy_hits / 30 >= 0.95
