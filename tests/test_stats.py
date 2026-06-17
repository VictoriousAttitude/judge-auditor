from __future__ import annotations

import numpy as np

from judge_auditor.analysis.stats import (
    bootstrap_ci,
    interpret_icc,
    interpret_kappa,
    wilson_ci,
)


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def test_bootstrap_point_is_full_sample_statistic():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = bootstrap_ci(data, _mean, n_boot=500, seed=0)
    assert ci.point == 3.0
    assert ci.low < ci.point < ci.high


def test_bootstrap_is_deterministic_with_seed():
    rng = np.random.default_rng(42)
    data = list(rng.normal(0, 1, 200))
    a = bootstrap_ci(data, _mean, n_boot=400, seed=7)
    b = bootstrap_ci(data, _mean, n_boot=400, seed=7)
    assert (a.point, a.low, a.high) == (b.point, b.low, b.high)


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    data = list(rng.normal(10.0, 2.0, 300))
    ci = bootstrap_ci(data, _mean, n_boot=1000, seed=3)
    assert ci.low <= ci.point <= ci.high


def test_bootstrap_ci_width_shrinks_with_n():
    rng = np.random.default_rng(2)
    small = bootstrap_ci(list(rng.normal(0, 1, 30)), _mean, n_boot=500, seed=5)
    large = bootstrap_ci(list(rng.normal(0, 1, 3000)), _mean, n_boot=500, seed=5)
    assert large.width < small.width


def test_bootstrap_ci_coverage_is_near_nominal():
    # Calibration: across many datasets, the 95% CI for the mean should contain
    # the true mean ~95% of the time (the bootstrap's defining property).
    master = np.random.default_rng(123)
    trials, hits = 200, 0
    for _ in range(trials):
        data = list(master.normal(10.0, 2.0, 80))
        ci = bootstrap_ci(data, _mean, n_boot=300, seed=0)
        if ci.low < 10.0 < ci.high:
            hits += 1
    coverage = hits / trials
    # Generous band: nominal 0.95, ~5 SD wide, so false failures are negligible.
    assert 0.86 < coverage < 1.0


def test_bootstrap_empty():
    ci = bootstrap_ci([], _mean, n_boot=100)
    assert np.isnan(ci.low) and np.isnan(ci.high)


def test_wilson_ci_symmetric_at_half():
    ci = wilson_ci(50, 100)
    assert ci.point == 0.5
    assert abs((ci.low + ci.high) / 2 - 0.5) < 1e-9
    # Known Wilson interval for 50/100 at 95% is roughly [0.404, 0.596].
    assert abs(ci.low - 0.404) < 0.01
    assert abs(ci.high - 0.596) < 0.01


def test_wilson_ci_bounded_near_one():
    ci = wilson_ci(100, 100)
    assert ci.point == 1.0
    assert ci.high <= 1.0 + 1e-9
    assert ci.low < 1.0


def test_interpret_kappa_bands():
    assert interpret_kappa(-0.1) == "poor"
    assert interpret_kappa(0.1) == "slight"
    assert interpret_kappa(0.3) == "fair"
    assert interpret_kappa(0.5) == "moderate"
    assert interpret_kappa(0.7) == "substantial"
    assert interpret_kappa(0.9) == "almost perfect"
    assert interpret_kappa(1.0) == "almost perfect"


def test_interpret_icc_bands():
    assert interpret_icc(0.3) == "poor"
    assert interpret_icc(0.6) == "moderate"
    assert interpret_icc(0.8) == "good"
    assert interpret_icc(0.95) == "excellent"
