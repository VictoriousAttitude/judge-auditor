from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from judge_auditor.analysis.power_analysis import power_analysis
from judge_auditor.config import JudgeMode, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet


def scalar_set_with_noise(n_examples: int, sigma: float, runs: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_examples):
        base = float(rng.integers(2, 9))
        eid = f"ex{i}"
        for j in range(runs):
            s = base + rng.normal(0, sigma)
            records.append(JudgmentRecord(eid, j, 0, None, str(s), True, score=s))
    return JudgmentSet(JudgeMode.SCALAR, "m", records)


def pairwise_set(per_example: list[list[Winner]]) -> JudgmentSet:
    records = []
    for i, winners in enumerate(per_example):
        for j, w in enumerate(winners):
            ordering = "AB" if j % 2 == 0 else "BA"
            records.append(JudgmentRecord(f"ex{i}", j, 0, ordering, "x", True, winner=w))
    return JudgmentSet(JudgeMode.PAIRWISE, "m", records)


def test_scalar_mde_matches_analytical_formula():
    js = scalar_set_with_noise(200, sigma=1.0, runs=10, seed=1)
    res = power_analysis(js, alpha=0.05, power=0.80)
    # Recover sigma_w ~ 1.0 from the injected noise.
    assert abs(res.sigma_w - 1.0) < 0.1
    factor = norm.ppf(0.975) + norm.ppf(0.80)
    expected = factor * res.sigma_w * math.sqrt(2.0 / res.n_examples)
    assert abs(res.mde - expected) < 1e-9


def test_scalar_power_curve_monotonic_decreasing():
    js = scalar_set_with_noise(50, sigma=1.5, seed=2)
    res = power_analysis(js)
    mdes = [m for _, m in res.power_curve]
    assert mdes == sorted(mdes, reverse=True)  # larger n => smaller MDE
    assert all(b < a for a, b in zip(mdes, mdes[1:], strict=False))


def test_scalar_required_n_inverts_mde():
    js = scalar_set_with_noise(80, sigma=1.0, seed=3)
    target = 0.5
    res = power_analysis(js, target_effect=target)
    factor = norm.ppf(0.975) + norm.ppf(0.80)
    expected = math.ceil(2.0 * (factor * res.sigma_w / target) ** 2)
    assert res.required_n == expected
    # The MDE at the required n should be at or just under the target effect.
    mde_at_req = factor * res.sigma_w * math.sqrt(2.0 / res.required_n)
    assert mde_at_req <= target + 1e-9


def test_lower_noise_gives_smaller_mde():
    quiet = power_analysis(scalar_set_with_noise(100, sigma=0.5, seed=4))
    noisy = power_analysis(scalar_set_with_noise(100, sigma=2.0, seed=4))
    assert quiet.mde < noisy.mde


def test_pairwise_perfect_consistency_full_discriminability():
    # Judge always returns A => modal agreement 1.0 => discriminability 1.0.
    res = power_analysis(pairwise_set([[Winner.A] * 8 for _ in range(40)]))
    assert abs(res.effective_accuracy - 1.0) < 1e-9
    assert abs(res.discriminability - 1.0) < 1e-9
    assert res.mde_winrate is not None and res.mde_winrate > 0


def test_pairwise_coinflip_has_infinite_noise_floor():
    # Half A, half B per example => modal rate 0.5 => discriminability 0 => inf MDE.
    res = power_analysis(pairwise_set([[Winner.A] * 4 + [Winner.B] * 4 for _ in range(40)]))
    assert abs(res.effective_accuracy - 0.5) < 1e-9
    assert res.discriminability == 0.0
    assert math.isinf(res.mde_winrate)
    assert res.required_pairs is None  # cannot detect anything


def test_pairwise_winrate_curve_monotonic_decreasing():
    runs = [Winner.A] * 6 + [Winner.B] * 2  # modal rate 0.75
    res = power_analysis(pairwise_set([runs for _ in range(30)]))
    margins = [m for _, m in res.winrate_power_curve]
    assert margins == sorted(margins, reverse=True)


def test_higher_consistency_lowers_winrate_floor():
    high = power_analysis(pairwise_set([[Winner.A] * 7 + [Winner.B] for _ in range(30)]))
    low = power_analysis(pairwise_set([[Winner.A] * 5 + [Winner.B] * 3 for _ in range(30)]))
    assert high.mde_winrate < low.mde_winrate


def test_empty_scalar_set_is_safe():
    js = JudgmentSet(JudgeMode.SCALAR, "m", [])
    res = power_analysis(js)
    assert res.n_examples == 0
    assert res.mde is None


def test_pairwise_required_pairs_for_target_margin():
    # A discriminating judge (mostly-agreeing) with a target effect yields required_pairs.
    js = pairwise_set([[Winner.A] * 8 for _ in range(20)])
    res = power_analysis(js, target_effect=0.1)
    assert res.discriminability is not None and res.discriminability > 0
    assert res.required_pairs is not None and res.required_pairs >= 1


def test_pairwise_no_multi_run_examples_is_unavailable():
    # Every example has a single winner rating: effective accuracy is undefined.
    js = pairwise_set([[Winner.A], [Winner.B], [Winner.A]])
    res = power_analysis(js, target_effect=0.1)
    assert res.n_examples == 0
    assert res.effective_accuracy is not None and math.isnan(res.effective_accuracy)
    assert res.mde_winrate is None
    assert res.required_pairs is None
