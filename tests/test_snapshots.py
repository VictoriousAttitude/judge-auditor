"""Golden-snapshot tests for the three report renderers.

The renderers are pure and deterministic (no clock, no randomness, no numpy in
the formatting path), so their output should be byte-for-byte stable. These tests
pin that output against committed snapshots: any change to the terminal, HTML, or
JSON layout shows up as an explicit, reviewable diff rather than slipping through.

To stay robust across numpy / Python / SciPy versions, the fixtures are *fixed*
``ReliabilityReport`` objects built from literal values (no bootstrap, no audit
run). That isolates the test from estimator-level numerical drift: it checks
rendering, which the analysis tests do not.

Regenerate the snapshots after an intentional layout change with::

    python -m tests.test_snapshots --update
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from judge_auditor.analysis.audit import ReliabilityReport
from judge_auditor.analysis.consistency import ConsistencyResult
from judge_auditor.analysis.position_bias import PositionBiasResult
from judge_auditor.analysis.power_analysis import PowerAnalysisResult
from judge_auditor.analysis.scale_analysis import ScaleAnalysisResult
from judge_auditor.analysis.stats import CI
from judge_auditor.analysis.verbosity_bias import VerbosityBiasResult
from judge_auditor.config import JudgeMode
from judge_auditor.report.html import render_html
from judge_auditor.report.json_report import render_json
from judge_auditor.report.terminal import render_terminal

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def scalar_report() -> ReliabilityReport:
    """A clean, highly-consistent scalar judge: HIGH verdict, no flags."""
    return ReliabilityReport(
        model="demo-scalar",
        mode=JudgeMode.SCALAR,
        n_examples=40,
        n_records=800,
        parse_failure_rate=0.0,
        consistency=ConsistencyResult(
            mode=JudgeMode.SCALAR,
            n_examples=40,
            runs_per_example=20,
            icc_oneway=CI(0.820, 0.740, 0.880),
            icc_twoway=CI(0.810, 0.730, 0.870),
            icc_interpretation="good",
            mean_within_sd=0.650,
        ),
        verbosity=VerbosityBiasResult(
            mode=JudgeMode.SCALAR,
            n_examples=40,
            threshold=0.3,
            flagged=False,
            spearman_rho=0.050,
            spearman_p=0.620,
            partial_rho=0.020,
        ),
        scale=ScaleAnalysisResult(
            mode=JudgeMode.SCALAR,
            n=800,
            score_min=1.0,
            score_max=10.0,
            num_bins=10,
            bin_values=[float(i) for i in range(1, 11)],
            histogram=[5, 10, 20, 40, 80, 140, 200, 160, 100, 45],
            distinct_values_used=10,
            effective_range=0.860,
            max_window_fraction=0.420,
            compressed=False,
            compressed_values=[],
        ),
        power=PowerAnalysisResult(
            mode=JudgeMode.SCALAR,
            n_examples=40,
            alpha=0.05,
            power=0.80,
            target_effect=None,
            sigma_w=0.650,
            mde=0.440,
            power_curve=[(10, 0.88), (20, 0.62), (30, 0.51), (50, 0.39), (100, 0.28)],
            required_n=None,
        ),
        position=None,
        overall="HIGH",
        notes=[],
    )


def pairwise_report() -> ReliabilityReport:
    """A moderately-consistent pairwise judge with a mild first-position bias."""
    return ReliabilityReport(
        model="demo-pairwise",
        mode=JudgeMode.PAIRWISE,
        n_examples=40,
        n_records=640,
        parse_failure_rate=0.0,
        consistency=ConsistencyResult(
            mode=JudgeMode.PAIRWISE,
            n_examples=40,
            runs_per_example=16,
            fleiss_kappa=CI(0.550, 0.450, 0.640),
            kappa_interpretation="moderate",
            mean_agreement=0.780,
            min_agreement=0.500,
            median_agreement=0.810,
        ),
        verbosity=VerbosityBiasResult(
            mode=JudgeMode.PAIRWISE,
            n_examples=40,
            threshold=0.3,
            flagged=False,
            longer_response_win_rate=0.520,
            length_winrate_rho=0.100,
            length_winrate_p=0.400,
        ),
        scale=ScaleAnalysisResult(
            mode=JudgeMode.PAIRWISE,
            n=640,
            n_decisions=600,
            win_rate_a=0.500,
            win_rate_b=0.440,
            tie_rate=0.060,
            indistinguishable=False,
        ),
        power=PowerAnalysisResult(
            mode=JudgeMode.PAIRWISE,
            n_examples=40,
            alpha=0.05,
            power=0.80,
            target_effect=None,
            effective_accuracy=0.780,
            discriminability=0.560,
            mde_winrate=0.180,
            winrate_power_curve=[(10, 0.40), (20, 0.28), (50, 0.18), (100, 0.13)],
            required_pairs=None,
        ),
        position=PositionBiasResult(
            n_decisions=600,
            first_position_rate=CI(0.570, 0.530, 0.610),
            first_preference_p_value=0.001,
            favored_position="first",
            n_examples=40,
            n_flipped=6,
            flip_rate=CI(0.150, 0.070, 0.280),
            flipped_examples=["ex3", "ex7", "ex11", "ex18", "ex25", "ex33"],
            tie_rate=0.060,
        ),
        overall="MODERATE",
        notes=["Position bias toward the first-presented response (flip rate 15%)."],
    )


_RENDERERS = {"terminal": render_terminal, "json": render_json, "html": render_html}
_EXT = {"terminal": "txt", "json": "json", "html": "html"}
_FIXTURES = {"scalar": scalar_report, "pairwise": pairwise_report}

_CASES = [
    (fixture, fmt, f"{fixture}.{_EXT[fmt]}")
    for fixture in _FIXTURES
    for fmt in _RENDERERS
]


@pytest.mark.parametrize("fixture,fmt,filename", _CASES)
def test_render_matches_snapshot(fixture: str, fmt: str, filename: str) -> None:
    rendered = _RENDERERS[fmt](_FIXTURES[fixture]())
    expected = (SNAPSHOT_DIR / filename).read_text(encoding="utf-8")
    assert rendered == expected


def _update_snapshots() -> None:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    for fixture, fmt, filename in _CASES:
        rendered = _RENDERERS[fmt](_FIXTURES[fixture]())
        (SNAPSHOT_DIR / filename).write_text(rendered, encoding="utf-8")
        print(f"wrote {filename}")


if __name__ == "__main__":
    if "--update" in sys.argv:
        _update_snapshots()
    else:
        print("pass --update to regenerate snapshots")
