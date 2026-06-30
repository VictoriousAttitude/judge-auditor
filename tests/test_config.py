from __future__ import annotations

import pytest

from judge_auditor.config import AuditConfig, JudgeConfig, JudgeMode


def test_scalar_score_bounds_must_be_ordered():
    with pytest.raises(ValueError, match="score_min must be < score_max"):
        JudgeConfig(
            model="m",
            prompt_template="{prompt} {response}",
            mode=JudgeMode.SCALAR,
            score_min=10.0,
            score_max=1.0,
        )


def test_audit_config_runs_must_be_positive():
    with pytest.raises(ValueError, match="runs_per_example must be >= 1"):
        AuditConfig(runs_per_example=0)


def test_audit_config_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
        AuditConfig(max_concurrency=0)
