from __future__ import annotations

import pytest

from judge_auditor.config import EvalExample, JudgeConfig, JudgeMode

from .responders import GOOD, PAIRWISE_TEMPLATE, SCALAR_TEMPLATE


@pytest.fixture
def pairwise_config() -> JudgeConfig:
    return JudgeConfig(
        model="mock-judge",
        prompt_template=PAIRWISE_TEMPLATE,
        mode=JudgeMode.PAIRWISE,
    )


@pytest.fixture
def scalar_config() -> JudgeConfig:
    return JudgeConfig(
        model="mock-judge",
        prompt_template=SCALAR_TEMPLATE,
        mode=JudgeMode.SCALAR,
    )


@pytest.fixture
def pairwise_examples() -> list[EvalExample]:
    # response_a is always the "good" one, so the canonical winner is A.
    return [
        EvalExample(id=f"ex{i}", prompt=f"q{i}", response_a=GOOD, response_b="bad")
        for i in range(4)
    ]


@pytest.fixture
def scalar_examples() -> list[EvalExample]:
    return [
        EvalExample(id=f"ex{i}", prompt=f"q{i}", response_a=GOOD * (i % 3))
        for i in range(4)
    ]
