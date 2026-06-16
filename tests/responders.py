"""Mock-judge "model organisms" for tests and validation.

Each responder is a ``(messages, config) -> str`` function suitable for
:class:`judge_auditor.runner.backends.mock.MockBackend`. They simulate judges
with known, controllable behavior so we can assert the runner (and later the
analysis) recovers what we injected.
"""

from __future__ import annotations

from judge_auditor.config import JudgeConfig

# Test prompt templates whose rendered text these responders can introspect.
PAIRWISE_TEMPLATE = (
    "Question: {prompt}\nResponse A: {response_a}\nResponse B: {response_b}\n"
    "Which response is better?"
)
SCALAR_TEMPLATE = "Question: {prompt}\nResponse: {response}\nScore it 1-10."

# A "good" response is marked with this token; responders prefer/score by count.
GOOD = "GOOD"


def _sections(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Split a rendered pairwise prompt into its A-section and B-section text."""
    text = messages[-1]["content"]
    b = text.index("Response B:")
    return text[text.index("Response A:") : b], text[b:]


def content_pref_pairwise(messages: list[dict[str, str]], config: JudgeConfig) -> str:
    """Consistent, content-driven judge: prefers the section with more GOOD tokens.

    Because it judges *content* (not position), its canonical verdict is stable
    under position swapping — the ideal "null" pairwise judge.
    """
    sec_a, sec_b = _sections(messages)
    a, b = sec_a.count(GOOD), sec_b.count(GOOD)
    if a > b:
        return "Response A is clearly better. [[A]]"
    if b > a:
        return "Response B is clearly better. [[B]]"
    return "They are equally good. [[C]]"


def always_first_pairwise(messages: list[dict[str, str]], config: JudgeConfig) -> str:
    """Maximally position-biased judge: always picks the first-presented response."""
    return "[[A]]"


def malformed(messages: list[dict[str, str]], config: JudgeConfig) -> str:
    """Judge that never emits a parseable verdict."""
    return "I have some thoughts but cannot decide."


def content_score_scalar(messages: list[dict[str, str]], config: JudgeConfig) -> str:
    """Deterministic scalar judge: score scales with GOOD-token count."""
    text = messages[-1]["content"]
    score = min(config.score_max, config.score_min + text.count(GOOD) * 3)
    return f'{{"score": {score}}}'
