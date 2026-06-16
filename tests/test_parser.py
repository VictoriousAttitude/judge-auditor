from __future__ import annotations

import pytest

from judge_auditor.config import PairwiseChoice
from judge_auditor.runner.parser import parse_pairwise, parse_scalar


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"winner": "A"}', PairwiseChoice.FIRST),
        ('here is my verdict {"verdict": "B"} done', PairwiseChoice.SECOND),
        ("[[A]]", PairwiseChoice.FIRST),
        ("The better one is [[B]].", PairwiseChoice.SECOND),
        ("[[C]]", PairwiseChoice.TIE),
        ("Response A is better.", PairwiseChoice.FIRST),
        ("I think Assistant 2 wins this round.", PairwiseChoice.SECOND),
        ("Both responses are equally good.", PairwiseChoice.TIE),
        ("It's a tie.", PairwiseChoice.TIE),
    ],
)
def test_parse_pairwise_formats(text: str, expected: PairwiseChoice) -> None:
    choice, err = parse_pairwise(text)
    assert err is None
    assert choice is expected


def test_parse_pairwise_uses_last_decisive_mention() -> None:
    # Reasoning mentions A then concludes B.
    text = "Response A starts strong, but ultimately Response B is more complete."
    choice, err = parse_pairwise(text)
    assert err is None
    assert choice is PairwiseChoice.SECOND


@pytest.mark.parametrize("text", ["", "   ", "I cannot tell which is better."])
def test_parse_pairwise_failures(text: str) -> None:
    choice, err = parse_pairwise(text)
    assert choice is None
    assert err is not None


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"score": 7}', 7.0),
        ('{"rating": 8.5}', 8.5),
        ("Score: 6", 6.0),
        ("Rating - 9", 9.0),
        ("I'd give it 7/10.", 7.0),
        ("That is 4 out of 10.", 4.0),
        ("8", 8.0),
    ],
)
def test_parse_scalar_formats(text: str, expected: float) -> None:
    score, err = parse_scalar(text, 1.0, 10.0)
    assert err is None
    assert score == expected


def test_parse_scalar_out_of_range_is_failure() -> None:
    score, err = parse_scalar("Score: 42", 1.0, 10.0)
    assert score is None
    assert err is not None and "outside range" in err


@pytest.mark.parametrize("text", ["", "no number here", "great answer overall"])
def test_parse_scalar_failures(text: str) -> None:
    score, err = parse_scalar(text, 1.0, 10.0)
    assert score is None
    assert err is not None
