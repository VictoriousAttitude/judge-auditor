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


# --- Adversarial pairwise corpus: pin down the documented strategy precedence ---


@pytest.mark.parametrize(
    "text,expected",
    [
        # Structured JSON outranks a conflicting bracket marker.
        ('{"winner": "B"} but visually [[A]] looked nice', PairwiseChoice.SECOND),
        # First JSON block lacks a verdict key; the second one supplies it.
        ('{"note": "thinking"} ... {"verdict": "B"}', PairwiseChoice.SECOND),
        # Bracket marker is case-insensitive and tolerates inner whitespace.
        ("My call: [[ b ]]", PairwiseChoice.SECOND),
        # A tie phrase outranks incidental labelled mentions of A and B.
        ("Response A and Response B are equally good here.", PairwiseChoice.TIE),
        # Lowercase / word-form verdict values still map.
        ('{"choice": "tie"}', PairwiseChoice.TIE),
        ('{"preferred": "first"}', PairwiseChoice.FIRST),
        # Numeric position labels resolve to presented position.
        ("Ultimately Output 1 is the stronger answer.", PairwiseChoice.FIRST),
    ],
)
def test_parse_pairwise_adversarial(text: str, expected: PairwiseChoice) -> None:
    choice, err = parse_pairwise(text)
    assert err is None
    assert choice is expected


@pytest.mark.parametrize(
    "text",
    [
        '{"winner": "Z"}',  # JSON present but the value is not a recognized verdict
        "It depends entirely on what you value in a response.",  # hedge, no verdict
    ],
)
def test_parse_pairwise_adversarial_failures(text: str) -> None:
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


# --- Adversarial scalar corpus --------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # JSON score outranks a later fraction/number.
        ('{"score": 3}. Honestly closer to 9/10 in spirit.', 3.0),
        # Fraction outranks a stray trailing number.
        ("I rate this 7/10, far above the 2 alternatives.", 7.0),
        # Decimal fraction and "out of" phrasing.
        ("That is 8.5 out of 10.", 8.5),
        # Labelled equals-sign form.
        ("score=9", 9.0),
        # Inclusive boundaries are valid.
        ("Score: 10", 10.0),
        ("Score: 1", 1.0),
    ],
)
def test_parse_scalar_adversarial(text: str, expected: float) -> None:
    score, err = parse_scalar(text, 1.0, 10.0)
    assert err is None
    assert score == expected


@pytest.mark.parametrize(
    "text",
    [
        "I give one a 3 and the other a 4.",  # two bare numbers: ambiguous, no pick
        "Score: -1",  # below the configured floor
        "Rating: 11",  # above the configured ceiling
    ],
)
def test_parse_scalar_adversarial_failures(text: str) -> None:
    score, err = parse_scalar(text, 1.0, 10.0)
    assert score is None
    assert err is not None
