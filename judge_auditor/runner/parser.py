"""Verdict extraction from free-form judge output.

Judges rarely emit perfectly structured output, so each parser tries an ordered
list of strategies from most to least reliable (structured JSON -> explicit
markers -> natural-language patterns -> last-resort heuristics). Parsing returns
``(value, error)``; a non-``None`` error means the response was unparseable and
should be recorded as a parse failure rather than silently dropped.
"""

from __future__ import annotations

import json
import re

from ..config import PairwiseChoice

# --- Pairwise -------------------------------------------------------------------

# Maps a textual label the judge might emit to a presented position. "A" / "1"
# / "first" all mean the first-presented response; "C" / "tie" mean a tie.
_FIRST_TOKENS = {"a", "1", "first", "one"}
_SECOND_TOKENS = {"b", "2", "second", "two"}
_TIE_TOKENS = {"c", "tie", "draw", "equal", "both", "neither", "same"}

_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)
# MT-Bench style explicit marker, e.g. "[[A]]", "[[B]]", "[[C]]".
_BRACKET_MARKER = re.compile(r"\[\[\s*([ABC])\s*\]\]", re.IGNORECASE)
# "Response A", "Assistant B", "Answer A", "Output 1", optionally "... is better".
_LABELLED = re.compile(
    r"\b(?:response|assistant|answer|output|model|option)\s*([AB12])\b",
    re.IGNORECASE,
)
_VERDICT_KEYS = ("winner", "verdict", "choice", "better", "preferred", "result")


def _token_to_choice(token: str) -> PairwiseChoice | None:
    t = token.strip().lower()
    if t in _FIRST_TOKENS:
        return PairwiseChoice.FIRST
    if t in _SECOND_TOKENS:
        return PairwiseChoice.SECOND
    if t in _TIE_TOKENS:
        return PairwiseChoice.TIE
    return None


def _from_json(text: str) -> PairwiseChoice | None:
    for match in _JSON_BLOCK.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):  # pragma: no cover - a {...} block always loads to a dict
            continue
        for key in _VERDICT_KEYS:
            if key in obj:
                choice = _token_to_choice(str(obj[key]))
                if choice is not None:
                    return choice
    return None


def parse_pairwise(text: str) -> tuple[PairwiseChoice | None, str | None]:
    """Extract which presented position won. Returns ``(choice, error)``."""
    if not text or not text.strip():
        return None, "empty response"

    # 1. Structured JSON anywhere in the output.
    choice = _from_json(text)
    if choice is not None:
        return choice, None

    # 2. Explicit MT-Bench bracket marker.
    marker = _BRACKET_MARKER.search(text)
    if marker:
        choice = _token_to_choice(marker.group(1))
        if choice is not None:
            return choice, None

    # 3. Tie keywords (checked before labelled mentions so "both are equal"
    #    doesn't get misread by an incidental "Response A" reference).
    lowered = text.lower()
    if re.search(r"\b(tie|draw|equally good|equally strong|same quality|no winner)\b", lowered):
        return PairwiseChoice.TIE, None

    # 4. Last decisive labelled mention ("Response A", "Assistant 2", ...).
    labelled = _LABELLED.findall(text)
    if labelled:
        choice = _token_to_choice(labelled[-1])
        if choice is not None:
            return choice, None

    return None, f"no verdict found in: {text[:120]!r}"


# --- Scalar ---------------------------------------------------------------------

_SCORE_KEYS = ("score", "rating", "grade", "value", "points")
# "7/10" or "7 out of 10": capture the numerator.
_FRACTION = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*\d+(?:\.\d+)?", re.IGNORECASE)
# "Score: 7", "Rating - 8.5", "score=9".
_LABELLED_SCORE = re.compile(
    r"(?:score|rating|grade|points?)\s*(?:is|:|=|-)?\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_ANY_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _score_from_json(text: str) -> float | None:
    for match in _JSON_BLOCK.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):  # pragma: no cover - a {...} block always loads to a dict
            continue
        for key in _SCORE_KEYS:
            if key in obj:
                try:
                    return float(obj[key])
                except (TypeError, ValueError):
                    continue
    return None


def parse_scalar(
    text: str, score_min: float, score_max: float
) -> tuple[float | None, str | None]:
    """Extract a numeric score within ``[score_min, score_max]``.

    A number found outside the configured range is treated as a parse failure
    (the judge ignored the scale), not silently clamped.
    """
    if not text or not text.strip():
        return None, "empty response"

    candidates: list[float] = []

    js = _score_from_json(text)
    if js is not None:
        candidates.append(js)

    frac = _FRACTION.search(text)
    if frac:
        candidates.append(float(frac.group(1)))

    labelled = _LABELLED_SCORE.search(text)
    if labelled:
        candidates.append(float(labelled.group(1)))

    if not candidates:
        # Last resort: a lone number in a short response is likely the score.
        nums = _ANY_NUMBER.findall(text)
        if len(nums) == 1:
            candidates.append(float(nums[0]))

    if not candidates:
        return None, f"no score found in: {text[:120]!r}"

    value = candidates[0]
    if not (score_min <= value <= score_max):
        return None, f"score {value} outside range [{score_min}, {score_max}]"
    return value, None
