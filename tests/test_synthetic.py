from __future__ import annotations

import pytest

from judge_auditor import synthetic as S
from judge_auditor.config import JudgeMode, PairwiseChoice, Winner
from judge_auditor.synthetic import _choice_for, _winner_for

# --- tie round-trip in the position mapping helpers -----------------------------


@pytest.mark.parametrize("ordering", ["AB", "BA"])
def test_winner_and_choice_helpers_round_trip_tie(ordering: str) -> None:
    assert _winner_for(PairwiseChoice.TIE, ordering) is Winner.TIE
    assert _choice_for(Winner.TIE, ordering) is PairwiseChoice.TIE


# --- validation guards reject out-of-range inputs -------------------------------


def test_scalar_judge_rejects_icc_out_of_range() -> None:
    with pytest.raises(ValueError, match="icc must be in"):
        S.scalar_judge(icc=1.0)


def test_scalar_validity_rejects_rho_out_of_range() -> None:
    with pytest.raises(ValueError, match="rho must be in"):
        S.scalar_judge_with_validity(rho=2.0)


def test_pairwise_accuracy_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="accuracy must be in"):
        S.pairwise_judge_with_accuracy(2.0)


def test_scalar_rubric_sensitivity_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="sensitivity must be in"):
        S.scalar_judge_with_rubric_sensitivity(sensitivity=2.0)


def test_scalar_rubric_sensitivity_rejects_too_few_variants() -> None:
    with pytest.raises(ValueError, match="at least 2 rubric variants"):
        S.scalar_judge_with_rubric_sensitivity(sensitivity=0.5, n_variants=1)


def test_pairwise_rubric_sensitivity_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="flip_fraction must be in"):
        S.pairwise_judge_with_rubric_sensitivity(flip_fraction=2.0)


def test_pairwise_rubric_sensitivity_rejects_too_few_variants() -> None:
    with pytest.raises(ValueError, match="at least 2 rubric variants"):
        S.pairwise_judge_with_rubric_sensitivity(flip_fraction=0.5, n_variants=1)


def test_scalar_sycophancy_rejects_strength_out_of_range() -> None:
    with pytest.raises(ValueError, match="strength must be in"):
        S.scalar_judge_with_sycophancy(strength=2.0)


def test_scalar_anchoring_rejects_strength_out_of_range() -> None:
    with pytest.raises(ValueError, match="strength must be in"):
        S.scalar_judge_with_anchoring(strength=-0.1)


def test_pairwise_sycophancy_rejects_strength_out_of_range() -> None:
    with pytest.raises(ValueError, match="strength must be in"):
        S.pairwise_judge_with_sycophancy(strength=2.0)


def test_pairwise_flip_rate_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="flip_rate must be in"):
        S.pairwise_judge_with_flip_rate(2.0)


def test_pairwise_first_rate_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="must be valid probabilities"):
        S.pairwise_judge_with_first_rate(0.8, tie_rate=0.5)


# --- exercised generator bodies / branches --------------------------------------


def test_scalar_validity_quantize_yields_integer_bounded_scores() -> None:
    js, examples = S.scalar_judge_with_validity(
        rho=0.5, n_examples=6, runs=4, quantize=True, seed=1
    )
    assert len(examples) == 6
    for r in js.records:
        assert r.score == float(int(r.score))
        assert 1.0 <= r.score <= 10.0


def test_first_rate_judge_emits_ties_when_tie_rate_set() -> None:
    js, _ = S.pairwise_judge_with_first_rate(0.4, tie_rate=0.4, n_examples=20, runs=8, seed=2)
    choices = {r.choice for r in js.records}
    assert PairwiseChoice.TIE in choices  # the tie branch fired
    assert Winner.TIE in {r.winner for r in js.records}


def test_compressed_scalar_judge_uses_only_its_levels() -> None:
    js, examples = S.compressed_scalar_judge(
        levels=(5, 6), n_examples=8, runs=4, sigma_w=0.0, seed=0
    )
    assert js.mode is JudgeMode.SCALAR
    assert {r.score for r in js.records} == {5.0, 6.0}
