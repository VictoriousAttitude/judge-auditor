from __future__ import annotations

import numpy as np

from judge_auditor.analysis.verbosity_bias import (
    _stratify_length_effect,
    verbosity_bias,
    word_count,
)
from judge_auditor.config import EvalExample, JudgeMode, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet


def words(n: int) -> str:
    return " ".join(["w"] * n)


def scalar_case(scores_by_length: dict[int, float], runs: int = 3):
    examples, records = [], []
    for i, (length, score) in enumerate(scores_by_length.items()):
        eid = f"ex{i}"
        examples.append(EvalExample(id=eid, prompt="q", response_a=words(length)))
        for j in range(runs):
            records.append(JudgmentRecord(eid, j, 0, None, str(score), True, score=score))
    return JudgmentSet(JudgeMode.SCALAR, "m", records), examples


def test_scalar_length_correlated_scores_flag():
    # Score increases monotonically with length => Spearman rho ~ 1, and with
    # enough examples (n >= min_n) the correlation is significant => flagged.
    data = {
        5: 1.0, 10: 2.0, 15: 2.5, 20: 3.0, 30: 4.0, 40: 5.0,
        55: 6.0, 70: 6.5, 90: 7.5, 110: 8.0, 140: 9.0, 180: 10.0,
    }
    js, examples = scalar_case(data)
    res = verbosity_bias(js, examples)
    assert res.spearman_rho is not None and res.spearman_rho > 0.95
    assert res.spearman_p is not None and res.spearman_p < 0.05
    assert res.flagged


def test_scalar_small_n_large_rho_not_flagged():
    # A sizeable |rho| (~0.5 > threshold) on only a handful of examples is NOT
    # statistically significant and falls below min_n: the flag must stay off.
    # This is the small-n false positive the significance/min-n gate fixes.
    data = {5: 3.0, 12: 7.0, 30: 4.0, 60: 8.0, 120: 6.0}
    js, examples = scalar_case(data)
    res = verbosity_bias(js, examples)
    assert res.n_examples == 5
    assert res.spearman_rho is not None and abs(res.spearman_rho) > res.threshold
    assert not res.flagged


def test_scalar_length_independent_scores_not_flagged():
    rng = np.random.default_rng(0)
    examples, records = [], []
    for i in range(40):
        eid = f"ex{i}"
        length = int(rng.integers(5, 100))
        score = float(rng.integers(1, 11))  # independent of length
        examples.append(EvalExample(id=eid, prompt="q", response_a=words(length)))
        for j in range(3):
            records.append(JudgmentRecord(eid, j, 0, None, str(score), True, score=score))
    js = JudgmentSet(JudgeMode.SCALAR, "m", records)
    res = verbosity_bias(js, examples)
    assert abs(res.spearman_rho) < 0.4
    assert not res.flagged


def test_scalar_partial_correlation_removes_quality_confound():
    # Length and score are correlated only THROUGH quality: longer == higher
    # quality == higher score. Controlling for quality should null the partial rho.
    # Length and quality are correlated but NOT perfectly collinear (noise added),
    # otherwise the partial-correlation denominator degenerates to zero.
    rng = np.random.default_rng(0)
    examples, records = [], []
    for i in range(40):
        quality = float(i)
        # length tracks quality with independent noise (avoids rank collinearity)
        length = max(1, int(round(5 + 2 * i + rng.normal(0, 6))))
        # score tracks quality (not length) with its own independent noise, so the
        # score~quality rank correlation is high but < 1 (denominator stays nonzero)
        score = quality + rng.normal(0, 1.5)
        eid = f"ex{i}"
        examples.append(
            EvalExample(id=eid, prompt="q", response_a=words(length), quality_label=quality)
        )
        records.append(JudgmentRecord(eid, 0, 0, None, str(score), True, score=score))
        records.append(JudgmentRecord(eid, 1, 0, None, str(score), True, score=score))
    js = JudgmentSet(JudgeMode.SCALAR, "m", records)
    res = verbosity_bias(js, examples)
    assert res.spearman_rho > 0.9  # raw correlation is strong (via quality)
    assert res.partial_rho is not None and abs(res.partial_rho) < 0.2  # vanishes


def test_pairwise_longer_response_wins():
    # response_a is always longer AND always wins => longer-win-rate ~ 1, rho > 0.
    examples, records = [], []
    for i in range(20):
        eid = f"ex{i}"
        examples.append(
            EvalExample(id=eid, prompt="q", response_a=words(50 + i), response_b=words(5))
        )
        for j in range(6):
            ordering = "AB" if j < 3 else "BA"
            records.append(
                JudgmentRecord(eid, j, 0, ordering, "x", True, winner=Winner.A)
            )
    js = JudgmentSet(JudgeMode.PAIRWISE, "m", records)
    res = verbosity_bias(js, examples)
    assert res.longer_response_win_rate is not None
    assert res.longer_response_win_rate > 0.95


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("  spaced   out  ") == 2
    assert word_count("") == 0


# --- within-quality-stratum interaction (feature (a)) ---------------------------


def stratified_case(tiers, runs: int = 5):
    """Build a scalar set: each tier = (quality, short_score, long_score) with 4
    short (len 10-13) + 4 long (len 80-83) examples."""
    examples, records = [], []
    i = 0
    for quality, short_score, long_score in tiers:
        for length, score in (
            *[(10 + k, short_score) for k in range(4)],
            *[(80 + k, long_score) for k in range(4)],
        ):
            eid = f"e{i}"
            i += 1
            examples.append(
                EvalExample(id=eid, prompt="q", response_a=words(length), quality_label=quality)
            )
            for j in range(runs):
                records.append(JudgmentRecord(eid, j, 0, None, str(score), True, score=score))
    return JudgmentSet(JudgeMode.SCALAR, "m", records), examples


def test_within_stratum_interaction_caught_when_global_misses_it():
    # Length penalty lives ONLY in the top-quality tier (correct-but-verbose docked
    # ~5 pts); other tiers have no length effect. This mimics the live Sonnet finding:
    # the global correlation washes it out, but the stratified check catches it.
    tiers = [(9.0, 10.0, 5.0), (7.0, 7.0, 7.0), (5.0, 5.0, 5.0), (2.0, 2.0, 2.0)]
    js, examples = stratified_case(tiers)
    res = verbosity_bias(js, examples)

    assert not res.flagged  # the global score~length correlation misses it
    assert res.stratified_flagged  # the within-stratum check catches it
    assert res.strata is not None and len(res.strata) == 1  # only the q=9 tier has signal
    top = res.strata[0]
    assert top.quality == 9.0
    assert top.flagged
    assert top.score_gap < -3.0  # longer responses scored markedly lower
    assert top.spearman_rho < -0.5


def test_stratified_no_length_effect_not_flagged():
    # Discrete quality, lengths vary, but score depends only on quality (constant
    # within each tier) => no within-stratum length signal => not flagged.
    tiers = [(9.0, 9.0, 9.0), (5.0, 5.0, 5.0), (2.0, 2.0, 2.0)]
    js, examples = stratified_case(tiers)
    res = verbosity_bias(js, examples)
    assert not res.stratified_flagged
    assert res.strata is None or all(not s.flagged for s in res.strata)


def test_continuous_quality_label_skips_stratification():
    # Many distinct quality values (continuous) => no discrete strata to hold quality
    # fixed => stratified analysis stays off; rely on the global/partial correlation.
    rng = np.random.default_rng(0)
    examples, records = [], []
    for i in range(40):
        eid = f"c{i}"
        examples.append(
            EvalExample(id=eid, prompt="q", response_a=words(10 + i), quality_label=float(i))
        )
        score = i / 4.0 + rng.normal(0, 0.3)
        for j in range(2):
            records.append(JudgmentRecord(eid, j, 0, None, str(score), True, score=score))
    js = JudgmentSet(JudgeMode.SCALAR, "m", records)
    res = verbosity_bias(js, examples)
    assert res.strata is None
    assert not res.stratified_flagged


def test_no_quality_label_skips_stratification():
    js, examples = scalar_case({5: 2.0, 10: 4.0, 20: 6.0, 40: 8.0})
    res = verbosity_bias(js, examples)  # examples carry no quality_label
    assert res.strata is None
    assert not res.stratified_flagged


def test_scalar_lengths_override_is_used():
    js, examples = scalar_case({10: 5.0})
    eid = examples[0].id
    res = verbosity_bias(js, examples, lengths={eid: 999})
    assert res.n_examples == 1  # the override-length branch executed


def test_scalar_skips_record_for_unknown_example():
    js, examples = scalar_case({10: 5.0, 20: 6.0})
    js.records.append(JudgmentRecord("ghost", 0, 0, None, "5", True, score=5.0))
    res = verbosity_bias(js, examples)
    assert res.n_examples == 2  # the ghost example (absent from examples) is skipped


def test_scalar_skips_example_with_no_parsed_scores():
    js, examples = scalar_case({10: 5.0, 20: 6.0})
    examples.append(EvalExample(id="bad", prompt="q", response_a=words(30)))
    js.records.append(JudgmentRecord("bad", 0, 0, None, "junk", False, parse_error="x"))
    res = verbosity_bias(js, examples)
    assert res.n_examples == 2  # the all-unparseable example is skipped


def test_pairwise_skips_example_without_response_b():
    examples = [
        EvalExample(id="ok", prompt="q", response_a=words(10), response_b=words(20)),
        EvalExample(id="nob", prompt="q", response_a=words(10), response_b=None),
    ]
    records = [
        JudgmentRecord("ok", 0, 0, "AB", "x", True, winner=Winner.A),
        JudgmentRecord("ok", 1, 0, "BA", "x", True, winner=Winner.B),
        JudgmentRecord("nob", 0, 0, "AB", "x", True, winner=Winner.A),
    ]
    js = JudgmentSet(JudgeMode.PAIRWISE, "m", records)
    res = verbosity_bias(js, examples)
    assert res.n_examples == 1  # the response_b-less example is skipped


def test_stratify_skips_undersized_stratum():
    quals = [0.0] * 8 + [1.0] * 3
    scores = [float(x) for x in (1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3)]
    lens = [float(x) for x in (1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1)]
    out = _stratify_length_effect(
        scores, lens, quals,
        max_strata=6, min_stratum_n=8, threshold=0.3, p_threshold=0.05,
    )
    assert [se.quality for se in out] == [0.0]  # the 3-example stratum is dropped


def test_stratify_alternate_split_when_median_at_minimum():
    quals = [0.0] * 8 + [1.0] * 8
    lens = [float(x) for x in (1, 1, 1, 1, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8)]
    scores = [float(x) for x in (5, 6, 4, 7, 3, 8, 2, 9, 1, 2, 3, 4, 5, 6, 7, 8)]
    out = _stratify_length_effect(
        scores, lens, quals,
        max_strata=6, min_stratum_n=4, threshold=0.3, p_threshold=0.05,
    )
    assert sorted(se.quality for se in out) == [0.0, 1.0]
