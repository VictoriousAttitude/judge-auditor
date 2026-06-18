from __future__ import annotations

import numpy as np

from judge_auditor.analysis.verbosity_bias import verbosity_bias, word_count
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
