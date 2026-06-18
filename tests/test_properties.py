"""Property-based invariants for the estimators and the record round-trip.

The example-based tests pin each statistic to a handful of known inputs; these
``hypothesis`` tests instead assert the *mathematical guarantees* that must hold
for every admissible input. A formula bug that happens to give the right answer
on the canned examples (a sign slip, a denominator that can exceed 1, a clamp
that is missing) tends to show up as a violated invariant on some adversarial
input hypothesis discovers.

The invariants checked here:

* Fleiss' kappa is at most 1 (it chance-corrects agreement; >1 is impossible).
* ICC(1,1) and ICC(2,1) are at most 1 (a ratio of variance components).
* The Wilson interval stays inside ``[0, 1]`` and brackets the point estimate.
* A partial Spearman correlation is a correlation: it lies in ``[-1, 1]`` (or is
  NaN for a degenerate, constant input).
* A ``JudgmentSet`` survives a ``to_dict`` / ``from_dict`` round-trip unchanged.

Degenerate inputs (all-equal columns, single-category ratings) legitimately
produce NaN; the invariants are asserted only where a finite value is defined.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from judge_auditor.analysis.consistency import fleiss_kappa, icc_oneway, icc_twoway
from judge_auditor.analysis.stats import wilson_ci
from judge_auditor.analysis.verbosity_bias import _partial_spearman
from judge_auditor.config import JudgeMode, PairwiseChoice, Winner
from judge_auditor.records import JudgmentRecord, JudgmentSet

_TOL = 1e-9

# --- Estimator invariants -------------------------------------------------------


@st.composite
def _fleiss_counts(draw: st.DrawFn) -> np.ndarray:
    """An n x c per-subject category-count matrix; every row sums to m >= 2."""
    n = draw(st.integers(min_value=2, max_value=20))
    c = draw(st.integers(min_value=2, max_value=5))
    m = draw(st.integers(min_value=2, max_value=16))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(c))
    return rng.multinomial(m, probs, size=n).astype(float)


@given(_fleiss_counts())
@settings(max_examples=200, deadline=None)
def test_fleiss_kappa_never_exceeds_one(counts: np.ndarray) -> None:
    k = fleiss_kappa(counts)
    if not math.isnan(k):
        assert k <= 1.0 + _TOL


@st.composite
def _score_matrix(draw: st.DrawFn) -> np.ndarray:
    """An n x k targets-by-runs matrix of finite, bounded scores (n, k >= 2)."""
    n = draw(st.integers(min_value=2, max_value=12))
    k = draw(st.integers(min_value=2, max_value=8))
    elements = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    return draw(hnp.arrays(dtype=np.float64, shape=(n, k), elements=elements))


@given(_score_matrix())
@settings(max_examples=200, deadline=None)
def test_icc_never_exceeds_one(matrix: np.ndarray) -> None:
    for icc in (icc_oneway(matrix), icc_twoway(matrix)):
        if not math.isnan(icc):
            assert icc <= 1.0 + _TOL


@given(
    n=st.integers(min_value=1, max_value=10_000),
    data=st.data(),
)
@settings(max_examples=200, deadline=None)
def test_wilson_interval_is_a_valid_subinterval_of_unit(n: int, data: st.DataObject) -> None:
    successes = data.draw(st.integers(min_value=0, max_value=n))
    ci = wilson_ci(successes, n)
    assert ci.low >= 0.0 - _TOL
    assert ci.low <= ci.point + _TOL
    assert ci.point <= ci.high + _TOL
    assert ci.high <= 1.0 + _TOL


@given(
    rows=st.lists(
        st.tuples(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
        min_size=4,
        max_size=60,
    )
)
@settings(max_examples=200, deadline=None)
def test_partial_spearman_stays_in_correlation_range(
    rows: list[tuple[float, float, float]],
) -> None:
    x = [r[0] for r in rows]
    y = [r[1] for r in rows]
    z = [r[2] for r in rows]
    rho = _partial_spearman(x, y, z)
    if not math.isnan(rho):
        assert -1.0 - _TOL <= rho <= 1.0 + _TOL


# --- Record round-trip ----------------------------------------------------------


_finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9)
_opt_int = st.none() | st.integers(min_value=0, max_value=10**6)


@st.composite
def _record(draw: st.DrawFn) -> JudgmentRecord:
    return JudgmentRecord(
        example_id=draw(st.text(min_size=1, max_size=12)),
        run_index=draw(st.integers(min_value=0, max_value=100)),
        rubric_variant=draw(st.integers(min_value=0, max_value=5)),
        ordering=draw(st.none() | st.sampled_from(["AB", "BA"])),
        raw_response=draw(st.text(max_size=40)),
        parse_ok=draw(st.booleans()),
        choice=draw(st.none() | st.sampled_from(list(PairwiseChoice))),
        winner=draw(st.none() | st.sampled_from(list(Winner))),
        score=draw(st.none() | _finite),
        parse_error=draw(st.none() | st.text(max_size=40)),
        model=draw(st.text(max_size=12)),
        prompt_tokens=draw(_opt_int),
        completion_tokens=draw(_opt_int),
        latency_s=draw(st.none() | _finite),
        timestamp=draw(_finite),
    )


@given(
    mode=st.sampled_from(list(JudgeMode)),
    model=st.text(max_size=12),
    records=st.lists(_record(), max_size=12),
)
@settings(max_examples=150, deadline=None)
def test_judgment_set_survives_json_round_trip(
    mode: JudgeMode, model: str, records: list[JudgmentRecord]
) -> None:
    js = JudgmentSet(mode=mode, model=model, records=records)
    assert JudgmentSet.from_dict(js.to_dict()) == js
