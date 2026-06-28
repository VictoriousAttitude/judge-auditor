"""Cross-validate the hand-rolled statistics against authoritative references.

Every headline estimator in this package is implemented from first principles
(pure NumPy ANOVA mean squares for ICC, the closed-form Fleiss' kappa, a Wilson
score interval, a rank partial correlation) so the runtime install stays light
and type-clean. The risk of a hand-rolled estimator is a silent formula bug, so
here we pin each one to an independent, peer-reviewed implementation:

* Fleiss' kappa  -> ``statsmodels.stats.inter_rater.fleiss_kappa``
* Wilson interval -> ``statsmodels.stats.proportion.proportion_confint(method="wilson")``
* ICC(1,1)/ICC(2,1) -> ``pingouin.intraclass_corr`` (``ICC(1,1)`` / ``ICC(A,1)``)
* partial Spearman  -> ``pingouin.partial_corr(method="spearman")``

statsmodels and pingouin are heavy and are *not* part of the shipped or default
``dev`` install; these tests ``importorskip`` so a lean checkout still passes.
Install them with ``pip install -e ".[refcheck]"`` to exercise this file.

One literature anchor (the Shrout & Fleiss 1979 worked example) runs
unconditionally: it depends on nothing but our own code and the published table,
so the cross-validation never silently vanishes when the optional libs are absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from judge_auditor.analysis.consistency import fleiss_kappa, icc_oneway, icc_twoway
from judge_auditor.analysis.stats import cohen_kappa, wilson_ci
from judge_auditor.analysis.verbosity_bias import _partial_spearman

# --- Literature anchor (no optional dependencies) -------------------------------

# Shrout & Fleiss (1979), "Intraclass correlations: uses in assessing rater
# reliability", Table 1: 6 targets rated by 4 judges. The paper reports
# ICC(1,1)=0.17 and ICC(2,1)=0.29 for this exact matrix.
_SHROUT_FLEISS_1979 = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)


def test_icc_matches_shrout_fleiss_1979_worked_example():
    assert icc_oneway(_SHROUT_FLEISS_1979) == pytest.approx(0.17, abs=0.005)
    assert icc_twoway(_SHROUT_FLEISS_1979) == pytest.approx(0.29, abs=0.005)


# --- statsmodels cross-checks ---------------------------------------------------


def test_cohen_kappa_matches_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    a = [int(x) for x in rng.integers(0, 3, size=50)]
    b = [int(x) for x in rng.integers(0, 3, size=50)]
    assert abs(cohen_kappa(a, b) - float(skm.cohen_kappa_score(a, b))) < 1e-9


def test_fleiss_kappa_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.inter_rater")
    rng = np.random.default_rng(0)
    counts = rng.multinomial(16, [0.5, 0.3, 0.2], size=25).astype(float)
    assert fleiss_kappa(counts) == pytest.approx(sm.fleiss_kappa(counts), abs=1e-12)


def test_wilson_interval_matches_statsmodels():
    prop = pytest.importorskip("statsmodels.stats.proportion")
    for successes, n in [(37, 50), (1, 20), (19, 20), (0, 8), (8, 8)]:
        ci = wilson_ci(successes, n)
        low, high = prop.proportion_confint(successes, n, alpha=0.05, method="wilson")
        assert ci.low == pytest.approx(low, abs=1e-12)
        assert ci.high == pytest.approx(high, abs=1e-12)


# --- pingouin cross-checks ------------------------------------------------------


def _long_frame(matrix: np.ndarray):
    pd = pytest.importorskip("pandas")
    n, k = matrix.shape
    rows = [
        {"target": i, "rater": j, "score": float(matrix[i, j])}
        for i in range(n)
        for j in range(k)
    ]
    return pd.DataFrame(rows)


def test_icc_matches_pingouin():
    pg = pytest.importorskip("pingouin")
    rng = np.random.default_rng(1)
    n, k = 15, 5
    matrix = rng.normal(size=(n, k)) + rng.normal(size=(n, 1))  # between-target variance
    res = pg.intraclass_corr(
        data=_long_frame(matrix), targets="target", raters="rater", ratings="score"
    ).set_index("Type")
    assert icc_oneway(matrix) == pytest.approx(float(res.loc["ICC(1,1)", "ICC"]), abs=1e-9)
    assert icc_twoway(matrix) == pytest.approx(float(res.loc["ICC(A,1)", "ICC"]), abs=1e-9)


def test_partial_spearman_matches_pingouin():
    pg = pytest.importorskip("pingouin")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(2)
    z = rng.normal(size=40)
    x = 0.5 * z + rng.normal(size=40)
    y = 0.3 * z + rng.normal(size=40)
    frame = pd.DataFrame({"x": x, "y": y, "z": z})
    expected = float(
        pg.partial_corr(data=frame, x="x", y="y", covar="z", method="spearman")["r"].iloc[0]
    )
    assert _partial_spearman(list(x), list(y), list(z)) == pytest.approx(expected, abs=1e-9)
