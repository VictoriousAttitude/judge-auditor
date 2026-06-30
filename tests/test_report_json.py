from __future__ import annotations

import json
from enum import Enum

from judge_auditor.analysis.audit import audit
from judge_auditor.report.json_report import _jsonable, render_json, report_to_dict
from judge_auditor.report.recommendations import recommendations

from .test_audit import (
    compressed_scalar,
    consistent_pairwise,
    position_biased_pairwise,
    reliable_scalar,
)


class _PlainEnum(Enum):
    X = 1


def test_jsonable_handles_plain_enum_dict_and_fallback():
    assert _jsonable(_PlainEnum.X) == 1  # non-str enum serialized by value
    assert _jsonable({"k": 1.0, 2: "v"}) == {"k": 1.0, "2": "v"}  # dict keys stringified
    assert _jsonable(object()).startswith("<object")  # last-resort str() fallback


def test_render_json_is_valid_and_has_no_nan():
    # The coin-flip pairwise organism produces inf/nan internally; JSON must stay valid.
    js, examples = position_biased_pairwise()
    out = render_json(audit(js, examples))
    parsed = json.loads(out)  # raises if NaN/Infinity leaked through
    assert "NaN" not in out and "Infinity" not in out
    assert parsed["overall"] == "LOW"
    assert parsed["mode"] == "pairwise"
    assert parsed["power"]["mde_winrate"] is None  # inf -> null


def test_report_to_dict_includes_nested_results_and_recommendations():
    js, examples = reliable_scalar()
    d = report_to_dict(audit(js, examples))
    assert d["model"] == "judge-x"
    assert d["consistency"]["icc_oneway"]["point"] is not None
    assert d["scale"]["histogram"]  # list of bin counts
    assert isinstance(d["recommendations"], list)


def test_clean_scalar_has_only_noise_floor_recommendation():
    js, examples = reliable_scalar()
    recs = recommendations(audit(js, examples))
    # No bias/consistency/scale flags => only the always-on noise-floor note remains.
    assert len(recs) == 1
    assert "Noise floor" in recs[0]


def test_compressed_scalar_recommends_reducing_scale():
    js, examples = compressed_scalar()
    recs = recommendations(audit(js, examples))
    assert any("compressed" in r.lower() for r in recs)


def test_position_biased_recommends_swapping():
    js, examples = position_biased_pairwise()
    recs = recommendations(audit(js, examples))
    assert any("Position bias" in r for r in recs)
    assert any("no discriminating power" in r for r in recs)  # coin-flip noise floor


def test_consistent_pairwise_no_position_recommendation():
    js, examples = consistent_pairwise()
    recs = recommendations(audit(js, examples))
    assert not any("Position bias" in r for r in recs)
