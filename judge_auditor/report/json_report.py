"""Machine-readable JSON view of a :class:`ReliabilityReport`.

For CI integration and programmatic consumption. The whole report tree (nested
result dataclasses, enums, confidence intervals) is serialized generically, and —
critically — non-finite floats (NaN/inf, which legitimately arise from undefined
correlations or a zero-discriminability noise floor) are mapped to ``null`` so the
output is *always valid JSON* (``allow_nan=False``), never the ``NaN`` token that
breaks strict parsers.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from ..analysis.audit import ReliabilityReport
from .recommendations import recommendations


def _jsonable(obj: Any) -> Any:
    """Recursively convert a report tree into JSON-safe primitives."""
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None  # NaN / inf -> null
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list | tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return str(obj)  # last-resort fallback (should not normally trigger)


def report_to_dict(report: ReliabilityReport) -> dict[str, Any]:
    """JSON-safe dict of the full report, with recommendations attached."""
    data = _jsonable(report)
    assert isinstance(data, dict)
    data["recommendations"] = recommendations(report)
    return data


def render_json(report: ReliabilityReport, *, indent: int | None = 2) -> str:
    """Serialize the report as valid JSON (no NaN/Infinity tokens)."""
    return json.dumps(report_to_dict(report), indent=indent, allow_nan=False)
