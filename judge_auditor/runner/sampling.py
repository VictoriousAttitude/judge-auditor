"""Stratified sampling of eval examples for cost-bounded audits.

When an eval set is large, auditing every item is wasteful. We draw a sample
stratified by ``metadata["category"]`` (when present) so each category keeps its
share of the sample; otherwise we fall back to a plain seeded random sample.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from ..config import EvalExample


def stratified_sample(
    examples: list[EvalExample], k: int, seed: int = 0
) -> list[EvalExample]:
    """Return a deterministic sample of ``k`` examples, stratified by category."""
    if k >= len(examples):
        return list(examples)

    rng = random.Random(seed)
    strata: dict[str, list[EvalExample]] = defaultdict(list)
    for ex in examples:
        strata[str(ex.metadata.get("category", "_"))].append(ex)

    # Proportional allocation, rounding up so small strata are represented.
    selected: list[EvalExample] = []
    total = len(examples)
    for items in strata.values():
        rng.shuffle(items)
        take = min(len(items), math.ceil(k * len(items) / total))
        selected.extend(items[:take])

    # Rounding up can overshoot k; trim deterministically.
    rng.shuffle(selected)
    selected = selected[:k]
    # Preserve original ordering for reproducible, readable output.
    order = {ex.id: i for i, ex in enumerate(examples)}
    selected.sort(key=lambda ex: order[ex.id])
    return selected
