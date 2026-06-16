from __future__ import annotations

from collections import Counter

from judge_auditor.config import EvalExample
from judge_auditor.runner.sampling import stratified_sample


def _examples(n: int, categories: list[str] | None = None) -> list[EvalExample]:
    out = []
    for i in range(n):
        meta = {"category": categories[i % len(categories)]} if categories else {}
        out.append(EvalExample(id=f"ex{i}", prompt="q", response_a="a", metadata=meta))
    return out


def test_returns_all_when_k_exceeds_size():
    ex = _examples(5)
    assert stratified_sample(ex, 10) == ex


def test_sample_size_and_determinism():
    ex = _examples(50)
    s1 = stratified_sample(ex, 12, seed=3)
    s2 = stratified_sample(ex, 12, seed=3)
    assert len(s1) == 12
    assert [e.id for e in s1] == [e.id for e in s2]


def test_preserves_original_order():
    ex = _examples(30)
    order = {e.id: i for i, e in enumerate(ex)}
    sample = stratified_sample(ex, 10, seed=0)
    idx = [order[e.id] for e in sample]
    assert idx == sorted(idx)


def test_strata_are_represented():
    ex = _examples(60, categories=["math", "code", "writing"])
    sample = stratified_sample(ex, 9, seed=7)
    counts = Counter(e.metadata["category"] for e in sample)
    # Each of the three equal-sized strata should contribute to the sample.
    assert set(counts) == {"math", "code", "writing"}
    assert len(sample) == 9
