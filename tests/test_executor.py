from __future__ import annotations

from collections import Counter

import pytest

from judge_auditor.config import AuditConfig, EvalExample, JudgeConfig, JudgeMode, Probe, Winner
from judge_auditor.runner.backends.mock import MockBackend
from judge_auditor.runner.executor import JudgeExecutor

from .responders import (
    PAIRWISE_TEMPLATE,
    always_first_pairwise,
    content_pref_pairwise,
    content_score_scalar,
    malformed,
)


async def test_run_count_pairwise(pairwise_config, pairwise_examples):
    audit = AuditConfig(runs_per_example=15)
    backend = MockBackend(content_pref_pairwise)
    ex = JudgeExecutor(backend, pairwise_config, audit)
    result = await ex.run(pairwise_examples)
    # K runs per example, every call recorded.
    assert len(result) == 15 * len(pairwise_examples)
    assert backend.call_count == len(result)
    assert ex.count_tasks(pairwise_examples) == len(result)


async def test_position_swap_is_balanced(pairwise_config, pairwise_examples):
    audit = AuditConfig(runs_per_example=15)
    ex = JudgeExecutor(MockBackend(content_pref_pairwise), pairwise_config, audit)
    result = await ex.run(pairwise_examples)
    per_example = result.for_example("ex0")
    orders = Counter(r.ordering for r in per_example)
    # Odd K: spare run goes to A,B ordering.
    assert orders["AB"] == 8
    assert orders["BA"] == 7


async def test_content_judge_is_position_invariant(pairwise_config, pairwise_examples):
    """A content-driven judge yields the same canonical winner under both orders."""
    ex = JudgeExecutor(
        MockBackend(content_pref_pairwise), pairwise_config, AuditConfig(runs_per_example=6)
    )
    result = await ex.run(pairwise_examples)
    assert all(r.parse_ok for r in result.records)
    # response_a is the GOOD one for every example, so A wins regardless of order.
    assert {r.winner for r in result.records} == {Winner.A}


async def test_position_biased_judge_flips_with_order(pairwise_config, pairwise_examples):
    """Always-pick-first judge => winner depends purely on presentation order."""
    ex = JudgeExecutor(
        MockBackend(always_first_pairwise), pairwise_config, AuditConfig(runs_per_example=10)
    )
    result = await ex.run(pairwise_examples)
    for r in result.records:
        assert r.winner == (Winner.A if r.ordering == "AB" else Winner.B)


async def test_parse_failures_are_recorded_not_dropped(pairwise_config, pairwise_examples):
    ex = JudgeExecutor(
        MockBackend(malformed), pairwise_config, AuditConfig(runs_per_example=4)
    )
    result = await ex.run(pairwise_examples)
    assert len(result) == 4 * len(pairwise_examples)  # nothing dropped
    assert result.parse_failure_rate == 1.0
    for r in result.records:
        assert not r.parse_ok
        assert r.winner is None
        assert r.parse_error
        assert r.raw_response  # raw text always kept


async def test_scalar_scores_recorded(scalar_config, scalar_examples):
    ex = JudgeExecutor(
        MockBackend(content_score_scalar), scalar_config, AuditConfig(runs_per_example=3)
    )
    result = await ex.run(scalar_examples)
    assert all(r.ordering is None for r in result.records)
    assert all(r.parse_ok and r.score is not None for r in result.records)


async def test_concurrency_does_not_lose_records(pairwise_config, pairwise_examples):
    ex = JudgeExecutor(
        MockBackend(content_pref_pairwise),
        pairwise_config,
        AuditConfig(runs_per_example=8, max_concurrency=5),
    )
    result = await ex.run(pairwise_examples)
    assert len(result) == 8 * len(pairwise_examples)


async def test_checkpoint_resume_skips_completed(tmp_path, pairwise_config, pairwise_examples):
    ckpt = str(tmp_path / "audit.jsonl")
    audit = AuditConfig(runs_per_example=6, checkpoint_path=ckpt, checkpoint_every=3)

    backend1 = MockBackend(content_pref_pairwise)
    result1 = await JudgeExecutor(backend1, pairwise_config, audit).run(pairwise_examples)
    expected = 6 * len(pairwise_examples)
    assert backend1.call_count == expected

    # A fresh run against the same checkpoint should make zero new calls.
    backend2 = MockBackend(content_pref_pairwise)
    result2 = await JudgeExecutor(backend2, pairwise_config, audit).run(pairwise_examples)
    assert backend2.call_count == 0
    assert len(result2) == expected == len(result1)


async def test_checkpoint_partial_resume(tmp_path, pairwise_config, pairwise_examples):
    ckpt = str(tmp_path / "audit.jsonl")

    # First pass over a single example, then resume over the full set.
    audit = AuditConfig(runs_per_example=4, checkpoint_path=ckpt)
    backend1 = MockBackend(content_pref_pairwise)
    await JudgeExecutor(backend1, pairwise_config, audit).run(pairwise_examples[:1])
    assert backend1.call_count == 4

    backend2 = MockBackend(content_pref_pairwise)
    result = await JudgeExecutor(backend2, pairwise_config, audit).run(pairwise_examples)
    # Only the 3 not-yet-done examples are re-run (4 runs each).
    assert backend2.call_count == 4 * (len(pairwise_examples) - 1)
    assert len(result) == 4 * len(pairwise_examples)


async def test_sampling_limits_examples(pairwise_config):
    examples = [
        EvalExample(id=f"ex{i}", prompt="q", response_a="GOOD", response_b="bad")
        for i in range(20)
    ]
    audit = AuditConfig(runs_per_example=2, sample_size=5, seed=1)
    ex = JudgeExecutor(MockBackend(content_pref_pairwise), pairwise_config, audit)
    result = await ex.run(examples)
    assert len(result.example_ids) == 5
    assert len(result) == 2 * 5


def test_pairwise_config_requires_response_b(scalar_config):
    examples = [EvalExample(id="x", prompt="q", response_a="only one")]
    pairwise = JudgeConfig(
        model="m", prompt_template=PAIRWISE_TEMPLATE, mode=JudgeMode.PAIRWISE
    )
    ex = JudgeExecutor(MockBackend(content_pref_pairwise), pairwise, AuditConfig())
    with pytest.raises(ValueError, match="response_b"):
        ex.count_tasks(examples)


def test_config_validates_template_placeholders():
    with pytest.raises(ValueError, match="missing placeholders"):
        JudgeConfig(model="m", prompt_template="no slots", mode=JudgeMode.SCALAR)


# --- Rubric variants ------------------------------------------------------------


def test_templates_property_lists_base_then_variants():
    cfg = JudgeConfig(
        model="m",
        prompt_template="Q: {prompt}\nR: {response}\nScore 1-10.",
        mode=JudgeMode.SCALAR,
        rubric_variants=("Rate {response} for {prompt} on 1-10.",),
    )
    assert len(cfg.templates) == 2
    assert cfg.templates[0] == cfg.prompt_template


def test_config_validates_variant_placeholders():
    with pytest.raises(ValueError, match="missing placeholders"):
        JudgeConfig(
            model="m",
            prompt_template="Q: {prompt}\nR: {response}\nScore.",
            mode=JudgeMode.SCALAR,
            rubric_variants=("this variant forgot the slots",),
        )


async def test_runs_fan_out_over_rubric_variants(scalar_examples):
    cfg = JudgeConfig(
        model="m",
        prompt_template="Q: {prompt}\nR: {response}\nScore 1-10.",
        mode=JudgeMode.SCALAR,
        rubric_variants=("Rephrased: rate {response} for {prompt} 1-10.",),
    )
    ex = JudgeExecutor(MockBackend(content_score_scalar), cfg, AuditConfig(runs_per_example=3))
    result = await ex.run(scalar_examples)
    # K runs per (example, variant); two variants => double the records.
    assert len(result) == 3 * 2 * len(scalar_examples)
    assert {r.rubric_variant for r in result.records} == {0, 1}
    # Each example/variant cell has exactly K runs.
    cell = [r for r in result.records if r.example_id == "ex0" and r.rubric_variant == 1]
    assert len(cell) == 3


# --- Probes ---------------------------------------------------------------------


async def test_scalar_probes_add_balanced_conditions(scalar_config, scalar_examples):
    audit = AuditConfig(runs_per_example=4, probe_sycophancy=True, probe_anchoring=True)
    ex = JudgeExecutor(MockBackend(content_score_scalar), scalar_config, audit)
    result = await ex.run(scalar_examples)
    # NEUTRAL + 2 sycophancy + 2 anchoring directions, K runs each.
    n = len(scalar_examples)
    assert len(result) == 4 * 5 * n
    probes = Counter(r.probe for r in result.records)
    assert probes[Probe.NEUTRAL] == 4 * n
    for p in (Probe.SYCOPHANCY_UP, Probe.SYCOPHANCY_DOWN, Probe.ANCHOR_UP, Probe.ANCHOR_DOWN):
        assert probes[p] == 4 * n


async def test_pairwise_anchoring_probe_is_ignored(pairwise_config, pairwise_examples):
    """Anchoring needs a numeric scale, so it adds no calls in pairwise mode."""
    audit = AuditConfig(runs_per_example=4, probe_sycophancy=True, probe_anchoring=True)
    ex = JudgeExecutor(MockBackend(content_pref_pairwise), pairwise_config, audit)
    result = await ex.run(pairwise_examples)
    probes = {r.probe for r in result.records}
    assert Probe.ANCHOR_UP not in probes and Probe.ANCHOR_DOWN not in probes
    assert Probe.SYCOPHANCY_UP in probes and Probe.SYCOPHANCY_DOWN in probes


async def test_probes_ride_only_canonical_rubric(scalar_examples):
    cfg = JudgeConfig(
        model="m",
        prompt_template="Q: {prompt}\nR: {response}\nScore 1-10.",
        mode=JudgeMode.SCALAR,
        rubric_variants=("Rephrased: rate {response} for {prompt} 1-10.",),
    )
    audit = AuditConfig(runs_per_example=2, probe_sycophancy=True)
    ex = JudgeExecutor(MockBackend(content_score_scalar), cfg, audit)
    result = await ex.run(scalar_examples)
    # Probe records only exist at rubric_variant 0 (bounded call count).
    probe_variants = {r.rubric_variant for r in result.records if r.probe is not Probe.NEUTRAL}
    assert probe_variants == {0}


async def test_probe_prefix_appears_in_prompt(scalar_examples):
    seen: list[tuple[str, str]] = []

    def capture(messages, config):
        seen.append((messages[-1]["content"], config.model))
        return '{"score": 5}'

    cfg = JudgeConfig(
        model="m",
        prompt_template="BASE {prompt} {response}",
        mode=JudgeMode.SCALAR,
    )
    audit = AuditConfig(runs_per_example=1, probe_sycophancy=True)
    ex = JudgeExecutor(MockBackend(capture), cfg, audit)
    await ex.run(scalar_examples[:1])
    texts = [t for t, _ in seen]
    # The up/down sycophancy cues are prepended; the neutral run has no prefix.
    assert any("high score" in t for t in texts)
    assert any("low score" in t for t in texts)
    assert any(t.startswith("BASE") for t in texts)


async def test_system_prompt_is_prepended(scalar_examples):
    seen: list[list[dict[str, str]]] = []

    def capture(messages, config):
        seen.append(messages)
        return '{"score": 5}'

    cfg = JudgeConfig(
        model="m",
        prompt_template="Q: {prompt}\nR: {response}\nScore 1-10.",
        mode=JudgeMode.SCALAR,
        system_prompt="You are a strict grader.",
    )
    ex = JudgeExecutor(MockBackend(capture), cfg, AuditConfig(runs_per_example=1))
    await ex.run(scalar_examples[:1])
    assert seen[0][0] == {"role": "system", "content": "You are a strict grader."}
    assert seen[0][-1]["role"] == "user"


async def test_scalar_parse_failures_are_recorded(scalar_config, scalar_examples):
    ex = JudgeExecutor(
        MockBackend(malformed), scalar_config, AuditConfig(runs_per_example=3)
    )
    result = await ex.run(scalar_examples)
    assert result.parse_failure_rate == 1.0
    for r in result.records:
        assert not r.parse_ok
        assert r.score is None
        assert r.parse_error


async def test_checkpoint_skips_blank_lines(tmp_path, scalar_config, scalar_examples):
    # A checkpoint file padded with blank lines must load cleanly.
    ckpt = tmp_path / "audit.jsonl"
    audit = AuditConfig(runs_per_example=1, checkpoint_path=str(ckpt))
    backend1 = MockBackend(content_score_scalar)
    await JudgeExecutor(backend1, scalar_config, audit).run(scalar_examples)

    contents = ckpt.read_text(encoding="utf-8")
    ckpt.write_text("\n" + contents + "\n\n", encoding="utf-8")

    backend2 = MockBackend(content_score_scalar)
    result = await JudgeExecutor(backend2, scalar_config, audit).run(scalar_examples)
    assert backend2.call_count == 0  # everything resolved from the checkpoint
    assert len(result) == len(scalar_examples)


async def test_variants_render_different_prompts(scalar_examples):
    seen: list[str] = []

    def capture(messages, config):
        seen.append(messages[-1]["content"])
        return '{"score": 5}'

    cfg = JudgeConfig(
        model="m",
        prompt_template="BASE {prompt} {response}",
        mode=JudgeMode.SCALAR,
        rubric_variants=("ALT {prompt} {response}",),
    )
    ex = JudgeExecutor(MockBackend(capture), cfg, AuditConfig(runs_per_example=1))
    await ex.run(scalar_examples[:1])
    assert any(s.startswith("BASE") for s in seen)
    assert any(s.startswith("ALT") for s in seen)
