"""The measurement substrate: run a judge K times per example and collect verdicts.

Key responsibilities (Layer 0 of the architecture):

* repeated runs (K per example) to expose self-disagreement;
* position swapping for pairwise mode, so position bias and self-consistency are
  measured *from the same calls* (half the runs use order A,B; half use B,A);
* bounded-concurrency async execution with per-call parsing;
* checkpoint/resume via an append-only JSONL log so an interrupted audit can pick
  up exactly where it left off.

It performs no statistics — that is the analysis layer's job.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass

from ..config import AuditConfig, EvalExample, JudgeConfig, JudgeMode, PairwiseChoice, Winner
from ..records import JudgmentRecord, JudgmentSet, TaskKey
from .parser import parse_pairwise, parse_scalar
from .protocol import JudgeBackend
from .sampling import stratified_sample


@dataclass(frozen=True)
class _Task:
    example: EvalExample
    run_index: int
    ordering: str | None  # "AB" | "BA" | None (scalar)
    rubric_variant: int = 0

    @property
    def key(self) -> TaskKey:
        return (self.example.id, self.run_index, self.rubric_variant, self.ordering)


def _canonical_winner(choice: PairwiseChoice, ordering: str) -> Winner:
    """Map the presented position the judge picked back to the content (A/B)."""
    if choice is PairwiseChoice.TIE:
        return Winner.TIE
    first_is_a = ordering == "AB"
    if choice is PairwiseChoice.FIRST:
        return Winner.A if first_is_a else Winner.B
    return Winner.B if first_is_a else Winner.A


class JudgeExecutor:
    """Drives a :class:`JudgeBackend` to collect a :class:`JudgmentSet`."""

    def __init__(
        self,
        backend: JudgeBackend,
        judge_config: JudgeConfig,
        audit_config: AuditConfig | None = None,
    ) -> None:
        self.backend = backend
        self.judge = judge_config
        self.audit = audit_config or AuditConfig()

    # -- planning ----------------------------------------------------------------

    def select_examples(self, examples: list[EvalExample]) -> list[EvalExample]:
        if self.audit.sample_size is not None:
            return stratified_sample(examples, self.audit.sample_size, self.audit.seed)
        return list(examples)

    def _build_tasks(self, examples: list[EvalExample]) -> list[_Task]:
        k = self.audit.runs_per_example
        tasks: list[_Task] = []
        for ex in examples:
            if self.judge.mode is JudgeMode.PAIRWISE:
                if ex.response_b is None:
                    raise ValueError(f"example {ex.id!r} has no response_b for pairwise mode")
                n_ab = (k + 1) // 2  # the spare run (odd K) goes to the A,B ordering
                for i in range(k):
                    ordering = "AB" if i < n_ab else "BA"
                    tasks.append(_Task(ex, i, ordering))
            else:
                for i in range(k):
                    tasks.append(_Task(ex, i, None))
        return tasks

    def count_tasks(self, examples: list[EvalExample]) -> int:
        """Total judge calls a full audit of these examples will make."""
        return len(self._build_tasks(self.select_examples(examples)))

    # -- prompt rendering --------------------------------------------------------

    def _render(self, task: _Task) -> list[dict[str, str]]:
        ex = task.example
        if self.judge.mode is JudgeMode.PAIRWISE:
            assert ex.response_b is not None
            if task.ordering == "AB":
                first, second = ex.response_a, ex.response_b
            else:
                first, second = ex.response_b, ex.response_a
            user = self.judge.prompt_template.format(
                prompt=ex.prompt, response_a=first, response_b=second
            )
        else:
            user = self.judge.prompt_template.format(prompt=ex.prompt, response=ex.response_a)

        messages: list[dict[str, str]] = []
        if self.judge.system_prompt:
            messages.append({"role": "system", "content": self.judge.system_prompt})
        messages.append({"role": "user", "content": user})
        return messages

    # -- execution ---------------------------------------------------------------

    async def _run_task(self, task: _Task) -> JudgmentRecord:
        messages = self._render(task)
        resp = await self.backend.call(messages, self.judge)

        record = JudgmentRecord(
            example_id=task.example.id,
            run_index=task.run_index,
            rubric_variant=task.rubric_variant,
            ordering=task.ordering,
            raw_response=resp.text,
            parse_ok=False,
            model=self.judge.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            latency_s=resp.latency_s,
            timestamp=time.time(),
        )

        if self.judge.mode is JudgeMode.PAIRWISE:
            choice, err = parse_pairwise(resp.text)
            if choice is not None:
                record.choice = choice
                assert task.ordering is not None
                record.winner = _canonical_winner(choice, task.ordering)
                record.parse_ok = True
            else:
                record.parse_error = err
        else:
            score, err = parse_scalar(resp.text, self.judge.score_min, self.judge.score_max)
            if score is not None:
                record.score = score
                record.parse_ok = True
            else:
                record.parse_error = err

        return record

    async def run(self, examples: list[EvalExample]) -> JudgmentSet:
        """Run the full audit, returning every collected judgment."""
        selected = self.select_examples(examples)
        all_tasks = self._build_tasks(selected)

        done: dict[TaskKey, JudgmentRecord] = self._load_checkpoint()
        pending = [t for t in all_tasks if t.key not in done]

        sem = asyncio.Semaphore(self.audit.max_concurrency)
        write_lock = asyncio.Lock()
        buffer: list[JudgmentRecord] = []
        new_records: list[JudgmentRecord] = []

        async def worker(task: _Task) -> None:
            async with sem:
                record = await self._run_task(task)
            async with write_lock:
                new_records.append(record)
                buffer.append(record)
                if len(buffer) >= self.audit.checkpoint_every:
                    self._flush(buffer)
                    buffer.clear()

        await asyncio.gather(*(worker(t) for t in pending))
        self._flush(buffer)  # final partial batch

        records = list(done.values()) + new_records
        records.sort(key=lambda r: (r.example_id, r.rubric_variant, r.run_index))
        return JudgmentSet(mode=self.judge.mode, model=self.judge.model, records=records)

    # -- checkpointing -----------------------------------------------------------

    def _load_checkpoint(self) -> dict[TaskKey, JudgmentRecord]:
        path = self.audit.checkpoint_path
        if not path or not os.path.exists(path):
            return {}
        done: dict[TaskKey, JudgmentRecord] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = JudgmentRecord.from_dict(json.loads(line))
                done[record.key] = record  # last write wins on duplicate keys
        return done

    def _flush(self, records: list[JudgmentRecord]) -> None:
        path = self.audit.checkpoint_path
        if not path or not records:
            return
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.to_dict()) + "\n")
            f.flush()
