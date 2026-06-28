"""Runtime output data models: collected judgments.

The executor produces one :class:`JudgmentRecord` per judge call and groups them
into a :class:`JudgmentSet`. Analysis modules (Phase 2) are pure functions over a
``JudgmentSet`` — they never call the judge themselves. Records are JSON-round-
trippable so audits can be checkpointed and resumed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import JudgeMode, PairwiseChoice, Probe, Winner

# Identifies a single unit of work, used to dedupe on resume:
# (example_id, run_index, rubric_variant, probe, ordering).
TaskKey = tuple[str, int, int, str, str | None]


@dataclass
class JudgmentRecord:
    """One judge call and everything we learned from it.

    Both the raw response and the parsed verdict are kept: a high parse-failure
    rate is itself a reliability signal, so failures are recorded, never dropped.
    """

    example_id: str
    run_index: int
    rubric_variant: int  # 0 = the original rubric
    ordering: str | None  # "AB" | "BA" for pairwise, None for scalar
    raw_response: str
    parse_ok: bool

    # Prompt perturbation under which this judgment was collected (NEUTRAL = the
    # unperturbed audit data that every headline metric is computed on).
    probe: Probe = Probe.NEUTRAL

    # Pairwise verdict.
    choice: PairwiseChoice | None = None  # the presented position the judge picked
    winner: Winner | None = None  # resolved to the canonical content (A/B/tie)

    # Scalar verdict.
    score: float | None = None

    parse_error: str | None = None

    # Telemetry.
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    timestamp: float = 0.0

    @property
    def key(self) -> TaskKey:
        return (
            self.example_id,
            self.run_index,
            self.rubric_variant,
            self.probe.value,
            self.ordering,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Serialize enums by value.
        d["choice"] = self.choice.value if self.choice is not None else None
        d["winner"] = self.winner.value if self.winner is not None else None
        d["probe"] = self.probe.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JudgmentRecord:
        d = dict(d)
        choice = d.get("choice")
        winner = d.get("winner")
        d["choice"] = PairwiseChoice(choice) if choice is not None else None
        d["winner"] = Winner(winner) if winner is not None else None
        d["probe"] = Probe(d["probe"]) if d.get("probe") is not None else Probe.NEUTRAL
        return cls(**d)


@dataclass
class JudgmentSet:
    """All judgments from one audit, plus the context needed to analyze them."""

    mode: JudgeMode
    model: str
    records: list[JudgmentRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def for_example(
        self,
        example_id: str,
        rubric_variant: int = 0,
        probe: Probe = Probe.NEUTRAL,
    ) -> list[JudgmentRecord]:
        return [
            r
            for r in self.records
            if r.example_id == example_id
            and r.rubric_variant == rubric_variant
            and r.probe is probe
        ]

    @property
    def example_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.records:
            seen.setdefault(r.example_id, None)
        return list(seen)

    @property
    def parse_failure_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(not r.parse_ok for r in self.records) / len(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "model": self.model,
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JudgmentSet:
        return cls(
            mode=JudgeMode(d["mode"]),
            model=d["model"],
            records=[JudgmentRecord.from_dict(r) for r in d["records"]],
        )

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> JudgmentSet:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
