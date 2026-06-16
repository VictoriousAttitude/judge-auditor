"""Judge Auditor — measure how much your LLM-as-judge disagrees with itself.

Audits the *evaluator*, not the model under test: self-consistency, position
/ verbosity bias, scale compression, and the statistical power (noise floor) of
a judge setup.
"""

from __future__ import annotations

from .config import AuditConfig, EvalExample, JudgeConfig, JudgeMode, PairwiseChoice, Winner
from .records import JudgmentRecord, JudgmentSet

__version__ = "0.1.0"

__all__ = [
    "AuditConfig",
    "EvalExample",
    "JudgeConfig",
    "JudgeMode",
    "JudgmentRecord",
    "JudgmentSet",
    "PairwiseChoice",
    "Winner",
]
