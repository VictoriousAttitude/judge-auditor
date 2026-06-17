"""Reproducible judge-comparison demo (no API key, fully seeded).

Audits five synthetic judges of known character on the same kind of eval set and
prints a Markdown comparison table — the same artifact you would get by pointing
``judge-audit`` at five real judge models, but deterministic so it runs in CI and
reproduces byte-for-byte. Swap the synthetic generators for real backends to
produce the table for GPT-4o / Claude / Llama / a local model.

    python examples/judge_comparison.py
"""

from __future__ import annotations

from judge_auditor import synthetic as S
from judge_auditor.analysis.audit import ReliabilityReport, audit
from judge_auditor.report.comparison import render_comparison_markdown


def _audited() -> list[tuple[str, ReliabilityReport]]:
    judges = [
        ("Reliable judge", S.scalar_judge(icc=0.90, quantize=True, seed=2)),
        ("Noisy judge", S.scalar_judge(icc=0.20, quantize=True, seed=3)),
        ("Compressed-scale judge", S.compressed_scalar_judge(seed=4)),
        ("Consistent judge", S.consistent_pairwise_judge(seed=8)),
        ("Position-biased judge", S.pairwise_judge_with_first_rate(1.0, seed=6)),
    ]
    return [(label, audit(js, exs)) for label, (js, exs) in judges]


def main() -> None:
    print("Judge reliability on a shared eval set (synthetic, reproducible):\n")
    print(render_comparison_markdown(_audited()))


if __name__ == "__main__":
    main()
