"""``judge-audit`` command-line interface.

Two commands:

* ``run`` — execute a full audit against a judge backend and emit a report.
* ``report`` — re-render a report from previously-saved raw judgments (no re-run).

The ``mock`` backend needs no API key and makes no network calls: it is both the CI
smoke-test driver and a way for users to dry-run their config/template wiring before
spending money on a real judge.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import typer

from .analysis.audit import ReliabilityReport, audit
from .config import AuditConfig, EvalExample, JudgeConfig, JudgeMode
from .records import JudgmentSet
from .report.html import render_html
from .report.json_report import render_json
from .report.terminal import render_terminal
from .runner.backends.mock import MockBackend
from .runner.executor import JudgeExecutor
from .runner.protocol import JudgeBackend

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Measure how much your LLM-as-judge disagrees with itself before you trust it.",
)


# --- loaders --------------------------------------------------------------------


def load_examples(path: Path) -> list[EvalExample]:
    """Load eval examples from a JSONL file or a JSON array/{"examples": [...]}."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        rows = data if isinstance(data, list) else data.get("examples", [])
    examples: list[EvalExample] = []
    for row in rows:
        examples.append(
            EvalExample(
                id=str(row["id"]),
                prompt=row["prompt"],
                response_a=row["response_a"],
                response_b=row.get("response_b"),
                quality_label=row.get("quality_label"),
                metadata=row.get("metadata", {}),
            )
        )
    return examples


def load_judge_config(path: Path) -> JudgeConfig:
    """Load a judge config from TOML or JSON (optionally nested under [judge])."""
    if path.suffix == ".toml":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    cfg: dict[str, Any] = data.get("judge", data)
    return JudgeConfig(
        model=cfg["model"],
        prompt_template=cfg["prompt_template"],
        mode=JudgeMode(cfg["mode"]),
        system_prompt=cfg.get("system_prompt"),
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 1024),
        response_format=cfg.get("response_format"),
        score_min=cfg.get("score_min", 1.0),
        score_max=cfg.get("score_max", 10.0),
    )


# --- backends -------------------------------------------------------------------


def _mock_responder(messages: list[dict[str, str]], config: JudgeConfig) -> str:
    """Deterministic, key-free judge for dry-runs and CI smoke tests."""
    content = messages[-1]["content"]
    h = int(hashlib.md5(content.encode("utf-8")).hexdigest(), 16)
    if config.mode is JudgeMode.PAIRWISE:
        return ("[[A]]", "[[B]]", "[[C]]")[h % 3]
    lo, hi = int(config.score_min), int(config.score_max)
    return f'{{"score": {lo + h % (hi - lo + 1)}}}'


def build_backend(name: str, base_url: str | None) -> JudgeBackend:
    """Construct a backend by name; raises on bad input or a missing API key."""
    if name == "mock":
        return MockBackend(_mock_responder)
    if name == "openai":
        from .runner.backends.openai import OpenAIBackend

        return OpenAIBackend(base_url=base_url) if base_url else OpenAIBackend()
    if name == "anthropic":
        from .runner.backends.anthropic import AnthropicBackend

        return AnthropicBackend(base_url=base_url) if base_url else AnthropicBackend()
    raise typer.BadParameter(f"unknown backend {name!r} (use: mock, openai, anthropic)")


# --- rendering ------------------------------------------------------------------


def _render(report: ReliabilityReport, fmt: str) -> str:
    if fmt == "terminal":
        return render_terminal(report)
    if fmt == "json":
        return render_json(report)
    if fmt == "html":
        return render_html(report)
    raise typer.BadParameter(f"unknown format {fmt!r} (use: terminal, html, json)")


def _emit(report: ReliabilityReport, fmt: str, out: Path | None) -> None:
    text = _render(report, fmt)
    if out is not None:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote {fmt} report to {out}")
    else:
        typer.echo(text)


async def _execute(
    executor: JudgeExecutor, backend: JudgeBackend, exs: list[EvalExample]
) -> JudgmentSet:
    try:
        return await executor.run(exs)
    finally:
        aclose = getattr(backend, "aclose", None)
        if aclose is not None:
            await aclose()


# --- commands -------------------------------------------------------------------


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", "-c", exists=True, help="Judge config (.toml/.json)"
    ),
    examples: Path = typer.Option(
        ..., "--examples", "-e", exists=True, help="Eval examples (.jsonl/.json)"
    ),
    backend: str = typer.Option("mock", "--backend", "-b", help="mock | openai | anthropic"),
    runs: int = typer.Option(15, "--runs", "-k", help="Runs per example (K)"),
    sample: int | None = typer.Option(
        None, "--sample", help="Audit a stratified sample of M examples"
    ),
    concurrency: int = typer.Option(10, "--concurrency", help="Max concurrent judge calls"),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", help="JSONL checkpoint path (resume)"
    ),
    fmt: str = typer.Option("terminal", "--format", "-f", help="terminal | html | json"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the report to this file"),
    save_judgments: Path | None = typer.Option(
        None, "--save-judgments", help="Save raw judgments JSON"
    ),
    base_url: str | None = typer.Option(None, "--base-url", help="Override the API base URL"),
    seed: int = typer.Option(0, "--seed", help="Sampling seed"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the cost confirmation prompt"),
) -> None:
    """Run a full audit against a judge and emit a reliability report."""
    judge_cfg = load_judge_config(config)
    exs = load_examples(examples)
    audit_cfg = AuditConfig(
        runs_per_example=runs,
        max_concurrency=concurrency,
        sample_size=sample,
        seed=seed,
        checkpoint_path=str(checkpoint) if checkpoint else None,
    )
    be = build_backend(backend, base_url)
    executor = JudgeExecutor(be, judge_cfg, audit_cfg)

    n_calls = executor.count_tasks(exs)
    if backend != "mock" and not yes:
        typer.echo(
            f"About to make {n_calls} judge calls to '{backend}' (model {judge_cfg.model})."
        )
        if not typer.confirm("Proceed?"):
            typer.echo("Aborted.")
            raise typer.Exit(1)

    js = asyncio.run(_execute(executor, be, exs))
    if save_judgments is not None:
        js.save_json(str(save_judgments))

    report = audit(js, exs, score_min=judge_cfg.score_min, score_max=judge_cfg.score_max)
    _emit(report, fmt, out)


@app.command()
def report(
    judgments: Path = typer.Option(
        ..., "--judgments", "-j", exists=True, help="Saved judgments JSON"
    ),
    examples: Path = typer.Option(
        ..., "--examples", "-e", exists=True, help="Eval examples (.jsonl/.json)"
    ),
    fmt: str = typer.Option("terminal", "--format", "-f", help="terminal | html | json"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the report to this file"),
    score_min: float = typer.Option(1.0, "--score-min", help="Scalar scale minimum"),
    score_max: float = typer.Option(10.0, "--score-max", help="Scalar scale maximum"),
) -> None:
    """Re-render a report from previously-saved raw judgments (no judge calls)."""
    js = JudgmentSet.load_json(str(judgments))
    exs = load_examples(examples)
    rep = audit(js, exs, score_min=score_min, score_max=score_max)
    _emit(rep, fmt, out)
