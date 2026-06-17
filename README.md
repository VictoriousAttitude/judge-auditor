# judge-auditor

[![CI](https://github.com/VictoriousAttitude/judge-auditor/actions/workflows/ci.yml/badge.svg)](https://github.com/VictoriousAttitude/judge-auditor/actions/workflows/ci.yml)

**Before trusting any LLM evaluation, measure how much your evaluator disagrees with itself. That self-disagreement is your noise floor, and only differences above it are real.**

`judge-auditor` is a reliability diagnostic for LLM-as-judge pipelines. Point it at
*your* judge (model + rubric + eval examples) and it produces a report of the judge's
failure modes with proper statistics:

- **Self-consistency** — does the judge agree with itself across repeated runs? (ICC for scalar scoring, Fleiss' κ for pairwise, with bootstrapped confidence intervals)
- **Position bias** — do verdicts flip when you swap response order?
- **Verbosity bias** — does the judge reward length instead of quality?
- **Scale compression** — is the judge really using its score range, or clustering in 2–3 bins?
- **Statistical power** — how large must a real quality difference be before your judge can detect it? (the *noise floor*)

## What this is **not**

- Not an eval framework. It does not run your evals — use whatever you already use.
- Not a judge implementation. You bring your rubric and scoring prompt; it audits them.
- Not a benchmark. It measures *your* setup, because paper-level bias findings don't transfer across models, rubrics, and temperatures.

It sits one meta-level above the eval pipeline: *"I don't run your evals — I tell you if your evals are trustworthy."*

## Status

Early development.

- ✅ **Runner** (Layer 0): backend-agnostic judge runner — repeated runs, position swapping, response parsing, bounded-concurrency async, checkpoint/resume. Supports pairwise and scalar modes. OpenAI / OpenAI-compatible and mock backends.
- 🚧 **Analysis** (Phase 2): consistency, position/verbosity bias, scale analysis, power/noise-floor — in progress.
- ⏳ **Report + CLI** (Phase 3) and **validation + methodology** (Phase 4).

## Install (from source)

```bash
git clone https://github.com/VictoriousAttitude/judge-auditor
cd judge-auditor
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart (runner)

Collect repeated judgments from a judge with position swapping built in. This uses the
mock backend (no API key); swap in `OpenAIBackend` to audit a real judge.

```python
import asyncio
from judge_auditor import AuditConfig, EvalExample, JudgeConfig, JudgeMode
from judge_auditor.runner import JudgeExecutor
from judge_auditor.runner.backends.mock import MockBackend

config = JudgeConfig(
    model="my-judge",
    mode=JudgeMode.PAIRWISE,
    prompt_template=(
        "Question: {prompt}\n"
        "Response A: {response_a}\nResponse B: {response_b}\n"
        "Which is better? Answer [[A]], [[B]], or [[C]] for a tie."
    ),
)
examples = [
    EvalExample(id="1", prompt="2+2?", response_a="4", response_b="four-ish"),
]

backend = MockBackend(lambda messages, cfg: "[[A]]")
executor = JudgeExecutor(backend, config, AuditConfig(runs_per_example=10))
judgments = asyncio.run(executor.run(examples))

print(len(judgments), "judgments collected")
print("parse failure rate:", judgments.parse_failure_rate)
```

The analysis modules (which turn collected judgments into the reliability report) are
landing next.

## License

MIT
