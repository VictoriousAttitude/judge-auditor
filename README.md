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

## The judge comparison table

The single most useful output: audit several judges on the same eval set and see how
their reliability differs. This table is produced by `python examples/judge_comparison.py`
from synthetic judges of known character (deterministic, no API key) — point the same
machinery at real backends to compare GPT-4o / Claude / Llama / a local model.

| Judge | Mode | Self-consistency | Position flip | Verbosity | Scale | Noise floor | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reliable judge | scalar | ICC 0.86 (good) | n/a | ok | full range | 0.44 pts | HIGH |
| Noisy judge | scalar | ICC 0.20 (poor) | n/a | ok | full range | 0.54 pts | LOW |
| Compressed-scale judge | scalar | ICC 0.74 (moderate) | n/a | ok | compressed | 0.11 pts | MODERATE |
| Consistent judge | pairwise | kappa 1.00 (almost perfect) | 0% | ok | ok | 13% margin | HIGH |
| Position-biased judge | pairwise | kappa -0.07 (poor) | 100% | ok | ok | no power | LOW |

Same eval set, very different trustworthiness: the position-biased judge flips its
verdict on every example and has *no* power to detect any true win-rate margin, while
the reliable judge resolves score differences down to ~0.44 points.

## Sample report

The terminal report for the position-biased judge above (`-f terminal`, abridged):

```
================================================================
 JUDGE RELIABILITY: LOW
================================================================
 Model: synthetic-position    Mode: pairwise
 Examples: 60    Judgments: 960    Parse failures: 0.0%

SELF-CONSISTENCY (pairwise)
  Fleiss' kappa:  -0.067 [-0.067, -0.067]  [poor]
  Mean agreement: 0.500   (min 0.500, median 0.500)

POSITION BIAS
  First-position rate: 1.000 [0.996, 1.000]   (p=0.000, favors: first)
  Flip rate:           1.000 [0.940, 1.000]   (60/60 examples)
  Tie rate:            0%

POWER / NOISE FLOOR (pairwise)
  Effective accuracy:   0.500  (discriminability 0.000)
  Min detectable margin at n=60: inf

FLAGS
  - Position bias toward the first-presented response (flip rate 100%).
================================================================
```

`-f html` produces the same content as a single self-contained HTML file (embedded
CSS, no external JS or CDN); `-f json` produces a machine-readable version with an
actionable `recommendations` list.

## What this is **not**

- Not an eval framework. It does not run your evals — use whatever you already use.
- Not a judge implementation. You bring your rubric and scoring prompt; it audits them.
- Not a benchmark. It measures *your* setup, because paper-level bias findings don't transfer across models, rubrics, and temperatures.

It sits one meta-level above the eval pipeline: *"I don't run your evals — I tell you if your evals are trustworthy."*

## Status

**v0.1, pre-1.0.** The analysis core is tested and cross-validated against reference implementations, but the project is young: install is from source (not yet on PyPI), the public API may change before 1.0, and the OpenAI / Anthropic backends are covered by mocked and failure-injection tests rather than a live-API integration test. Use it, file issues — just pin a commit if you depend on it.

- **Runner** (Layer 0) — done: backend-agnostic judge runner with repeated runs, position swapping, response parsing, bounded-concurrency async, checkpoint/resume. Supports pairwise and scalar modes. OpenAI / OpenAI-compatible, Anthropic, and mock backends.
- **Analysis** (Phase 2) — done: consistency (ICC / Fleiss' kappa), position and verbosity bias, scale analysis, power / noise-floor, all with bootstrapped confidence intervals.
- **Report + CLI** (Phase 3) — done: the `judge-audit` CLI with terminal, self-contained HTML, and JSON reports, plus an actionable recommendations engine.
- **Validation + methodology** (Phase 4) — done: end-to-end calibration / known-bias / null validation against synthetic judges, the comparison table above, and [METHODOLOGY.md](METHODOLOGY.md).
- **Testing** — a reliability tool has to be reliable itself, so the estimators carry their own evidence: the hand-rolled statistics are cross-validated against `statsmodels` / `pingouin` and a published worked example (Shrout & Fleiss 1979); their mathematical invariants are fuzzed with property-based tests; the detectors are checked by Monte-Carlo calibration (point-estimate unbiasedness, confidence-interval coverage, and false-alarm / sensitivity rates); backend retry and error handling are exercised by failure injection; the terminal / HTML / JSON reports are pinned by golden snapshots; and CI enforces a coverage floor.

## Install (from source)

```bash
git clone https://github.com/VictoriousAttitude/judge-auditor
cd judge-auditor
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart (CLI)

The `judge-audit` CLI runs a full audit and emits a report. The `mock` backend needs
no API key and makes no network calls, so you can dry-run your config and template
wiring before spending money on a real judge.

```bash
# Dry-run against the bundled example (mock backend, no API key):
judge-audit run -c examples/judge.toml -e examples/examples.jsonl -b mock -k 15

# Audit a real judge and write a self-contained HTML report:
export OPENAI_API_KEY=sk-...
judge-audit run -c examples/judge.toml -e examples/examples.jsonl \
    -b openai -k 15 -f html -o report.html

# Save the raw judgments, then re-render later without re-calling the judge:
judge-audit run -c judge.toml -e examples.jsonl -b openai \
    --save-judgments judgments.json
judge-audit report -j judgments.json -e examples.jsonl -f json
```

A judge config is TOML or JSON; eval examples are JSONL or a JSON array. For pairwise
judges set `mode = "pairwise"` and provide `response_b` on each example. Reports come in
three formats (`-f terminal | html | json`) and can be written to a file with `-o`.

Non-`mock` runs print an estimated call count and prompt for confirmation before
spending anything. As a rough guide, auditing 50 examples at `K = 15` is ~750 judge
calls (on the order of a few dollars with a frontier model); use `--sample` to audit a
stratified subset and `--checkpoint` to resume an interrupted run without paying twice.

## Quickstart (library)

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

To turn collected judgments into a reliability report in code, pass the `JudgmentSet`
and your examples to `judge_auditor.analysis.audit.audit(...)` and render the result
with one of the `judge_auditor.report` renderers.

## Methodology

Every metric — the ICC form choice, Fleiss' kappa, the position-bias binomial test,
the bootstrap CIs, and the noise-floor derivation (including its lower-bound caveat) —
is documented in [METHODOLOGY.md](METHODOLOGY.md), along with the tool's limitations
and the validation that proves it recovers known effects.

## License

MIT
