# Methodology

This document explains the statistics behind every number `judge-auditor` reports:
what is computed, why that estimator was chosen, and where the limits are. The
guiding idea is simple: **a judge that disagrees with itself sets a noise floor, and
no downstream evaluation can resolve a difference smaller than that floor.** Every
metric below either measures that self-disagreement or measures a systematic bias
that inflates it.

All headline metrics carry a confidence interval, because a single point estimate at
audit-scale sample sizes (n = 50-200 examples) is not interpretable on its own.

## Data model

The runner collects `K` repeated judgments per example. For pairwise judges the `K`
runs are split between the `A,B` and `B,A` orderings, so position bias is measured
from data we already have rather than a separate experiment. Each judgment keeps both
the parsed verdict and the raw response; parse failures are recorded, never dropped,
because a high parse-failure rate is itself a reliability signal.

The resampling unit throughout is the **example** (with all `K` of its runs kept
together). This respects the hierarchical structure of the data: runs within an
example are exchangeable, examples are the independent units.

## Self-consistency

### Scalar judges: ICC

For numeric scores we report the **intraclass correlation**, computed from one-way
and two-way ANOVA mean squares (pure NumPy, no `pingouin` dependency).

- **ICC(1,1), one-way random effects — the headline.** The `K` repeated runs of a
  single judge are *exchangeable*: run #3 on example A shares no identity with run #3
  on example B, so there is no crossed "rater" factor to model. The one-way model is
  the statistically correct choice for repeated runs of one judge.
- **ICC(2,1), two-way random effects, absolute agreement — reported alongside** for
  readers expecting the Shrout & Fleiss form. It converges with ICC(1,1) when there is
  no systematic run-position effect (which there is not, by exchangeability), so a
  large gap between the two is itself a diagnostic.

Both use **absolute agreement**, not Pearson correlation: a judge that always scores
two points high has `r = 1` but `ICC < 1`, and we want to catch that.

Interpretation bands follow Koo & Li (2016): `< 0.50` poor, `< 0.75` moderate,
`< 0.90` good, otherwise excellent.

### Pairwise judges: Fleiss' kappa

For A/B/tie verdicts we report **Fleiss' kappa** on the canonical (content-resolved)
winners. Kappa chance-corrects the raw agreement and needs only the per-example
category counts, not rater identity — again exactly right for exchangeable runs.
Interpretation bands follow Landis & Koch (1977): `< 0` poor, `< 0.20` slight,
`< 0.40` fair, `< 0.60` moderate, `< 0.80` substantial, otherwise almost perfect.

## Validity (vs ground truth)

Self-consistency measures *precision* — whether the judge agrees with itself. It says
nothing about *accuracy*: a judge can score every answer two points too low, or pick
the wrong winner every time, with perfect self-consistency. When you supply ground
truth (`quality_label` for scalar, `preferred_winner` for pairwise) the audit also
measures whether the judge agrees with the **truth**.

- **Scalar:** Pearson correlation (with a bootstrap CI) and Spearman rank correlation
  between the per-example mean judge score and the ground-truth quality label.
- **Pairwise:** Cohen's kappa (chance-corrected agreement, with a bootstrap CI), the
  raw agreement rate, and the accuracy excluding ties, between the judge's majority
  winner and the preferred winner.

Interpretation bands for the correlation/kappa follow the same magnitude convention:
`< 0.30` poor, `< 0.50` weak, `< 0.70` moderate, otherwise good.

Validity is reported only when at least two labeled examples are present, and stays
entirely silent (no verdict impact) otherwise, so users without ground truth see no
change. The flag is deliberately **not** a significance test: the worst judges have a
validity indistinguishable from zero, which a `p < 0.05` gate would let pass. Instead
the judge is flagged when the *upper* bound of the validity CI is below the moderate
threshold (Pearson `r` or Cohen's kappa) on at least `min_n = 8` labeled examples —
i.e. we can confidently rule out acceptable validity. A flagged judge downgrades the
overall verdict just like any other detected problem.

## Rubric robustness (vs paraphrasing)

A judge's verdict should reflect the response, not the incidental wording of the
rubric. When you supply one or more paraphrased rubrics (same intent, different
phrasing), the runner fans every example out across all variants and stamps each
judgment with its `rubric_variant`. This section asks whether the verdict survives
the paraphrase. If it does not, the judge is **brittle**: part of its signal is an
artifact of prompt phrasing rather than response quality.

Two design choices keep this honest:

- **Headline metrics use the canonical rubric only.** All within-judge metrics
  (self-consistency, the biases, scale, power) are computed on `rubric_variant == 0`
  alone, so cross-rubric disagreement does not leak into — and inflate — the
  self-consistency noise floor. For a single-rubric audit this is a no-op.
- **Agreement is measured on per-example *aggregates*.** We first collapse each
  example's `K` runs under a given variant to one number (mean score, or majority
  winner), then compare *across* variants. Averaging out within-variant run noise
  first means what remains is the systematic effect of phrasing.

- **Scalar:** cross-variant **ICC(2,1)** (two-way, absolute agreement) on the
  per-example × per-variant mean-score matrix, with a bootstrap CI resampling
  examples. Here the two-way form is the *correct* one — unlike the exchangeable
  repeat runs, a rubric variant is a genuine identifiable crossed factor (variant 1
  is the same rephrasing on every example). We also report the mean and max
  per-example score spread (max−min across the per-variant means).
- **Pairwise:** cross-variant **Fleiss' kappa** on the per-example majority winner
  under each variant (bootstrap CI), plus the **winner-flip rate** — the fraction of
  examples whose majority winner is not unanimous across variants.

Robustness is reported only when the audit ran at least two rubric variants and is
otherwise entirely silent (no verdict impact), exactly like validity without ground
truth. The flag uses the same confidently-rule-out logic: it fires when the bootstrap
CI's upper bound stays below the robust bar (ICC `< 0.75`, Fleiss' kappa `< 0.60`) on
at least `min_n = 8` complete examples. A flagged judge downgrades the verdict to
`MODERATE`, or to `LOW` when the point estimate is below the low bar (ICC `< 0.50`,
kappa `< 0.40`) — i.e. the verdict is barely phrasing-independent at all.

## Position bias (pairwise)

Two complementary views, both from the same `A,B` / `B,A` runs:

- **First-position preference rate.** Among non-tie decisions, how often the judge
  picks the first-*presented* response. A fair judge sits at 0.5. We test the
  deviation with an exact binomial test against 0.5 and report a Wilson score
  interval. This is the most statistically powerful position signal because it
  aggregates every decision.
- **Flip rate.** The fraction of examples whose *majority* canonical winner differs
  between the two orderings — the practical "how often is my verdict unstable under
  swapping" number from Zheng et al. (2023).

A high first-position rate *and* a high flip rate is systematic order bias; a high
flip rate with a ~0.5 first rate is just noise.

## Probe sensitivity (sycophancy and anchoring)

The biases above are *observational* — read off the verdicts the judge produced
unprompted. This section is *interventional*: it whispers a suggestion into the prompt
and measures how far the verdict follows. Both probes are opt-in (`--probe-sycophancy`,
`--probe-anchoring`), off by default so a plain audit makes no extra calls.

- **Sycophancy** prepends a stated user opinion (scalar: "this deserves a high/low
  score"; pairwise: "Response X is clearly better"). A judge that caves is rewarding
  agreement with the user, not response quality.
- **Anchoring** prepends an irrelevant numeric reference ("a previous reviewer scored
  this 10/10"). A judge that drifts toward it is swayed by a number carrying no signal.
  Anchoring needs a numeric scale, so it is scalar-only.

Each probe is collected as a balanced **up/down** pair: the runner re-judges every
example under both directions, at the canonical rubric only (`rubric_variant == 0`) to
keep the call count bounded. The effect is the per-example **swing** between the two
directions — `mean(up) − mean(down)` — averaged over examples. Differencing the two
directions cancels any per-example constant (the judge's baseline opinion of that
response), so what remains is the causal pull of the injected suggestion, free of the
response-quality confound.

- **Scalar:** the swing of the mean score, normalized to a fraction of the score range
  ("moved X% of the scale"), with a bootstrap CI resampling examples. The same math
  serves sycophancy and anchoring (only the injected text differs).
- **Pairwise:** the swing of the win rate for content A (ties counted as half), in
  `[−1, 1]`. The sycophancy cue is phrased against the *presented* label that holds the
  targeted content, so it stays a content suggestion orthogonal to position.

Unlike validity (which fires when it can *rule out* a good correlation), a bias flag
uses a **significance gate**: it fires only when the bootstrap CI excludes a *zero*
swing in the suggested direction (`effect.low` above the threshold) on at least
`min_n = 8` examples. A swing indistinguishable from zero is no evidence of bias. A
flagged probe downgrades the verdict to `MODERATE`, or to `LOW` when the swing is large
(scalar `> 15%` of scale; pairwise `> 25` win-rate points).

## Verbosity bias

Does the judge reward length instead of quality?

- **Scalar:** Spearman rank correlation between score and response length (whitespace
  word count by default; precomputed lengths accepted). When ground-truth
  `quality_label`s are supplied we also report the **partial** Spearman correlation of
  score and length controlling for quality, which separates "length tracks quality"
  from "the judge rewards length per se."
- **Pairwise:** the win rate of the longer response, and the rank correlation between
  length difference and win.

A judge is flagged when `|rho|` exceeds the threshold (default 0.30).

## Scale analysis

Is the judge actually using its range?

- **Scalar:** scores are binned to integers. We report the number of distinct values
  used, the **effective dynamic range** via normalized entropy
  (`entropy / log(num_bins)`), and the **densest-window share** (the largest fraction
  of scores falling in any few adjacent bins). A judge is flagged *compressed* when
  that share exceeds the threshold (default 0.70) — e.g. "92% of scores are 6, 7, or 8"
  means a 1-10 scale is really a 3-point scale.
- **Pairwise:** the win-A / win-B / tie split; a high tie rate marks the judge as
  unable to distinguish the responses.

## Statistical power and the noise floor

This is what turns the metrics into a decision rule.

### Scalar: minimum detectable effect

We estimate the pooled within-example standard deviation `sigma_w` (the judge's
measurement noise) from the repeated runs, then report the minimum detectable effect
for a two-group comparison of `n` examples per group at the requested power and alpha:

    MDE(n) = (z_{alpha/2} + z_{beta}) * sigma_w * sqrt(2 / n)

**This is a lower bound (a floor).** It counts only judge noise and ignores the
between-example variance a real comparison also carries, so the true MDE is at least
this large. We state that explicitly rather than overclaim. Score differences smaller
than the MDE are within judge noise — do not ship decisions based on them.

### Pairwise: discriminability and the detectable margin

The judge's self-consistency gives an effective per-pair accuracy `a` (the modal-
verdict probability). Its **discriminability** `2a - 1` attenuates any true win-rate
margin, so the minimum detectable *true* margin over `n` pairs is:

    MDE_margin(n) = (z_{alpha/2} + z_{beta}) * 0.5 / sqrt(n) / (2a - 1)

A judge that flips half the time (`a -> 0.5`) has zero discriminability and an
**infinite** noise floor: no number of pairs can rescue it. (This is exactly what the
"Position-biased judge" row of the comparison table shows.)

Both modes also answer the inverse question — given a target effect, how many
examples or pairs are required.

## Confidence intervals

- **Bootstrap (ICC, kappa, flip rate as a complex statistic):** nonparametric
  percentile bootstrap, resampling *examples* with replacement (default 2000
  replicates, seeded). Resampling examples — not individual runs — is what keeps the
  CI honest about the hierarchical structure. Degenerate replicates that return NaN are
  dropped before taking percentiles.
- **Wilson score interval (proportions):** for first-position rate and flip rate. It
  has better small-sample coverage than the Wald interval and stays well-behaved near
  0 and 1.

## Overall verdict

The verdict (`LOW` / `MODERATE` / `HIGH`) is deliberately conservative. It starts from
the headline self-consistency band (ICC for scalar, kappa for pairwise) and is only
ever **downgraded** by a detected bias or scale problem — never upgraded. An unreliable
signal cannot be rescued by the *absence* of one particular bias, so the worst finding
dominates.

## Limitations

- The scalar noise floor is a lower bound, as stated above; treat it as a necessary,
  not sufficient, threshold.
- Bias findings do not transfer across models, rubrics, and temperatures. This tool
  measures *your* setup; it is not a benchmark and does not claim population-level
  bias rates.
- Verbosity length uses a word-count proxy, not a tokenizer; supply precomputed token
  lengths if you need token-exact correlations.
- Partial-correlation control for verbosity requires ground-truth `quality_label`s.
- Validity requires ground truth (`quality_label` / `preferred_winner`); without it the
  audit measures only precision, not correctness, and the validity section is silent.
- Rubric robustness requires at least two rubric variants; without them the section is
  silent and the headline metrics run on the single rubric as before.
- Probe sensitivity (sycophancy / anchoring) is opt-in and adds judge calls; the probes
  measure susceptibility to the *specific* cues injected here, not every persuasion
  vector. **Self-enhancement bias** (a judge favoring its own authorship) is not yet
  probed — it needs a per-response authorship data model — and is a known gap.
- Kappa and ICC for self-consistency assume the `K` runs are exchangeable, so the
  headline metrics are computed on the canonical rubric (`rubric_variant == 0`) alone;
  cross-rubric disagreement is measured separately by the robustness section rather
  than folded into the noise floor.

## Validation

The estimators are validated end-to-end against synthetic judges with known,
constructed properties (`judge_auditor/synthetic.py`, `tests/test_validation.py`):

- **Calibration:** scores generated with a known ICC are recovered within the
  confidence interval; a pairwise set built with an exact 15% majority-flip rate is
  reported as ~15%; a known first-position rate is recovered.
- **Known-bias:** a maximally position-biased judge is detected (100% flip, favored
  first, verdict `LOW`); a high-noise judge is flagged `LOW`.
- **Null:** a self-consistent, unbiased judge reports high consistency, no flagged
  biases, a small noise floor, and verdict `HIGH`.
- **Validity:** a judge generated to be self-consistent yet uncorrelated with the
  ground truth (scalar `rho = 0`, or pairwise accuracy 0.5) is flagged on validity and
  downgraded to `LOW` despite a clean reliability profile; a judge built to track the
  truth (`rho = 0.9`) keeps its `HIGH` verdict.
- **Rubric robustness:** a judge built to ignore the paraphrase (sensitivity 0) keeps
  its `HIGH` verdict with no flag; one whose verdict is fully driven by phrasing
  (scalar sensitivity 1, or every pairwise winner flipping on the alternate rubric) is
  flagged brittle and downgraded to `LOW`; a pairwise set constructed with an exact
  flip fraction recovers that winner-flip rate; and an audit with a single rubric
  leaves the section silent and the verdict untouched.
- **Probe sensitivity:** a judge built to swing its score by a known fraction of the
  scale under the sycophancy / anchoring cue recovers that swing; a pairwise judge that
  complies on an exact fraction of examples recovers that win-rate swing; the verdict
  downgrades (to `LOW` for a large swing) while a probe-free or zero-swing audit leaves
  the section silent.

## References

- Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
  (arXiv:2306.05685) — position and verbosity bias, agreement rates.
- Wang et al. (2023), *Large Language Models are not Fair Evaluators*
  (arXiv:2305.17926) — position bias and calibration.
- Shrout & Fleiss (1979), *Intraclass correlations: uses in assessing rater
  reliability* — ICC forms.
- Koo & Li (2016), *A Guideline of Selecting and Reporting Intraclass Correlation
  Coefficients for Reliability Research* — ICC interpretation bands.
- Landis & Koch (1977), *The Measurement of Observer Agreement for Categorical Data* —
  kappa interpretation bands.
