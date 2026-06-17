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
- Kappa and ICC assume the `K` runs are exchangeable. If you deliberately vary the
  rubric across runs, analyze each variant separately.

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
