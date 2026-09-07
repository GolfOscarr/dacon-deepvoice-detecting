# 02 — Metrics and How We Measure Them

Two questions, answered in order: **which numbers do we track**, and **how is each one computed
so that it means what we think it means**.

Everything here is implemented once, in **`metrics/dacon.py`**
([implementation plan](06-implementation-plan.md)). No experiment computes its own EER.

---

## 1. The metric register

Every number we are allowed to quote. Anything not on this list is exploration, not evidence.

### Tier 1 — Official (what we are scored on)

Fixed by the competition ([competition/03](../competition/03-evaluation.md)). We do not get to
choose these.

| Metric | Definition | Pool | Weight in Score |
|---|---|---|---|
| `EER_file` | EER of `FILE_FAKE_PROB`, FAKE = 1 | all files | **0.45** |
| `EER_music` | EER of `MUSIC_FAKE_PROB` | **music-present only** | **0.27** |
| `EER_voice` | EER of `VOICE_FAKE_PROB` | **voice-present only** | **0.18** |
| `AUC_vp` | ROC-AUC of `VOICE_PRESENT_PROB` | all files | **0.05** |
| `AUC_mp` | ROC-AUC of `MUSIC_PRESENT_PROB` | all files | **0.05** |
| `ADS` | `0.5(1−EER_file) + 0.2(1−EER_voice) + 0.3(1−EER_music)` | — | 0.9 |
| `CPS` | `0.5·AUC_vp + 0.5·AUC_mp` | — | 0.1 |
| **`Score`** | `0.9·ADS + 0.1·CPS` | — | **the number** |

### Tier 2 — Decision (these promote or kill an idea)

Derived from Tier 1, and defined here because *how* they are derived is what makes them
trustworthy ([§4](#4-how-we-aggregate), [§6](#6-confidence-intervals)).

| Metric | Definition | Used for |
|---|---|---|
| `Score_mean` | mean over the 5 VAL folds | The headline local number |
| `Score_sd` | sd over the 5 VAL folds | ★ E5 tiebreaker — lower variance wins a tie |
| `Δ_paired` | candidate − incumbent, same folds/seeds/composition | **[P1](03-decision-protocol.md#3-the-promotion-rule)** — the promotion test |
| `CI95(Δ)` | group bootstrap over `artifact_family`, 10k resamples | P1's decision boundary |
| `worst_cell_eer` | max over cells 1–9 of that cell's file EER | **P2** — no cell may regress |
| `worst_family_eer` | max over artifact families | **P3** — no family may collapse |
| `shadow_delta` | `Score(VAL) − Score(SHADOW-a)`, paired via `shadow_of` | Domain-shift robustness |
| `runtime_s_per_file` | wall clock ÷ 1,200, on L4 | **P5** — hard budget 3.0 s |

### Tier 3 — Diagnostic (these say *where* we are weak)

Never a promotion criterion on their own. They decide what to work on next.

| Metric | Breakdown | Reads |
|---|---|---|
| `per_cell_eer` | cells 1–9 ([taxonomy](../data/02-label-taxonomy.md)) | Are cells 6/7 actually learned, or have the fake heads entangled? |
| `per_family_eer` | artifact family | Feeds [`E-A9`](../data/07-eda-plan.md) difficulty ranking |
| `per_fold_score` | fold 0–4 | Instability, and which family group is the outlier |
| `t3_pair_eer` | T3 matched pairs only | **The corpus-identity control** — see [VG4](04-audit-gates.md#vg4--corpus-identity-leakage) |
| `sung_eer` / `spoken_eer` | voice subtype | [R5](../data/09-risks-and-checks.md) sung-voice gap |
| `snr_bucket_eer` | component gain bucket | Detection tracks stem energy; shows the real failure region |
| `duration_bucket_eer` | 4–10 / 10–30 / 30–60 s | The test range's extremes |

### Tier 4 — Guardrail (these invalidate a run)

Not measures of quality — measures of whether the other numbers mean anything.
Thresholds pre-committed in [04](04-audit-gates.md).

| Metric | Gate |
|---|---|
| `shortcut_auc` | **< 0.60** — metadata-only logistic regression ([VG2](04-audit-gates.md#vg2--shortcut-audit)) |
| `adversarial_auc` | **< 0.60** — train-vs-VAL classifier ([VG3](04-audit-gates.md#vg3--adversarial-validation)) |
| `t3_gap` | `t3_pair_eer − pooled_eer` **≤ 0.10** ([VG4](04-audit-gates.md#vg4--corpus-identity-leakage)) |
| `n_unique_ratio` | **> 0.5** per output column ([VG5](04-audit-gates.md#vg5--output-sanity)) |
| `probe_openings` | **≤ 3** for the competition ([VG6](04-audit-gates.md#vg6--probe-budget)) |

### What we deliberately do **not** track

| Not tracked | Why |
|---|---|
| Accuracy, F1, precision/recall | Threshold metrics. We are scored on ranking; a threshold metric would make us optimize the wrong thing. ★ The broadcast study is the cautionary case: F1 fell 0.992 → 0.186 while AUC fell only 0.998 → 0.775 |
| Calibration error (ECE, Brier) | Worth **zero** points — EER and AUC are rank metrics ([§5](#5-what-the-metrics-are-and-are-not-invariant-to)). Calibration matters only for cross-condition score *comparability* in [Phase E](../../PROGRESS.md) |
| Training loss | Not a metric. Never quoted as evidence |
| Public LB score | Tracked, but as a **cross-check only** ([03 §4](03-decision-protocol.md#4-decision-hierarchy-)) |

---

## 2. The official definitions

Transcribed from [competition/03](../competition/03-evaluation.md).

```python
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def eer(y_true, y_score):
    """Official EER. FAKE is the positive class (1)."""
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2
```

⚠️ Three details that change the number:

| Detail | Why it matters |
|---|---|
| `drop_intermediate=False` | The default `True` prunes ROC vertices; the `argmin` then lands on a different point |
| `pos_label=1` with **FAKE = 1** | Inverting the class inverts the score ordering |
| `argmin`, **not** interpolation | The official code picks the nearest grid point rather than solving `fpr = fnr`. Most textbook EER implementations interpolate and will disagree slightly |

## 3. The masked pools

> ※ Voice EER은 음성이 존재하는 샘플에서만, Music EER은 음악이 존재하는 샘플에서만 계산됩니다.

**The mask uses the organizers' ground-truth presence labels, not our predictions.** Our presence
errors cannot corrupt the fake EERs. In cell terms
([taxonomy](../data/02-label-taxonomy.md#the-8-cells)): the voice pool is cells **1,2,5,6,7,8**
and the music pool is cells **3,4,5,6,7,8**.

```python
def dacon_score(df):
    """df: ground truth + our 5 prediction columns, one row per file."""
    v = df.voice_present.values.astype(bool)     # ground truth presence
    m = df.music_present.values.astype(bool)

    eer_file  = eer(df.file_fake.values,     df.FILE_FAKE_PROB.values)
    eer_voice = eer(df.voice_fake.values[v], df.VOICE_FAKE_PROB.values[v])
    eer_music = eer(df.music_fake.values[m], df.MUSIC_FAKE_PROB.values[m])

    auc_vp = roc_auc_score(df.voice_present.values, df.VOICE_PRESENT_PROB.values)
    auc_mp = roc_auc_score(df.music_present.values, df.MUSIC_PRESENT_PROB.values)

    ads = 0.5*(1-eer_file) + 0.2*(1-eer_voice) + 0.3*(1-eer_music)
    cps = 0.5*auc_vp + 0.5*auc_mp
    return dict(score=0.9*ads + 0.1*cps, ads=ads, cps=cps,
                eer_file=eer_file, eer_voice=eer_voice, eer_music=eer_music,
                auc_vp=auc_vp, auc_mp=auc_mp)
```

File ground truth is `voice_fake OR music_fake` **over present components**, taken from the cell
table — never from our own heads.

---

## 4. How we aggregate

The single most error-prone step, because EER is **not** a per-sample average and therefore does
not behave like accuracy under pooling.

### 🔴 Never pool raw OOF scores across folds

Each fold is scored by a *different model*. Their score scales differ, and EER is computed on the
merged ranking, so concatenating raw OOF scores measures the scale drift, not the models.
Measured on five folds of a model with true EER 0.100, each fold's scores rescaled by a
harmless monotone `a·s + b`:

| Aggregation | Result |
|---|---|
| Mean of per-fold EER | **0.1040** (sd 0.0056) ✅ |
| EER on pooled **raw** OOF scores | **0.1705** ❌ — 64% too high |
| EER on pooled **per-fold rank-normalized** scores | **0.1040** ✅ |

The size of the error depends on how far the per-fold scales drift, so 0.1705 is one draw, not a
constant — but the *direction* is not a draw. Pooling raw scores can only add apparent error,
never remove it, because the drift is independent of the label.

**Rule**: the reported metric is the **mean of per-fold metrics**, always reported with its sd.
Pooled-OOF is permitted only after per-fold rank normalization, and only for diagnostics that need
a single ranked list (error analysis, score histograms) — never for the headline number.

⚠️ Per-fold rank normalization is legal *locally*. It is a cross-file statistic and therefore
❌ forbidden inside `script.py` ([rule 2.4](../competition/04-rules.md)).

### The roll-up order does not matter ✅

`Score` is **linear** in the five component metrics, so `mean(Score per fold)` and
`Score(mean of each component metric)` are identical to floating-point precision (verified). Use
either; report both the component means and the Score so a reader can recompute.

This is only true of the **roll-up**. It is not true of the underlying EERs, which is the whole
point of the rule above.

### Small-pool folds

A fold whose masked pool falls below the [size floor](01-split-scheme.md#4-size-floors) is
reported but **excluded from the mean**, with the exclusion recorded. Averaging in a fold with 200
music-present files adds noise, not information — and VG1 A8 should have caught it first.

---

## 5. What the metrics are and are not invariant to

Knowing the invariances is what lets us compare a local number to anything else.

### ✅ Invariant to strictly monotone transforms

EER and ROC-AUC depend only on the ordering of scores. `x³`, `logit(x)`, any affine rescale — all
free. **Consequence**: calibration earns nothing, and score-shaping work should be spent on
ordering instead.

### ✅ Invariant to class prevalence

The ROC curve is built from the within-class score distributions, so changing the fake:real ratio
does not move EER. Verified at ratios 1:1 → 1:0.05 (variation stayed within sampling noise).
**Consequence**: we do not need to guess the test set's fake:real ratio
([taxonomy](../data/02-label-taxonomy.md)).

### ❌ **Not** invariant to composition *within* a class 🔴

This is the one that bites, and it is easy to conflate with the point above. Holding the class
ratio fixed at 1:1 and varying only the mix of *cells inside the FAKE class*:

| Share of the hard cell in the FAKE class | EER |
|---:|---:|
| 0% | 0.0337 |
| 25% | 0.1313 |
| 50% | 0.1987 |
| 75% | 0.2610 |
| 100% | 0.2970 |

A 9× swing with no change to the model. Three consequences:

1. **The evaluation composition must be frozen.** VAL is composed on the fly
   ([06](../data/06-augmentation-spec.md)), so the cell mix, SNR distribution and sequential-variant
   rate are pinned by a fixed `eval_seed` and recorded in the
   [experiment ledger](03-decision-protocol.md#5-experiment-ledger). Two runs with different
   `eval_seed` are not comparable.
2. **Local EER will not equal LB EER**, and that is not a bug. The test set's cell mix is unknown.
   Only *differences measured on the same composition* transfer.
3. **A composition change is a corpus change**, so it re-triggers [VG2 and VG3](04-audit-gates.md)
   and resets comparability with earlier ledger rows.

### ❌ Not invariant to tie structure

See [§7](#7-verified-properties). Ties are ranking information destroyed.

---

## 6. Confidence intervals

A metric without an interval is not a decision input.

| Quantity | Procedure |
|---|---|
| `CI95(metric)` | Group bootstrap, 10k resamples, resampling **`artifact_family`** with replacement |
| `CI95(Δ_paired)` | Same resamples applied to *both* candidates, difference taken **inside** each resample |
| Fold variance | Reported directly as sd over 5 folds — not bootstrapped |

⚠️ **Resample groups, not files.** Bootstrapping individual files treats 40 clips from one Jamendo
artist as 40 independent observations and yields intervals that are far too narrow. The resampling
unit is `artifact_family`, with `source_name` resampled within it.

⚠️ **Take the difference inside the resample.** The paired structure is the entire reason a
1-point gain is detectable — see [§7](#pairing-is-what-makes-small-gains-measurable-). Comparing
two independently-computed marginal CIs discards it and would have us throw away real gains.

---

## 7. Verified properties

Established by simulation, not argument. Reproduce all of it with
`.venv/bin/python scripts/verify_metric.py`.

### An all-constant submission scores exactly 0.5000 ★

`roc_curve` on a single distinct score emits exactly the vertices `(0,0)` and `(1,1)`, so
`|fpr − fnr| = [1, 1]`, `argmin` returns index 0, and `EER = (0 + 1)/2 = 0.5`. `roc_auc_score` on
constant scores is 0.5 by definition. Therefore

```
Score = 0.9 × (0.5×0.5 + 0.2×0.5 + 0.3×0.5) + 0.1 × (0.5×0.5 + 0.5×0.5) = 0.45 + 0.05 = 0.5000
```

Confirmed for constants 0.0, 0.123, 0.5 and 1.0. This is the
[first LB probe](05-lb-probe-plan.md#p0-is-the-highest-value-single-submission-in-the-competition):
if it does not return exactly 0.5000, our reading of the metric is wrong.

### EER sampling noise — the real precision limit

1,000 replications, Gaussian score model, balanced classes. 95% CI half-width, in EER points:

| Samples per class | true EER 0.20 | 0.10 | 0.05 |
|---:|---:|---:|---:|
| 150 | ±4.5 | ±3.4 | ±2.6 |
| 300 | ±3.2 | ±2.5 | ±1.8 |
| 600 | ±2.3 | ±1.7 | ±1.2 |
| 1,200 | ±1.5 | ±1.2 | ±0.9 |
| 3,000 | ±1.0 | ±0.7 | ±0.5 |

The estimator is **unbiased** — recovered means matched the true EER to within 0.002 at every size.

### Pairing is what makes small gains measurable 🔴

Two models on the **same** VAL slice produce correlated errors. Resolving a true 1.0-point gap
(0.100 vs 0.090), 400 replications:

| n/class | correlation ρ | sd of the *difference* | P(correct sign) |
|---:|---:|---:|---:|
| 600 | 0.0 | 0.0119 | 75% |
| 600 | 0.8 | 0.0083 | 86% |
| 1,200 | 0.0 | 0.0085 | 85% |
| **1,200** | **0.8** | **0.0059** | **95%** |
| 1,200 | 0.95 | 0.0044 | >99% |

This is the statistical basis for [§6](#6-confidence-intervals) and for the promotion rule.

### Score saturation is the one output-formatting risk that matters ⚠️

Rounding submitted probabilities is **harmless** — at true EER 0.0950, rounding to 6/4/3/2/1
decimal places gave 0.0950 / 0.0950 / 0.0950 / 0.0942 / 0.0958. Recorded as a negative finding so
nobody spends effort on output precision.

What *is* dangerous is saturation that collapses ranking **near the operating point**:

| Top/bottom fraction saturated | EER |
|---:|---:|
| 0% (baseline) | 0.0950 |
| 10% | 0.0950 |
| 30% | 0.0950 |
| 50% | 0.0950 |
| **80%** | **0.3017** |

Harmless until the saturated region reaches the EER threshold, then a 3× blow-up with no warning.
**Rule**: emit logits, convert with a float64 sigmoid, never round or clip, and assert
`n_unique > 0.5 × n` per column ([VG5](04-audit-gates.md#vg5--output-sanity)).

❌ The obvious fix — rank-normalizing each column to `(rank + 0.5)/n` — is **not** available in the
submission. [Rule 2.4](../competition/04-rules.md) requires per-file independence and names
*"rank calibration"* across the eval cohort among the forbidden operations. Use it in the local
harness; never in `script.py`.

## 8. Unit tests

`tests/test_metrics.py` — known-answer, no fixtures needed:

| # | Case | Expected |
|---|---|---|
| T1 | Constant column, any value | EER exactly 0.5, AUC exactly 0.5 |
| T2 | All five columns constant | Score exactly 0.5 |
| T3 | Perfect separation | EER 0.0, AUC 1.0, Score 1.0 |
| T4 | Perfectly inverted | EER 1.0, AUC 0.0, Score 0.0 |
| T5 | Monotone transform (`x`, `x³`, `logit(x)`) | Score unchanged to 1e-12 |
| T6 | Garbage `VOICE_FAKE_PROB` on music-only rows | Score unchanged |
| T7 | Row order shuffled | Score unchanged |
| T8 | Hand-computed 5-row fixture | Matches the hand calculation |
| T9 | `drop_intermediate=True` variant | **Differs** — guards the default creeping back |
| T10 | Single-head probe → recovered metric | Matches to 1e-12 ([05](05-lb-probe-plan.md#the-algebra)) |
| T11 | Fold aggregation with per-fold rescaling | Mean-of-folds correct; pooled-raw **wrong** ([§4](#4-how-we-aggregate)) |
| T12 | Class ratio 1:1 vs 1:0.2, same distributions | EER unchanged within tolerance |
| T13 | Empty masked pool (no music-present rows) | Raises, does not silently return 0.5 |
| T14 | Presence label all one class | Raises — AUC undefined |

## 9. Version pinning

The evaluation server runs **scikit-learn 1.8.0** on **Python 3.11.15**
([competition/02](../competition/02-submission.md#preinstalled-packages-do-not-pin-different-versions)).
`roc_curve`'s tie handling and vertex set are implementation details we depend on.

⚠️ Verification for this document ran on scikit-learn **1.9.0** / numpy 2.5.3. Re-run
`tests/test_metrics.py` under 1.8.0 before trusting a number to the 4th decimal.
