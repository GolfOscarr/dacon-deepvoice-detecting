# 03 — Decision Protocol

How an idea gets promoted or killed. Written down in advance so that a disappointing result
cannot be re-litigated after the fact.

## 1. Order of work 🔴

★ `[BirdCLEF playbook 2026]` E8: **`1) data split → 2) sampler/loss → 3) architecture →
4) optimizer`**. *"Common mistake: over-searching schedules before solving data shift and
imbalance."*

Concretely, for this competition: the split ([01](01-split-scheme.md)) and the corpus composition
([data/02](../data/02-label-taxonomy.md)) are worth more than any architecture choice, and the
optimizer schedule is worth close to nothing until the other three are settled. Budget attention
in that order.

Corollary (★ E9): **tune loss, sampler and augmentation as a package**, not one at a time — they
interact, and a sampler change that looks bad under the old loss can be the best option under the
new one.

## 2. Three speeds

★ E6: *"the best teams usually operate multiple experimental speeds: replay, medium CV, full final
validation."* Triage cheap, confirm expensive.

| Speed | Scope | Budget | Used for |
|---|---|---|---|
| **Replay** | fold 0 only, fixed seed, subsampled VAL, short schedule | ≤20 min | Killing ideas. Most ideas die here |
| **Medium** | all 5 folds, 1 seed, full VAL | ≤3 h | The default. Everything quotable runs at least here |
| **Full** | all 5 folds × 3 seeds, + SHADOW | ≤12 h | Promotion candidates only, and the final selection |

Replay results are **never quotable** — they are a filter, not evidence. An idea that survives
replay is re-run at Medium before it enters the ledger.

⚠️ Replay has a known failure mode: it systematically favours ideas that help early in training.
Anything schedule-, EMA- or checkpoint-related must skip Replay and start at Medium.

## 3. The promotion rule

An idea is promoted only if **all** of:

| # | Criterion | Threshold |
|---|---|---|
| P1 | Paired bootstrap CI of the **difference** in Score excludes 0 | 95%, 10k resamples, resampled **by group** (artifact_family), not by file |
| P2 | No **cell** regresses | no cell's EER worsens by >2 pts |
| P3 | No **artifact family** collapses | no family's EER goes above 0.45 (i.e. near-chance) |
| P4 | Fold variance does not grow materially | sd across folds ≤1.3× the incumbent's |
| P5 | Runtime margin holds | projected ≤3.0 s/file on L4, with ≥20% headroom |
| P6 | All gates pass | [VG1–VG6](04-audit-gates.md) green |

**P1 is a paired test by construction** — same folds, same seeds, same evaluation composition
seed. [02 §4](02-metric-harness.md#pairing-is-what-makes-small-gains-measurable-) shows why this
matters: unpaired comparison at our VAL size resolves a 1-point gain only 85% of the time; paired,
95%.

⚠️ **Resample by group, not by file.** Bootstrapping individual files treats 40 clips from one
Jamendo artist as 40 independent observations and produces CIs that are far too narrow. Resample
`artifact_family` (and `source_name` within it) with replacement.

### The tiebreaker

★ E5: *"a model with slightly lower mean but lower fold variance is often a better final
candidate."* When P1 is inconclusive (CI includes 0), prefer, in order: **lower fold variance →
better worst-cell EER → better SHADOW score → lower runtime**. Never "the higher public LB".

## 4. Decision hierarchy 🔴

★ E1: **OOF predictions, not public-LB scores, decide whether an idea survives.**

```
1. OOF mean over folds            ← the decision
2. Fold and seed stability
3. Subgroup robustness            ← per cell, per family, SHADOW
4. Runtime margin
5. Public LB                      ← weak cross-check ONLY
```

The public LB enters exactly two ways:
- as a **consistency check** — if VAL and LB move in opposite directions, something is wrong with
  the *split*, and the split gets fixed (it is not evidence that VAL is "too pessimistic");
- as the **per-head decomposition** in [05](05-lb-probe-plan.md), run on a fixed budget.

It never selects between two candidates. With Private = Public and 3 submissions/day for 22 days,
LB-driven selection over 1,200 samples is straightforward overfitting, and it is precisely what
the [2nd-stage rubric](../competition/03-evaluation.md#5-second-stage-judging-종합-평가-100-points)
is designed to discount — 70 of 100 points are the reports.

## 5. Experiment ledger

One append-only row per Medium-or-better run, `experiments.parquet`. Doubles as raw material for
the 2nd-stage **모델 개발** section (25 pts), which is scored on *"학습 및 성능 개선 과정의
체계성"* — the systematicity of the improvement process, i.e. exactly this table.

```
exp_id · timestamp · git_sha · parent_exp_id · hypothesis(one sentence)
scheme_version · corpus_version · eval_seed · speed(replay|medium|full)
config_path · train_seeds
score_mean · score_sd · ads · cps
eer_file · eer_voice · eer_music · auc_vp · auc_mp        (mean over folds)
per_cell_eer(json) · per_family_eer(json) · per_fold_score(json)
t3_pair_eer · sung_eer · spoken_eer · snr_bucket_eer(json)
shadow_score · shadow_delta
vg1..vg6(pass|fail|na) · runtime_s_per_file
verdict(promoted|killed|inconclusive) · note
```

Rules:
- **`hypothesis` is written before the run**, not after. A run without a prior hypothesis is
  exploration and is logged as `verdict=inconclusive` regardless of its score.
- ★ *"Every branch should answer one question only."* Two changes in one run means the run cannot
  attribute the effect and is not promotable.
- Killed ideas stay in the ledger. The negative results are report material and stop us
  rediscovering the same dead end.

## 6. Loss selection under a ranking metric

★ E8/F8: *"the best loss may be the one that yields the strongest OOF ordering rather than the
nicest raw probabilities."* Since EER and AUC are rank metrics
([02 §5](02-metric-harness.md#5-what-the-metrics-are-and-are-not-invariant-to)), calibration is worth **zero** points directly.
Select losses on OOF EER, and treat any calibration work as serving cross-condition score
*comparability* ([Phase E](../../PROGRESS.md)), not the metric.

➕ Worth testing late: ★ `[G2Net 2021, 3rd]` **rank-loss fine-tune at low LR for ~2 epochs** at the
end, reported ~1 bps. Cheap, and aligned with the metric by construction.

## 7. Stopping rules

| Situation | Action |
|---|---|
| Presence AUC ≥ 0.99 on a generator- and source-disjoint split | **Stop investing.** Worth 0.05 each ([data/03](../data/03-presence-data.md)) |
| An idea is inconclusive at Full speed twice | Kill it. Do not run a third time hoping for the CI to move |
| VAL and SHADOW disagree on candidate ordering | Neither promotes. Investigate the condition axis that flipped it |
| VAL improves while LB stalls across ≥3 probes | Suspect the split, not the LB. Re-run [VG2/VG3](04-audit-gates.md) |
