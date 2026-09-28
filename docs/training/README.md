# Training Objective (Phase D)

What we optimize, and in what order. [`pipelines/`](../pipelines/README.md) delivers the batch;
this directory says what the loss does with it.

| File | Contents |
|---|---|
| [01-what-the-metric-demands.md](01-what-the-metric-demands.md) | ⭐ The derivation — why the objective is what it is, and why most of it is not free |
| [02-the-loss.md](02-the-loss.md) | ⭐ **The committed spec**, term by term, with two changes to shipped defaults |
| [03-ruled-out.md](03-ruled-out.md) | Every technique investigated, with the measured number that closed it |
| [04-schedule.md](04-schedule.md) | S1–S3, why S4 is dropped, and the four experiments |
| [05-corrections.md](05-corrections.md) | Claims elsewhere in the repo this evidence overturns |
| [06-tier-list.md](06-tier-list.md) | ⭐ **What to implement, in order** — S/A/B/X tiers for objectives and training methods |

**Runs (2026-09-24 → 09-29), in order:**

| File | Contents |
|---|---|
| [07-first-run-plan.md](07-first-run-plan.md) | Run 1 plan (strategy-v3) |
| [08-ko-synth-report.md](08-ko-synth-report.md) | Synthesised Korean fake voice (track K) |
| [09-data-synthesis-plan.md](09-data-synthesis-plan.md) | Round-2 synthesis priorities |
| [10-run1-diagnosis.md](10-run1-diagnosis.md) | Where run 1's errors are |
| [11-run2-design.md](11-run2-design.md) | Run 2 design (strategy-v4) |
| [12-run2-results.md](12-run2-results.md) | Run 2 results, LB 0.81144 |
| [13-run3-plan.md](13-run3-plan.md) | Run 3 plan (superseded by 15–17) |
| 14-ddp-plan.md | DDP + XLS-R-1B plan (on branch `feat/ddp-1b`) |
| [15-data-plan.md](15-data-plan.md) | strategy-v5 data plan |
| [16-lessons-learned.md](16-lessons-learned.md) | ⭐ What we learned, runs 1–3 and strategy-v5 |
| [17-final-runs.md](17-final-runs.md) | ⭐ **Final runs**: 1B, runs A–D, strategy-v6/v6b/v6c, soups, ensembles, final packages |

## The finding, in one paragraph

🔴 **The objective is close to a solved problem here, and the literature's headline gains do not
transfer.** Four research axes — ranking/AUC surrogates, one-class and margin losses,
multi-resolution frame supervision, noise-robust losses — produced **nothing** above our local
resolution threshold of ≈1 EER point — let alone the ±1.7 / ±2.5 pt leaderboard noise floor. The reason is structural rather than accidental: anti-spoofing and
audio-tagging report minDCF, Cllr, actDCF, Macro-F1 and accuracy, all threshold- or
calibration-sensitive, and **EER and ROC-AUC are invariant to monotone transforms** — for focal loss
this is a *theorem*, not an observation. The one finding that cleared the floor is not about which
loss but about **where losses attach**: supervising utterance and frame level through one shared
head measured **0.71–3.63 EER points worse** than utterance-only, and our shipped head is that
configuration.

## Consequences

| | |
|---|---|
| `clip_weight` | 0.5 → **1.0** (clip-only), frame term becomes ablation **T1** |
| `LossConfig.weights` | uniform → **metric-proportional** (.45/.27/.18/.05/.05) |
| `ranking_weight` | stays **0** — TFPARN's own ablation moves EER 12.91 → 12.92 |
| Stage **S4** | 🔴 **dropped** — frees engineer-days |
| Focal / margin / one-class / label smoothing | ❌ not adopted |
| New sampler constraints | **C1** positive rate ∈ [0.2, 0.8] · **C2** present-count floor · **C3** `f₅=f₆=f₇=f₈` |

## Non-negotiables

- 🔴 **Check which metric produced a number before adopting it.** Quoting an ACC/F1/minDCF headline
  as if it were an EER result is the single easiest way to adopt something worth zero
- 🔴 **Masked losses mirror the masked EER pools** — not an optimization, a requirement
- 🔴 **We choose our class balance**, so we remove BCE's imbalance pathology in the sampler rather
  than buying a loss term to survive it
- ⚠️ **Most objective knobs are undecidable for us.** ★ *"A question we cannot resolve is not an
  experiment — it is a decision to make by argument and leave alone"*
  ([09 §B](../architecture/09-open-questions.md))
- Never let a loss term undo the non-saturating output map ([architecture/01 §3.2](../architecture/01-design-envelope.md#32-ranking-only-scoring-and-the-saturation-trap-))

## Status

| | |
|---|---|
| Objective derived and committed | ✅ [01](01-what-the-metric-demands.md), [02](02-the-loss.md) |
| Evidence base | ✅ deep read, 4 axes, ~20 papers; [papers/09](../papers/09-training-losses.md) promotion criterion discharged |
| Schedule | ✅ [04](04-schedule.md) |
| Sampler constraints C1/C2/C3 | ✅ landed in [`pipelines/02 §4`](../pipelines/02-sampler.md#4--constraints-the-objective-imposes--c1-and-c2) and [`pipelines/05`](../pipelines/05-invariants.md) (I8, I9) |
| Corrections | ✅ **all applied** — [05](05-corrections.md) is now the audit trail. `C-2` and `D-11` changed on inspection |
| Anything trained | ⬜ **no** |
