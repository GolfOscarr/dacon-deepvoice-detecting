# 05 — Combining Multiple Models

Ensembling is well supported by the evidence for this task and **permitted by the rules** — but the
naive Kaggle form of it is ❌ forbidden, and the obvious form of it is 🔴 unsound under a ranking
metric. Both traps are specific and avoidable. This file is how we combine models without falling
into either.

★ The evidence that it is worth doing at all:

| Source | Finding |
|---|---|
| ★ ASVspoof 5 organizers | *"submissions using an ensemble of sub-systems tend to perform better"* |
| ★ AT-ADD 2026, both tracks | *"Structured fusion, not naive averaging"* is listed as a cross-cutting pattern across both tracks' top-5; the Track 1 winner ran a **3-member W2V-BERT ensemble** |
| ★ AT-ADD runners-up | Multi-scale XLSR (0.3B/1B/2B) score fusion · per-window **median** + logistic-regression fusion · weighted logit fusion with CQCC cross-attention branches |
| ☆/★ Kaggle, across competitions | *"The ensemble aggregation function is a hyperparameter"* — ☆ `min()` `[BirdCLEF 2024, 1st]` · ★ geometric mean `[Freesound 2019, 1st]` · ★ CMA-ES logit blend `[G2Net 2021, **3rd**]` · ★ 4-of-13 voting `[Cornell 2020, 1st]`. ⚠️ Not all are winners' choices, and not all are primary-sourced |

---

## 1. Two traps, stated first

### 🔴 Trap 1 — rank averaging is forbidden

The standard Kaggle ensemble is **rank-average across the test set**: convert each model's scores
to ranks over all test rows, then average. It is scale-free, robust, and ❌ **illegal here**.

> 금지: **다른 파일 샘플의 정보·예측값·통계 등을 활용**하여 예측값을 생성하거나 보정하는 방식
> — rule 2.4 ([competition/04](../competition/04-rules.md))

Ranking file *i* requires the scores of files *j ≠ i*. So does score standardization over the
cohort, and so does any adaptive blend weight computed from the test distribution. All out.

✅ **What remains legal is everything that is a pure function of one file**: logit averaging,
weighted logit fusion, geometric mean, min/max, a fixed learned combiner. The blend *weights* may
absolutely be fitted — on **our own validation data**, offline, and frozen into the shipped model.
That is not test-set information.

### 🔴 Trap 2 — probability averaging across models is not scale-safe

This one is subtler and follows from the metric itself.

EER is invariant to monotone transforms, so **each model is free to sit on its own arbitrary
monotone scale** and still score identically alone. Model A may output 0.98 where model B outputs
0.55 for equally confident FAKE calls. Averaging them is then dominated by whichever model happens
to use more of the [0,1] range — and 🔷 the average of two well-ordered scores on different scales
is not guaranteed to be well-ordered.

**The fix, and it is legal:** fit a **monotone calibration per model on our own validation pool**
(isotonic or Platt), freeze it into the weights, and fuse *after* it. Calibration is the thing
ranking metrics let us skip for a single head and force us to do the moment we fuse.

⚠️ This is the same cross-condition comparability problem that appears in
[01 §3.4](01-design-envelope.md#34--overlapping-classes-and-one-pooled-ranking) and in the file-head
construction ([04](04-heads-and-pooling.md#5-the-file-head)). Three different places, one underlying
cause: **ranking metrics hide scale mismatch until you combine something.**

---

## 2. The taxonomy, priced

Not all "combining" costs the same. Ours is an **inference-bound** design
([01 §2](01-design-envelope.md#2-the-compute-ceiling)), so price is the first sort key.

| Level | What it is | Inference cost | Zip cost | Verdict |
|---|---|---|---|---|
| **Weight-space** | Average the weights of several checkpoints ("checkpoint soup") | **zero** | **zero** | ⭐ **Free. Do it by default** |
| **Distillation** | Train an ensemble, ship one student that imitates it | **zero** | **zero** | ⭐ **Our main answer** — [§4](#4-the-main-answer-ensemble-at-training-time-ship-one-model) |
| **Input-level (TTA)** | Same model, several crops/views, pooled within the file | ×crops | zero | ✅ Already required for coverage ([04 §6](04-heads-and-pooling.md#6-temporal-coverage-tiling-not-sampling)) |
| **Feature-level** | Several frontends → one head | ×frontends (shared head) | ×frontends | ✅ Already in candidate B — this is what two specialist frontends *is* |
| **Score-level** | Several full models → combiner | ×models | ×models | ⚠️ Only with measured headroom |
| **Decision-level** | Voting over hard labels | ×models | ×models | ❌ Discards the ranking information the metric scores |

🔴 **Decision-level fusion is disqualified by the metric.** Voting produces a coarse, heavily tied
score; EER and ROC-AUC read the full ordering. Never vote here.

---

## 3. The free tier — take this first

**Checkpoint soup.** ☆ Averaging weights across epochs instead of early-stopping is credited to
`[BirdCLEF 2024, 2nd]` ([kaggle/05 F5](../kaggle/05-transferable-playbook.md)) — secondary-sourced,
and a runner-up's choice rather than a winner's. It produces an
ensemble-like effect at **exactly zero inference and packaging cost**, which given our budget makes
it the highest ratio of value to price available. 🔷 With 8×H200 we can soup widely across
epochs and data orders.

⚠️ Weight averaging only works between checkpoints in the same loss basin — same architecture, same
run or same init. **So it does not extend to different random inits**, and an earlier draft of this
file wrongly suggested souping across seeds. Different seeds are an *ensemble* input
([§5](#5-diversity-is-what-pays-not-count)), not a soup input. Souping is for epochs and
same-init runs; it is not a substitute for combining genuinely different models.

**Multi-crop pooling** is already mandatory for coverage, so its cost is booked against the label
semantics, not against ensembling. Note the aggregation choice differs by head: ★ AT-ADD's
runner-up used per-window **median** for robustness, while our labels want `frame_max`. 🔷 Both,
per head, decided by CV.

---

## 4. The main answer: ensemble at training time, ship one model

🔴 **This is the design consequence of 8×H200 training and one L4 inference.**

The published ensembles we would most like to copy are precisely the ones we cannot run. ★ The
AT-ADD runner-up's multi-scale XLSR fusion (0.3B + 1B + 2B) is assessed as **"likely over budget"**
in [survey/05](../survey/05-models.md), and three large frontends also strain the 10 GB zip.

But nothing stops us from **running that ensemble as a teacher** and shipping a student that
imitates it:

```
training (8×H200, unconstrained)          inference (L4, ~10× real-time)
──────────────────────────────            ──────────────────────────────
XLS-R-2B      ─┐                          truncated speech frontend  ─┐
XLS-R-1B      ─┤                                                      ├→ 3 branches → 5 heads
SSLAM / EAT-L ─┼→ frozen teacher ensemble truncated audio frontend   ─┘
Demucs v4 res.─┘   ↓ stop-gradient MSE
                 student  ────────────────────────────────────────────→ this is what ships
```

☆ The mechanism is measured: embedding distillation with a **stop-gradient** gave 0.898 vs 0.876
in the Kaggle notebook that documents it, and the notebook explains why the stop-gradient matters:

> *"Without the stop-gradient, the classification loss and distillation loss would fight over the
> backbone's feature representation."*
> — [kaggle/06 §3](../kaggle/06-notebook-code.md)

🔷 **The reframing:** ensembling is normally a way to spend inference compute for accuracy. For us
it is a way to spend *training* compute for accuracy, because that is the resource we have. A
teacher ensemble costs us nothing we are short of.

Mechanics, precision and the risks are in [06](06-compression.md).

---

## 5. Diversity is what pays, not count

★ *"Ensemble only models that are both strong and different. Three nearly identical models rarely
beat two diverse ones. Check OOF correlation."*
([kaggle/05 G5](../kaggle/05-transferable-playbook.md))

🔷 Ranked by how *different* the member is from our candidate-B student, which is the axis that
matters — not by its solo score:

| Axis | Member | Why it is different |
|---|---|---|
| **Training objective** | One-class / real-only member (candidate F) | Never sees a generator, so it cannot overfit one. Our binding constraint is cross-generator (46.4% EER published) |
| **Input band** | A low-band-only model (0–4 kHz) | ★ Low-frequency subbands cut EER **up to 25% relative** under codecs (D9), and our telephone slice lives there. Genuinely different evidence, not a different seed |
| **Representation** | Handcrafted spectral ⊕ SSL | ★ CtrSVDD baselines: raw waveform 13.75% and **LFCC 16.15%** both far ahead of mel 25.19% — LFCC is a real, cheap, different view |
| **Window length** | Short-window vs long-window | Localization vs structure ([04 §6](04-heads-and-pooling.md)) |
| **Frontend** | Speech SSL vs general-audio SSL | Already inside B as feature-level fusion |
| **Seed** | Same everything, different seed | ⚠️ Cheapest and **least** valuable. Soup it instead of ensembling it |

⚠️ Note what is *not* on this list: a second graph-attention backend on the same features. ★ The
survey rates backbone choice ⭐⭐ against frontend ⭐⭐⭐⭐⭐, and backbone innovation moves In-the-Wild
EER about **1 point per year**. Two backbones over one frontend is close to one model twice.

---

## 6. Choosing the combiner

The Kaggle finding is that the combiner is a hyperparameter, not a constant: ☆ `min()` cut false
positives in one competition, ★ geometric mean was the winner's choice in another, and ★ a
CMA-ES-optimized logit blend placed 3rd in a third.

🔷 For our metric specifically, two candidates stand out:

- **Weighted logit fusion**, weights fitted on our validation pool by direct **EER/AUC
  optimization** rather than by log-loss. ☆ There is literature for exactly this — *Ensemble
  Learning for AUC Maximization via Surrogate Loss* and *Online AUC Optimization via Second-order
  Surrogate Loss* ([papers/09](../papers/09-training-losses.md)). Fitting a combiner on log-loss
  optimizes calibration we are not scored on.
- **Per-head weights.** Nothing requires one blend across all five columns. The music head and the
  voice head have different members' strengths, and the columns are scored independently.

★ Start with weighted blending before any stacker — *"a fragile stacker can fail under time
pressure"* ([kaggle/05 G6](../kaggle/05-transferable-playbook.md)). With 22 days, that warning has
force.

⚠️ **Every combiner weight is fitted offline and frozen.** Anything fitted at inference time on the
test cohort is a rule-2.4 violation, however innocuous it looks.

---

## 7. The decision rule

🔷 Ensembling competes with **temporal coverage** for the same budget, and we think coverage wins
first — a member that re-examines the same 33% of a 60 s file cannot find a fake segment in the
other 67% ([04 §6](04-heads-and-pooling.md#6-temporal-coverage-tiling-not-sampling)).

Spend the inference budget in this order:

1. **Full temporal coverage** of every file. Non-negotiable; it is a label-semantics requirement.
2. **Weight-space and distillation combining** — free, so it is never traded against anything.
3. **Two specialist frontends** (feature-level) — already candidate B.
4. **Then**, and only against *measured* headroom from [07](07-runtime-budget.md), score-level
   members from the diversity table, most-different first. 🔷 The corrected budget
   ([07 §2](07-runtime-budget.md#2-estimated-line-items-)) leaves ~3.7–10× margin, so one extra
   member is plausibly affordable — which makes measuring worth doing rather than a formality.

⚠️ Two hard stops on tier 4: the **10 GB zip** binds before the runtime does for large frontends
([01 §1.2](01-design-envelope.md#12-everything-loads-offline-from-model)), and every extra
component is another offline-packaging failure mode. ★ *"A model that cannot survive notebook
deployment is a teacher candidate, not a final submission candidate"*
([kaggle/05 G8](../kaggle/05-transferable-playbook.md)) — which, read against §4, is less a
restriction than an instruction: **demote it to teacher and keep the accuracy.**

★ Finally, keep two submission tracks — a conservative proven blend and a higher-upside
experimental one ([kaggle/05 G9](../kaggle/05-transferable-playbook.md)). ⚠️ With Private = Public
and 3 submissions/day, the conservative track must always be the one currently on the board.
