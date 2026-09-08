# 02 — The Loss

The committed objective. [01](01-what-the-metric-demands.md) derives it; [03](03-ruled-out.md)
records what was eliminated and by which measurement.

---

## 1. The spec

```
L = Σ_b  w_b · masked_mean( BCE(clip_logits_b, y_b), present_b )
  + α  · MSE(distill_emb, teacher_emb.detach())          # stage-gated, see §7
```

| Knob | Committed value | Change from today |
|---|---|---|
| `SEDHeadConfig.clip_weight` | **1.0** (clip only) | ⚠️ was 0.5 — §3 |
| `LossConfig.weights` | **file .45 · music .27 · voice .18 · v_pres .05 · m_pres .05** | ⚠️ was all 1.0 — §4 |
| `LossConfig.ranking_weight` | **0.0** | unchanged (§5) |
| `LossConfig.label_smoothing` | **0.0** | unchanged (§5) |
| `LossConfig.distill_weight` | α = 1.0, stage-gated | unchanged (§7) |

🔴 **It is notable mainly for what it does not contain**: no frame term, no ranking term, no focal
loss, no margin or one-class loss, no label smoothing, no learned task weighting. Each absence is
an entry in [03](03-ruled-out.md) with a measured number behind it, not an omission.

⚠️ **This is deliberately close to the simplest loss that respects the masks.** That is the
conclusion of the evidence, not a failure to look: four research axes produced nothing above our ≈1 pt
local resolution threshold ([01 §3](01-what-the-metric-demands.md#3--the-epistemic-constraint-most-objective-questions-are-undecidable-for-us)),
and the one finding that cleared it is about *where losses attach*
(§3), not which loss.

---

## 2. Masked BCE on clip logits — the whole objective

★ Each component loss is masked to the files where that component is present, mirroring the masked
EER pools exactly ([01 §2.1](01-what-the-metric-demands.md)). Implemented via `BranchConfig.masked_by`
in [`models.losses.multitask_loss`](../../models/AGENTS.md); this is PC-Mix's scheme.

`clip_logits = Σ attention · frame_logits` — attention-pooled, so the file-level score is a learned
soft pooling over frames rather than a mean.

---

## 3. 🔴 `clip_weight = 1.0` — the one change worth engineer-days

**Today both shipped configs set `clip_weight = 0.5` on all five branches**, making the loss
`0.5·BCE(clip) + 0.5·BCE(frame_max)`. `models/config.py` records this as chosen by symmetry. It is
the only number in the loss with no evidence behind it, and the evidence that exists says it is
the wrong one.

★ Lin Zhang, Xin Wang, Erica Cooper, Junichi Yamagishi, *Multi-Task Learning in Utterance-Level and
Segmental-Level Spoof Detection*, ASVspoof 2021 Workshop pp. 9–15
([PDF](https://www.isca-archive.org/asvspoof_2021/zhang21_asvspoof.pdf)) — **read from the primary
PDF**. Table 5, PartialSpoof, utterance EER %:

| Model | Utt Dev | **Utt Eval** | Seg Dev | Seg Eval |
|---|---|---|---|---|
| **Utterance single-task** | 3.96 | **6.33** | 32.69 | 44.00 |
| Segment single-task | 4.01 | 7.69 | 6.38 | 15.93 |
| **UttU** — uni-branch MTL, pooled-sum score | 8.86 | **9.96** | 7.31 | 20.04 |
| **SegU** — uni-branch MTL, min-over-frames score | 4.82 | **7.04** | 6.82 | 17.75 |
| MulBS — binary-branch | 2.98 | **5.90** | 6.56 | 17.55 |

**Our configuration is the union of UttU and SegU** — one shared `cla` layer produces
`frame_logits`, `clip_logits` is a pooled function of them, and *both* are supervised. Those are
the two rows that are **worse than utterance-only**, by 0.71 and 3.63 points.

The mechanism is stated in §6.2 and diagrammed as Fig. 4, *"Contradiction phenomenon in UttU/SegU"*:

> *"After introducing the segmental labels to the utterance-basic model in UttU, the segmental-level
> detection performs better but the utterance-level detection degrades — vice versa in SegU …
> indicating that introducing another level's labels is meaningful but sharing the entire neural
> network can reduce the original level of performance. This is explainable because conflict cases
> exist, such as how some bona fide feature vectors in a spoofed trial might be updated by two
> gradients from opposite directions."*

### ⚠️ Four limits on transferring that number — read before treating 3.63 as ours

1. 🔴 **Their footnote 6 scopes the mechanism**: *"This explanation is applicable to models that use
   length-normalized vectors for classification, e.g., angular softmax and additive-margin
   softmax."* Their criterion is **P2SGrad-MSE**, angular, on a hypersphere. **We use plain BCE on
   unnormalized logits.** The general claim — two opposing gradients on one shared representation —
   is architecture-general; the geometric magnitude is not ours to quote.
2. ⚠️ **No dispersion is reported.** §5.2 averages six seeds (`10⁰–10⁵`); Tables 4/5/6 are bare
   numbers with no std, CI or significance test. The 3.63 gap is large enough to survive most
   plausible noise; **the 0.71 gap is not testable.**
3. 🔷 **It bites on sequential compositions, not on overlaps.** The conflict needs *"bona fide
   feature vectors in a spoofed trial"*. In a 순차적 file, genuinely real segments sit inside a
   fake-labelled file — live. In a fully overlapped mix every frame carries the fake component, so
   there is no bona fide frame to conflict. Our exposure scales with the sequential fraction and
   with cells 6/7 ([data/02](../data/02-label-taxonomy.md)).
4. ⚠️ Their frontend is LFCC + SELCNN + BiLSTM with no augmentation, not an SSL encoder.

### 🔷 A second reason, specific to us

Their utterance branch uses **average pooling**, which cannot express "any frame is fake" — hence
their need for a segment-level path. Ours is **learned attention**, which can peak arbitrarily and
therefore already subsumes what `frame_max` provides. The redundancy argument is ours alone and is
untested, but it points the same way.

### The decision

🔴 **Default to `clip_weight = 1.0`, and make the frame term an ablation rather than an
assumption.** Rationale: clip-only is the best single-task row (6.33), it costs *nothing* to adopt
(one config value), and we currently do not know whether our frame loss helps or hurts. Running the
control is cheap; shipping an unexamined 0.5 is not.

⚠️ **What is lost, honestly stated.** Utterance single-task scores **44.00** segmental EER — chance.
Clip-only has no localized notion of *which part* of a file is fake. Two consequences: if the File
head's route to cells 6/7 runs through frame evidence, we lose it; and the frame-level
interpretability artifact is worth **15 of the 100 second-stage report points**
([architecture/04](../architecture/04-heads-and-pooling.md)). ⚠️ Note the artifact survives anyway —
`frame_logits` still receive gradient through `clip_logits = Σ attention · frame_logits`; they are
simply no longer *directly* supervised.

⚠️ **Nobody has measured whether frame supervision improves file-level *ranking* of short fake
components.** That is the question that would decide between clip-only and MulBS, and it does not
exist in the literature. It is **T1** in [04 §4](04-schedule.md#6-the-experiments-that-are-actually-worth-running).

### If T1 shows the frame term carries signal — MulBS, not a shared head

| | |
|---|---|
| **Split point** | at the per-component embedding sequence `h`; everything below untouched |
| **Stays shared** | SSL frontend, trunk, per-component stem producing `h` |
| **Duplicated** | 5 heads × 2 small projections. *Frame path*: `h` → own projection → frame logits → frame loss, masked to composed files. *Clip path*: `h` → own projection → attention pool → clip logit → component loss on all files |
| 🔴 **Load-bearing constraint** | `clip_logits` must **not** be computed from the frame logits the frame loss touches. Today one tensor carries both gradients |
| **Inference score** | the clip path alone — that is what 5.90 measures. A `w·clip + (1−w)·frame_max` mixture over a binary-branch model is **unmeasured in any paper** |
| **Cost** | 5 projection layers. No change to trunk, frontend, pipeline or composition code |

⚠️ **Do not adopt warm-up staging (UttBW, 5.66) on the strength of that number.** §5.2 states the
warm-up models used *"the best pre-trained model in the development set"* plus hand-pinned seeds
`10⁶` and `10⁴` outside the standard range, while MulBS did not. It is dev-selected and not
comparable; the corroborating signature is SegBW holding the best dev utterance EER (2.53) and the
worst binary-branch eval (6.07). Treat 5.66 as an optimistic bound, and 0.24 points is 14% of our
floor regardless.

---

## 4. Per-head weights — metric-proportional, by argument

`LossConfig.weights` defaults to `1.0` everywhere, inherited from PC-Mix, whose metric weighted its
components equally. **Ours does not**: 0.45 / 0.27 / 0.18 / 0.05 / 0.05.

⚠️ **There is no literature on this.** A targeted search found **no study** testing loss weights
proportional to evaluation-metric weights — MTL work assumes an unweighted mean of per-task metrics.
[09 B11](../architecture/09-open-questions.md) predicted exactly this ("⚠️ probably not resolvable
individually — set by argument"), and that is what we do.

The nearest evidence, ★ Kendall & Gal CVPR 2018 ([arXiv:1705.07115](https://arxiv.org/abs/1705.07115))
Table 1, CityScapes:

| Weighting | Seg IoU ↑ |
|---|---|
| Equal (unweighted sum) | 50.1% |
| **Grid-searched constants** | **62.8%** |
| Learned uncertainty | 63.4% |

Equal → tuned constants is **+12.7**; tuned → learned is **+0.6**. ⚠️ But their gain comes from
mismatched loss *scales* (depth regression vs classification), and **our five heads are all BCE on
[0,1]** — so the analogous gain is far smaller. Their grid optimum was `0.89 : 0.01 : 0.10`, wildly
non-uniform, which at least shows uniform is not a safe default.

❌ **Skip every adaptive scheme** — GradNorm, PCGrad, DWA, CAGrad, uncertainty weighting. The
negative results are strong and directly on point:

| Source | Finding |
|---|---|
| ★ Kurin et al., NeurIPS 2022 ([2201.04122](https://arxiv.org/abs/2201.04122)) | Unitary scalarization "matches or improves upon" complex multi-task optimizers; they "introduce significant memory, runtime, and implementation overhead" |
| ★ Xin et al., NeurIPS 2022 | MTO methods "do not yield any performance improvements beyond what is achievable via traditional optimization"; results "can be fully replicated by simply optimizing a weighted average of the losses" |
| ☆ [2505.10347](https://arxiv.org/html/2505.10347) | At 2 tasks "all SMTOs performed closely to Unit. Scal."; fixed weights extracted from a trained SMTO match the SMTO |

Five homogeneous BCE heads on one backbone is the low-heterogeneity regime where all three agree
learned ≈ tuned constants — at roughly 5× the per-step cost for per-task gradients.

### 🔴 Music gets 0.27 and no difficulty bonus

The music head is simultaneously our **highest-weighted component** (0.27), our **hardest**
(46.4% cross-generator EER in the literature), and our **least diverse** (the ≥8-family shortfall,
[validation/01](../validation/01-split-scheme.md#-the-music-head-cannot-support-the-planned-split)).
The literature contradicts itself on whether to up-weight hard tasks — ☆ Guo et al. ECCV 2018 says
imbalanced difficulty "can lead to an unnecessary emphasis on easier tasks"; the hard-mining
literature says over-emphasis on hard samples risks overfitting and cannot distinguish hard from
mislabelled.

🔷 **Our resolution**: up-weighting helps when hardness is *optimization*-limited and hurts when it
is *data*-limited. Music is data-limited. **A loss weight cannot manufacture generator diversity** —
the fix is Phase B synthesis, not a coefficient. Set 0.27 and stop.

### ⚠️ The weight you set is not the weight in effect

`_masked_mean` divides by `mask.sum()`, so a head's **effective per-sample weight is `w_c / p_c`**,
where `p_c` is the fraction of the batch carrying that component — not `w_c`. Two defensible
readings, and they differ:

- **Head-level** gradient contribution per batch is `w_c ·  ∇(mean over subset)` — independent of
  `p_c`. By this reading `w_c` is already the right knob.
- **Per-sample** influence on the shared trunk is `w_c / p_c`. By this reading a masked head is
  silently amplified.

🔷 **Commit `w_c` = the metric weights, and log `w_c / p_c` per head as a standing diagnostic.**
Do not compensate analytically — that would couple the loss to the cell mix, which is exactly the
coupling [pipelines/02](../pipelines/02-sampler.md) exists to prevent. But do not tune `w_c` without
looking at `w_c / p_c`, or you are tuning a number that is not the one in effect.

---

## 5. Ranking term, focal, label smoothing — all zero

- **`ranking_weight = 0.0`.** ★ TFPARN's own ablation: pairwise moves EER **12.91 → 12.92**
  ([03 §1](03-ruled-out.md#1-pairwise-ranking-loss---weight-stays-0)). This **reverses the
  justification currently in `models/losses.py`'s docstring** — see [05](05-corrections.md).
- **Focal: not used.** ★ *Provably* zero under a ranking metric: the focal minimizer is a strictly
  monotone transform of the posterior (Charoenphakdee et al., CVPR 2021), and EER/AUC are invariant
  to strictly monotone transforms ([01 §2.2](01-what-the-metric-demands.md)).
- **`label_smoothing = 0.0`.** 🔷 By the same argument, not a citation: smoothing shrinks the
  optimum toward 0.5, which is a monotone transform of the posterior, so it cannot change the
  asymptotic ordering. Retained as a config knob for the two presence heads, where ROC-AUC over a
  possibly noisy presence label is the mildest case for it, but defaulted off.

---

## 6. Label-confidence tiers — one run, low priority

Our corpus has `label_confidence ∈ {exact, reported}` by construction
([pipelines/01](../pipelines/01-sample-contract.md)). ★ `[Freesound 2019, 1st]` used BCE on curated
and **Lsoft β=0.7** on noisy.

🔴 **There is a theorem that argues against expecting much.** ★ Menon, van Rooyen, Ong & Williamson,
ICML 2015 ([PMLR v37](https://proceedings.mlr.press/v37/menon15.pdf)) — §3 is titled *"BER and AUC
are immune to corruption."* **Corollary 3**:

```
AUC_{D_corr}(s) = (1 − α − β) · AUC_D(s) + (α + β)/2
```

A strictly increasing affine map of the clean AUC ⇒ **the argmax over scorers is identical**.
Optimizing AUC under class-conditional label noise optimizes AUC on clean data. ⚠️ The guarantee is
for **class-conditional, instance-independent** noise; our contamination (an undisclosed AI track in
a "real" pool) is plausibly *instance-dependent*, so this is a strong prior, not a proof for our
case.

**Verdict**: `l_soft` on the `reported` tier only, one β, one run, after everything else.
⚠️ **Do not copy `LqLoss` from the Freesound repository** — its form gives **zero gradient on every
negative**, benign in their sparse 80-class multi-label setting and broken in our 5-head binary one.
And `l_soft`'s target blend must stay under `no_grad`, or it degenerates into self-consistency.

---

## 7. Distillation — unchanged, and still conditional

`α · MSE(distill_emb, teacher_emb.detach())`, α = 1.0, exactly as
[architecture/06 §3](../architecture/06-compression.md) specifies. ⚠️ The borrowed
0.898-vs-0.876 result assumes a **trainable backbone**, and [09 B9](../architecture/09-open-questions.md)
flags that a frozen frontend removes the mechanism. Stage-gated, and not part of the base objective.

---

## 8. What the sampler must guarantee

★ E9: *"tune loss, sampler and augmentation as a package."* Three constraints below are properties
of the **sampler** that the loss depends on. They belong in
[pipelines/02](../pipelines/02-sampler.md) and are restated here because the loss is unsound
without them.

| # | Constraint | Why |
|---|---|---|
| **C1** | Per-head positive rate in **[0.2, 0.8]**, measured **per masked head, after masking** | Removes BCE's imbalance pathology so no AUC-surrogate term is needed ([01 §2.4](01-what-the-metric-demands.md#24--we-control-the-imbalance-so-we-remove-the-pathology-instead-of-patching-it)) |
| **C2** | A floor on **per-head present-count per batch** | `_masked_mean` scales as ~1/√n; at `n=2` the gradient is **16×** its `n=32` value — measurable, and pure variance |
| **C3** | Composedness label-independent **within each presence stratum** (`f₈` = 0 primary, 1 = strict), **and** `P(mixed\|FAKE) = P(mixed\|REAL)`, both over cells 1–8 | ★ Now measured, not inferred: PartialSpoof Fig. 5 puts EER **above 14% at zero concatenation boundaries** ([03 §4](03-ruled-out.md#4-multi-resolution-frame-supervision---downgraded-from-headline-to-optional)). ⚠️ Mixing is a structural transform too, and it is set by the **cell mix** — see the verified reference mix in [pipelines/02 §4](../pipelines/02-sampler.md#-c3-covers-mixing-not-only-concatenation--and-that-constrains-the-cell-mix) |

---

## 9. Resolvability

⚠️ Against a ≈1 pt local resolution threshold, **only one row here is worth an experiment.**

| Decision | Expected effect | Resolvable? |
|---|---|---|
| **`clip_weight` 0.5 → 1.0** | 0.71–3.63 pts (their regime) | ✅ **T1 — the only one** |
| MulBS split, if T1 warrants | ≤0.43 pts | ⚠️ no — decide by T1's sign, not its size |
| Per-head weights → metric-proportional | unknown, no literature | ❌ set by argument; verify it does not *hurt* |
| `ranking_weight` 0 | 0.01 pts | ❌ settled by TFPARN |
| Focal | 0 asymptotically | ❌ settled by theorem |
| Lsoft on `reported` tier | ≲1 pt | ❌ run once, expect nothing |
