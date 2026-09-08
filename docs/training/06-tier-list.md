# 06 — Tier List: What to Implement, In Order

The answer to *"which objective and training method is optimal."* Tiers use the repo's
[standard legend](../data/07-eda-plan.md#tier-legend-used-in-05-06-07):
**S** blocking · **A** high EV, evidence-backed · **B** moderate · **C** speculative ·
**X** rejected, recorded so it is not rediscovered.

---

## 🔴 The headline: do not implement multiple objectives

The objective is **three config values**, and implementing more of them is *negative* expected
value — it spends engineer-days from a ~10-day budget
([architecture/08 §4b](../architecture/08-training-recipe.md#4b--the-budget-nobody-costed-engineer-days))
to produce effects below the ≈1 pt threshold at which we can measure anything
([01 §3](01-what-the-metric-demands.md#3--the-epistemic-constraint-most-objective-questions-are-undecidable-for-us)).

**The leverage is not in the loss.** Ranked by measured effect size, the top of this list is
entirely *training method* — sampling and schedule — and the loss appears only once:

| Rank | Item | Measured effect | Kind |
|---|---|---|---|
| 1 | **C3** — stratified composedness balance *and* mixedness balance | EER **>14%** at zero boundaries vs low single digits | sampler |
| 2 | Codec-aware training (S3) | hard-negative FPR **98.7% → 8.0%** | schedule |
| 3 | DOSS domain capping | **2.77% vs 3.29%** EER (0.2k h balanced vs 6.4k h naive) | sampler |
| 4 | `clip_weight` 0.5 → 1.0 | **0.71–3.63** pts (source regime) | **loss** |
| 5 | everything else in the loss | **< 1 pt**, mostly < 0.3 | loss |

⚠️ Ranks 1–3 all clear the threshold comfortably. Rank 4 clears it in its source regime with
[four transfer caveats](02-the-loss.md#-four-limits-on-transferring-that-number--read-before-treating-363-as-ours).
Rank 5 is everything the original question was about.

---

## 1. Objectives

### Tier S — blocking, and all three are free

| # | Item | Why S | Cost |
|---|---|---|---|
| **O-S1** | **Masked component losses** — each component loss masked to files where that component is present | Mandated by the masked EER pools. An unmasked loss trains on what is *never scored* ([01 §2.1](01-what-the-metric-demands.md)) | ✅ already implemented |
| **O-S2** | **Non-saturating output map**, blended in logit space, squashed once | Guard, not gain: saturation measured **EER 0.0950 → 0.3017**. Rank normalization is ❌ forbidden by rule 2.4, so tie-freedom must come from numerics | ✅ already implemented |
| **O-S3** | 🔴 **`clip_weight = 1.0`** (clip-only) as the default, frame term demoted to ablation **T1** | The **only** objective-side finding above our resolution threshold. Our shipped head is the configuration that measured **0.71–3.63 pts worse** than utterance-only ([02 §3](02-the-loss.md#3--clipweight--10--the-one-change-worth-engineer-days)) | **one config value** |

### Tier A — do these next

| # | Item | Evidence | Cost |
|---|---|---|---|
| **O-A1** | **Metric-proportional head weights** `.45/.27/.18/.05/.05` | ⚠️ No literature exists ([09 B11](../architecture/09-open-questions.md) predicted this). But uniform is *inherited from PC-Mix, whose metric weighted components equally and ours does not*. Argued, not measured | one config value |
| **O-A2** | **Distillation MSE** against frozen teachers, stop-gradient | ☆ 0.898 vs 0.876, secondary-sourced. ⚠️ Conditional on [09 B9](../architecture/09-open-questions.md) — a frozen frontend removes the mechanism the recipe depends on | high (teachers) |

### Tier B — only if S and A are done

| # | Item | Note |
|---|---|---|
| **O-B1** | **MulBS split** (frame path separated from clip path) | 🔴 **Gated on T1's sign.** ≤0.43 pts, below threshold — build it only if T1 shows the frame term carries signal ([02 §3](02-the-loss.md#if-t1-shows-the-frame-term-carries-signal--mulbs-not-a-shared-head)) |
| **O-B2** | **`l_soft` β=0.7 on the `reported` tier only** | ⚠️ Menon Cor. 3: **AUC is provably immune to class-conditional label noise**. One run, expect nothing ([02 §6](02-the-loss.md#6-label-confidence-tiers--one-run-low-priority)) |
| **O-B3** | **Label smoothing on the two presence heads only** | 🔷 Zero by the monotone-transform argument. Its only case is ROC-AUC over a noisy presence label |

### Tier X — rejected, with the number that closed each

| # | Item | Closed by |
|---|---|---|
| **O-X1** | Pairwise ranking loss / S4 rank polish | ★ TFPARN's own ablation: EER **12.91 → 12.92**. All its gains are minDCF/Cllr/actDCF |
| **O-X2** | Focal loss | ★ **Provably zero** at the population optimum (Charoenphakdee et al., Thm 3/11 + Lemma 14) |
| **O-X3** | OC-Softmax / AM-Softmax / SAMO | ★ ASVspoof 5: plain softmax **16.32** beats every one-class variant (**24.88–27.44**) |
| **O-X4** | GradNorm / PCGrad / DWA / uncertainty weighting | ★ Kurin & Xin, NeurIPS 2022: unitary scalarization "matches or improves upon" them, at ~5× cost |
| **O-X5** | Multi-resolution frame supervision *as a headline* | ★ **+0.15 in-domain, −0.13 out-of-domain** |
| **O-X6** | Reinforcement learning | No sequential decision, no exploration; a higher-variance estimator of a surrogate that already measures 0.00 |

---

## 2. Training methods

🔴 **This is where the score is.** Three of the four largest measured effects available to us are here.

### Tier S — blocking

| # | Item | Evidence | Cost |
|---|---|---|---|
| **M-S1** | **C3** — composedness label-independent within each presence stratum (`f₈`=0 primary, `f₈`=1 strict), **and** mixedness independent of `FILE_FAKE` | ★ PartialSpoof Fig. 5: EER **above 14% at zero concatenation boundaries**, authors attributing it to overlap-add artifacts. ⚠️ Worse for us — their eval set is composed, our test set contains genuine unspliced audio. 🔴 Use the [verified reference cell mix](../pipelines/02-sampler.md#the-reference-cell-mix); a mix that passes C1 can still fail C3 | sampler policy |
| **M-S2** | **C1** — per-head positive rate ∈ **[0.2, 0.8]**, after masking | Removes BCE's imbalance pathology, which is the *only* reason any AUC-surrogate term would be worth adding ([01 §2.4](01-what-the-metric-demands.md#24--we-control-the-imbalance-so-we-remove-the-pathology-instead-of-patching-it)) | sampler constraint |
| **M-S3** | **C2** — floor on per-head present-count per batch | Measured here: `_masked_mean` scales ~1/√n; at n=2 the gradient is **16×** its n=32 value. Pure variance | sampler constraint |
| **M-S4** | **Artifact-family-disjoint splits + VG1** | Without it every number below is meaningless — a random split reports ~0.99 ([validation/01](../validation/01-split-scheme.md)) | already designed |
| **M-S5** | **Test-chain normalization** (A-S1) | *"The single highest-leverage step in the whole pipeline."* ⚠️ Blocked on **G1** `signal_chain.yaml`, which does not exist | blocked |

### Tier A — high expected value

| # | Item | Evidence | Cost |
|---|---|---|---|
| **M-A1** | ⭐ **S3 codec-aware training stage** | ★ **The best-evidenced item in the entire recipe**: hard-negative FPR **98.7% → 8.0%**, cross-codec drift **−83%**. Independently corroborated — ASVspoof 5's *hardest* condition is codec-10 (speex, 8 kHz narrowband), which **is our telephone slice** | schedule stage |
| **M-A2** | **DOSS domain capping**, `domain_key = source × generator`, start `N_c ≈ 500` | ★ **0.2k h domain-balanced → 2.77% EER vs 6.4k h naive → 3.29%**. Above threshold, and it is a sampler *weight*, so nothing is discarded | one sampler weight |
| **M-A3** | **Cross-domain MixUp**, hard/union labels | ★ The legal substitute for test-set pseudo-labeling, which rule 2.3 ❌ forbids. Three independent sources agree on `max` labels | composition policy |
| **M-A4** | **Checkpoint soup** across epochs / same-init runs | ☆ Ensemble effect at **exactly zero inference and packaging cost**. ⚠️ Same loss basin only — **not** across seeds | ~free |
| **M-A5** | **S1 → S2 staged** (independent, then joint) | ⚠️ Re-graded: the headline is ACC/F1; the EER deltas are **0.47–0.86 pts**, below threshold. Keep because it is our architecture anyway and nearly free — not because it is measurable ([04 §2](04-schedule.md#2--s2s-evidence-is-a-thresholded-metric-result--keep-the-stage-drop-the-claim)) | cheap |

### Tier B

| # | Item | Note |
|---|---|---|
| **M-B1** | Teacher ensemble (XLS-R-1B/2B, SSLAM/EAT, Demucs residuals) | Feeds O-A2. Expensive; rung 6 of 7 in the drop order |
| **M-B2** | Seed replication for fold variance | ★ E5: lower fold variance often beats higher mean. Compute-cheap, engineer-cheap |
| **M-B3** | RawBoost / MUSAN / RIR augmentation | ⚠️ Measured recipe is **milder than instinct**: gain ±6 dB, SNR 10–30 dB. *"MixUp does the heavy lifting, not signal mangling"* |

### Tier X — rejected

| # | Item | Closed by |
|---|---|---|
| **M-X1** | Warm-up staging (UttBW) | ❌ Dev-selected with hand-pinned seeds outside the standard range; not comparable to MulBS |
| **M-X2** | Test-set pseudo-labeling | ❌ Rule 2.3 |
| **M-X3** | Rank normalization / rank averaging across the cohort | ❌ Rule 2.4 |
| **M-X4** | Naive separate-then-detect | ❌ CompSpoof 0.668 vs 0.827; PC-Mix 51.38% vs 29.35% EER |
| **M-X5** | Scaling corpus hours | ❌ DOSS: 6.4k h naive *loses* to 0.2k h balanced |

---

## 3. The order to actually build in

⚠️ Strictly sequential. Each rung is cheap relative to the one below, and stopping anywhere leaves
something shippable.

```
1.  O-S1, O-S2                already done — verify, don't rebuild
2.  M-S4                      folds + VG1            ← nothing below is measurable without it
3.  M-S1, M-S2, M-S3          sampler constraints    ← the three biggest levers, all free
4.  O-S3 + T1                 clip_weight 1.0, ablated
5.  O-A1                      metric-proportional weights
6.  M-A2, M-A3                DOSS cap, cross-domain MixUp
7.  M-A5                      S1 → S2 staging
8.  M-A1                      ⭐ codec-aware stage   ← highest-evidence optional item
9.  M-A4                      checkpoint soup
10. M-S5                      test-chain normalization (unblocks when G1 lands)
--- everything below is upside, not foundation ---
11. O-B1                      MulBS, only if T1's sign warrants
12. M-B1 + O-A2               teachers + distillation
13. O-B2, O-B3, M-B2, M-B3
```

🔷 **Steps 1–5 are perhaps two engineer-days and contain every above-threshold objective decision.**
Step 8 is the single highest-value item that is not free.

---

## 4. 🔴 Ablations are scored — but the rejections are already the artifact

⚠️ **25 of 100 second-stage points** are 모델 개발, judged on *"학습 및 성능 개선 과정의 체계성"*
([architecture/08 §4](../architecture/08-training-recipe.md#4-what-the-h200s-are-for)). Ablations
are scored, not optional. That creates a real temptation to implement the Tier-X objectives just to
report them.

**Resist most of it.** [03](03-ruled-out.md) — four axes, ~20 papers, with the measured number that
closed each and the metric that produced it — **is** the systematic-process artifact, and it cost no
GPU time. A literature rejection with a citation is stronger report material than a noisy in-house
run that could not have resolved the effect anyway.

🔷 Two exceptions worth actually running, both nearly free because the code already exists:

| Ablation | Why it earns its cost |
|---|---|
| **T1** — `clip_weight` 1.0 vs 0.5 | Not a formality: it decides a shipped default, and its answer is genuinely unknown |
| **`ranking_weight` 0 vs 0.3**, once | One config value. Confirms TFPARN's null *on our data*, and converts a borrowed rejection into a measured one |

⚠️ Report both **with the noise floor stated**. An ablation that reports "no significant difference"
against a declared ≈1 pt resolution threshold demonstrates 체계성; one that reports a 0.2 pt
"improvement" as a finding demonstrates the opposite.
