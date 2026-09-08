# 01 — What the Metric Demands of the Objective

The objective is derived from the metric, not chosen from a menu. This file does the derivation;
[02](02-the-loss.md) states the result; [03](03-ruled-out.md) records what the derivation
eliminates and why.

---

## 1. The metric, exactly

```
Score = 0.9 × ADS + 0.1 × CPS
ADS   = 0.5·(1 − File EER) + 0.2·(1 − Voice EER) + 0.3·(1 − Music EER)
CPS   = 0.5·(Voice Presence ROC-AUC) + 0.5·(Music Presence ROC-AUC)
```

| Head | Metric | Effective weight |
|---|---|---|
| `FILE_FAKE_PROB` | 1 − EER | **0.45** |
| `MUSIC_FAKE_PROB` | 1 − EER | **0.27** |
| `VOICE_FAKE_PROB` | 1 − EER | 0.18 |
| `VOICE_PRESENT_PROB` | ROC-AUC | 0.05 |
| `MUSIC_PRESENT_PROB` | ROC-AUC | 0.05 |

★ Voice EER is computed **only over voice-present files**, Music EER only over music-present
files, using the organizers' ground truth ([competition/03](../competition/03-evaluation.md)).

---

## 2. Four consequences, in order of how much they constrain the loss

### 2.1 Masked heads — settled, and already implemented

Because the component EERs are masked pools, a voice-fake loss on a music-only file trains the
model on something that will never be scored. The loss mirrors the metric: each component loss is
masked to the files where that component is present. This is PC-Mix's scheme
([papers/04](../papers/04-component-partial.md)) and it is in
[`models.losses.multitask_loss`](../../models/AGENTS.md) via `BranchConfig.masked_by`.

### 2.2 🔴 The metric is invariant to monotone transforms — so calibration is worth exactly zero

EER and ROC-AUC read only the **ordering** of scores. Any objective term whose effect is to
reshape the score *scale* — to make probabilities better calibrated, or to move a decision
boundary — is worth nothing here, however much it is worth in the literature.

**This is provable for focal loss, not merely observed.** ★ Charoenphakdee, Vongkulbhisal,
Chairatanakul & Sugiyama, *On Focal Loss for Class-Posterior Probability Estimation*, CVPR 2021
([arXiv:2011.09172](https://arxiv.org/abs/2011.09172)):

- **Thm 3** — focal loss is classification-calibrated for any γ ≥ 0. Its proof (Appx. A.4)
  establishes order preservation: *"Since `q^{γ,∗}` preserves the order of `η` …"*
- **Thm 5** — focal loss is *not* strictly proper. **Thm 11** gives the closed form recovering `η`:
  `Ψᵢ^γ(v) = h^γ(vᵢ) / Σ_l h^γ(v_l)`.
- **Lemma 14** (Appx. A.3) — `h^γ` is **strictly increasing** on (0,1).

⚠️ **One step here is ours, not theirs, and it matters.** The paper's order-preservation statement
is *across classes for a fixed input* (`arg max_y`). AUC and EER need ordering **across inputs for
a fixed class** — a different claim. 🔷 The bridge: in the binary case Thm 11 reduces to
`q = h^γ(η) / (h^γ(η) + h^γ(1−η))`, a **fixed scalar function of `η` alone**, applied pointwise;
by Lemma 14 it is strictly increasing in `η`. So instance ordering is preserved too.

⇒ The focal minimizer is a **strictly monotone transform of the posterior**, and AUC/EER are
invariant to strictly monotone transforms. **Focal and BCE have identical AUC/EER at the
population optimum.** Focal moves calibration (Cllr), never the asymptotic ordering.

⚠️ This is an argument about the **population optimum**. It does not by itself forbid a
finite-sample difference — which is exactly the loophole §2.3 walks through.

⚠️ This is why so much of the anti-spoofing literature does not transfer: its headline metrics —
minDCF, Cllr, actDCF, Macro-F1, accuracy — are all threshold- or calibration-sensitive, and are
precisely what these losses buy. See [03](03-ruled-out.md) for the measured cases.

### 2.3 🔴 But invariance is asymptotic — finite-sample optimization is a real exception

The argument in 2.2 holds **at the population optimum**, not during finite-sample optimization.
Two losses with the same asymptotic target can converge to differently-ranked functions, and there
is a replicated class of cases where a loss term produces a genuine *ranking* gain:

| Case | Metric | Gain |
|---|---|---|
| AUC-margin (AUC-M), PatchCamelyon | ROC-AUC | 0.8394 → **0.8896** |
| AUC-margin, DDSM+ | ROC-AUC | 0.9392 → 0.9544 |
| pAUC-L, speaker verification (SITW Dev) | EER | 3.04 → **2.23** |

These are not calibration artifacts. The mechanism is specific and known: **under BCE the gradient
with respect to a negative sample scales with the predicted probability `p̂`**, so at low positive
rates BCE barely trains on negatives. AUC-surrogates repair an *optimization* failure, not a
*target-specification* failure.

> 🔴 **The rule this repo adopts.** Losses that reshape the score scale (focal, label smoothing)
> are worth **zero**. Losses that repair BCE's imbalance pathology are worth something
> **proportional to the imbalance**.

### 2.4 🔴 We control the imbalance, so we remove the pathology instead of patching it

The gain in 2.3 is a function of the positive rate `π`; the positive:negative gradient ratio is
about `(1−π)/π`. Measured, across the studies found:

| `π` | grad ratio | Measured AUC-surrogate gain | Source |
|---|---|---|---|
| 1% | 99:1 | **+0.050 AUC** | AUC-M, PatchCamelyon |
| 1.76% | 56:1 | +0.005–0.007 AUC | AUC-M, Melanoma |
| 3.3% | 29:1 | +0.00077 AUC | RankNet, Criteo (KDD 2024) |
| **10.3%** | 8.7:1 | **0.00 EER** | **TFPARN, ASVspoof 5** |
| 13% | 6.7:1 | +0.015 AUC | AUC-M, DDSM+ |
| 20.2% | 4:1 | +0.0006 AUC | AUC-M, CheXpert |
| 25.6% | 2.9:1 | +0.00016 AUC | RankNet, Criteo |

⚠️ **The column is non-monotone** — DDSM+ at 13% beats Melanoma at 1.76% — because dataset
difficulty confounds imbalance. No clean empirical threshold can be derived from it, and none is
claimed. What survives is: **everything at or above ~10% positives is nil or ≤0.015 AUC**, and the
two points closest to our metric and task (TFPARN at 10.3% bonafide, 0.00 EER; CheXpert at 20.2%,
+0.0006) are both effectively zero.

🔴 **The lever is ours.** Unlike every study above, we *compose* our corpus and choose the cell mix
([pipelines/02](../pipelines/02-sampler.md)). We do not have to accept an imbalance and then buy a
loss term to survive it — we can decline the imbalance.

> **Sampler constraint C1**: keep each head's effective **per-batch positive rate in [0.2, 0.8]**,
> measured **per masked head, after masking**. A head masked to 30% of files with 50/50 labels
> inside that subset is fine; a head at 50% of files with 5% positives inside it is not.

🔷 Confidence: high that at [0.2, 0.8] a pairwise term is worth 0; moderate-to-high that this is
the right lever, because it is **causal** (we set `π`) rather than correlational. ⚠️ The residual
uncertainty is that **no one has measured an AUC surrogate at balanced classes** — the regime we
intend to occupy is the regime with no data.

### 2.5 Saturation, not calibration, is the output risk

The one output-side property the metric *does* care about: measured, rounding predictions to 2 dp
is harmless while **saturating the operating point takes EER 0.0950 → 0.3017**
([architecture/01 §3.2](../architecture/01-design-envelope.md#32-ranking-only-scoring-and-the-saturation-trap-)).
Handled in [`models.outputs`](../../models/AGENTS.md) by blending in logit space and squashing once
with a non-saturating map. Nothing in the objective should undo it.

---

## 3. 🔴 The epistemic constraint: most objective questions are undecidable for us

🔴 **Two different thresholds, and they are routinely conflated — including elsewhere in this
repo.** Both matter, and an effect must clear the first to be *measurable by us at all*:

| Threshold | Value | What it governs |
|---|---|---|
| **Local resolution** — paired VAL, ≥1,200 per class per masked pool | ≈ **1.0 pt** at 95% (CI width ±1.2 on EER=0.10) | Whether *we* can tell two candidates apart ([validation/01 §4](../validation/01-split-scheme.md#4-size-floors)) |
| **Leaderboard sampling noise** — 1,200 test files | **±1.7** pts File · **±2.5** pts component | Whether an effect is visible on the LB at all ([validation/README](../validation/README.md)) |

⚠️ Pairing is what buys the first number: unpaired comparison at our VAL size resolves a 1-point
gain only **85%** of the time; paired, **95%**
([validation/03 §3](../validation/03-decision-protocol.md#3-the-promotion-rule)).

⚠️ Every objective-side effect in [03](03-ruled-out.md) is below the **tighter** of the two —
0.01 pts (pairwise), 0.15 (multi-resolution frame), 0.22 against a ±0.36 seed std (focal), 0.47–0.86
(joint training). The conclusions do not depend on which threshold is used; they fail both.

⚠️ Nearly every objective-side effect measured in [03](03-ruled-out.md) is **smaller than that**.
[09 §B](../architecture/09-open-questions.md) already states the governing principle:

> *"A question we cannot resolve is not an experiment — it is a decision to make by argument and
> leave alone."*

So this directory **commits to defaults by argument** and marks each as resolvable or not. It is
not a sweep plan. Sweeping an unresolvable knob spends engineer-days to produce noise, and
[architecture/08 §4b](../architecture/08-training-recipe.md#4b--the-budget-nobody-costed-engineer-days)
says we have roughly ten of them for all modelling.

## 4. What the derivation leaves

Once masking, monotone-invariance, imbalance and saturation are accounted for, **very little of the
objective is still free**. The loss that mirrors this metric is close to the simplest one that
respects the masks — which is most of what [02](02-the-loss.md) says.

🔴 The one genuinely decision-changing finding in this whole axis is not about *which* loss but
about **where the losses attach**: supervising utterance and frame level through one shared head
cost **0.71–3.63 EER points** versus utterance-only, which is above our noise floor by more than
2×. That is [02 §4](02-the-loss.md) and it is the only objective-side change here worth engineer-days.
