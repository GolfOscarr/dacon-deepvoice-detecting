# 08 — Training Recipe

What runs on the 8×H200. This file assumes candidate **B** ([03](03-candidates.md)) and the
compression plan in [06](06-compression.md).

⚠️ **The corpus does not exist yet.** Phase B is planned but not executed
([data/08](../data/08-build-plan.md)), and ★ the Kaggle ordering is explicit —
`1) data split → 2) sampler/loss → 3) architecture → 4) optimizer`, with a named warning against
over-searching architecture before solving data shift
([kaggle/05 E8](../kaggle/05-transferable-playbook.md)). Nothing here should start before the
folds exist and VG1 passes ([validation/](../validation/README.md)).

---

## 1. The staged schedule

★ Both component-level papers stage the same way, and both report the joint stage as the
**largest single gain**:

| Stage | What trains | Source |
|---|---|---|
| **S1 — independent** | Each branch alone, frozen frontends + adapters | PC-Mix: "independent, then joint"; CompSpoof: independent for 4 epochs |
| **S2 — joint** | All branches together, component losses masked to present components | ★ PC-Mix ACC **69.40 → 85.12** · CompSpoof F1 **0.668 → 0.908** |
| **S3 — codec-aware** | Same, over 4-way codec variants of every file | ★ ArtifactNet P2→P3: hard-negative FPR **98.7% → 8.0%**, cross-codec drift **−83%** |
| **S4 — ranking polish** | Low LR, pairwise ranking loss, ~2 epochs | ☆ ~1 bps in the source competition; aligns the model with a ranking metric |

★ S3's evidence is strong and it is a **training schedule**, not an architecture — so it is adopted
regardless of how the `E-A1` probe resolves ([03 E](03-candidates.md)). Our audio is MP3/WAV/FLAC
at 16 kHz with a telephone slice; codec variance is a certainty, not a hypothesis.

⚠️ S4 is worth ~1 basis point in its source competition. Do it last, and do not let it displace
data work.

---

## 2. The loss

Per head, from [04](04-heads-and-pooling.md):

```
L_head = 0.5·BCE(clip_logits, y) + 0.5·BCE(frame_max_logits, y)
       + λ_rank · pairwise_ranking(scores, y)          # S4
```

Total:

```
L = Σ_h  m_h · L_head(h)                                # m_h masks component losses to
  + α · MSE(student_emb, teacher_emb.detach())          #   files where that component is present
```

🔴 **The masks are not an optimization — they mirror the metric.** Voice EER is computed only over
voice-present files; a voice-fake loss on a music-only file trains the model on something that will
never be scored ([01 §3.3](01-design-envelope.md#33-masked-eers-mean-masked-losses)). PC-Mix does
exactly this with `λ_s = λ_e = λ_m = 1` on constructed mixtures.

⚠️ **The component weights are an unexamined default.** PC-Mix uses `λ_s = λ_e = λ_m = 1` because
its metric weighted its components equally. Ours does not — the score weights are **0.45 / 0.27 /
0.18**, and [01 §3.1](01-design-envelope.md#31-effective-weights) says to allocate capacity in that
order while this recipe then weights the losses equally. 🔷 That is a free knob pointing straight at
the objective and it should be swept, not inherited. ⚠️ Metric-proportional is the obvious first
guess but not obviously optimal — the heads differ in pool size and difficulty, and the music head
is both the highest-weighted component *and* the hardest, which can argue for either direction.

**Frame-level targets are free for us.** We compose the corpus, so we know which frames hold which
component and which are generated ([04 §4](04-heads-and-pooling.md)). Supervise at multiple
resolutions (40–640 ms, per PC-Mix and PartialSpoof).

☆ Two label-noise refinements worth carrying, both from Kaggle winners: **different loss per
label-confidence tier** (BCE on trusted data, Lsoft β=0.7 on noisy) and **two-stage clean→noisy**
training ([kaggle/05 F1, F2](../kaggle/05-transferable-playbook.md)). Our corpus has explicit
confidence tiers by construction — composed audio is exactly labelled, scraped real audio is not.

---

## 3. Augmentation, at the level that touches the architecture

Full spec is [data/06](../data/06-augmentation-spec.md); three points belong here because they
change what the model can learn.

🔴 **The governing rule**: every structural transform — mixing, concatenation, crossfade, gain —
must appear on **both sides of every label**. If cell 5 (mixed, both real) is only natural songs
while cells 6/7/8 are our artificial mixes, "artificially mixed" predicts FAKE perfectly, CV looks
superb, and the leaderboard does not move ([data/02](../data/02-label-taxonomy.md)).

⚠️ **Augmentation is milder in practice than we planned.** The measured winning recipe is gain
±6 dB, noise SNR 10–30 dB, time shift ±0.5 s — *"the heavy lifting is done by MixUp, not by signal
mangling"* ([kaggle/06 §5](../kaggle/06-notebook-code.md)).

🔴 **Cross-domain MixUp is the legal substitute for pseudo-labeling.** Mixing our clean synthetic
audio with real degraded audio (ASVspoof21 LA telephony, MUSAN, real music) bridges the domain gap
without touching the test set, which rule 2.3 forbids
([01 §1.5](01-design-envelope.md#15-rule-23--no-test-time-adaptation-)). Use hard union labels
(`max`), not weighted blends — three independent sources agree.

---

## 4. What the H200s are for

🔷 Ranked by value, given that training compute is our surplus resource
([05 §4](05-multi-model.md#4-the-main-answer-ensemble-at-training-time-ship-one-model)):

| Use | Why it needs the hardware |
|---|---|
| **Teacher ensemble** — XLS-R-1B/2B, SSLAM/EAT-large, Demucs residuals, all frozen | Imports an ensemble we could never run on an L4 |
| **Layer-probing sweeps** per frontend per head | Sets the truncation depth that sets the whole inference budget ([06 §1](06-compression.md)) |
| **Generator-disjoint CV**, run properly rather than on one fold | The binding constraint is cross-generator generalization; a random split measures nothing |
| **Seed replication** | ★ *"A model with slightly lower mean but lower fold variance is often a better final candidate"* ([kaggle/05 E5](../kaggle/05-transferable-playbook.md)) |
| **Checkpoint soup** across epochs and seeds | Free ensembling at zero inference cost ([05 §3](05-multi-model.md)) |
| **Ablations** | ⚠️ 25 of 100 2nd-stage points are 모델 개발, judged on *"학습 및 성능 개선 과정의 체계성"*. Ablations are scored, not optional |

⚠️ What the H200s are **not** for: scaling the corpus. 🔴 DOSS measured **0.2k h domain-balanced
→ 2.77% EER vs 6.4k h naive → 3.29%** ([papers/05](../papers/05-generalization.md)). More data is
not our lever; **domain balance and per-domain capping** are. Spare compute goes into teachers,
probes and ablations — not into more epochs over more hours.

---

## 4b. 🔴 The budget nobody costed: engineer-days

⚠️ *"Training compute is free"* is true of **H200-hours** and false of **engineer-days**, and this
directory has been quietly conflating them. §4 above implies six distinct training programmes —
teacher ensemble, probing sweeps, generator-disjoint CV, seed replication, checkpoint souping,
ablations — each needing code, on top of a corpus build that **has not started**
([data/08](../data/08-build-plan.md) puts it at days 3–12 of 22).

🔷 That leaves roughly **10 days for all modelling**, and six programmes in ten days is not
credible. This is the largest un-modelled risk in the architecture set, and the honest response is a
**stated drop order** rather than a plan that silently fails at the bottom.

### The ladder — build strictly in this order, stop wherever time runs out

| # | Deliverable | Why this rung | Drop only if |
|---|---|---|---|
| 1 | **P0-a** all-constant contract probe | Nothing else is real until the packaging works and it returns exactly 0.5000 | never |
| 2 | **P0-b** presence baseline (⚠️ singing-checked, [09 A7](09-open-questions.md)) | A non-degenerate score on the board; 0.10 of the metric | never |
| 3 | **Candidate A** — one trunk, 5 SED heads, masked losses | ⭐ **Stage one of B, not a fallback** — B strictly contains it, so none of this is throwaway. Gets all five heads scoring end to end | never |
| 4 | **Candidate B** — add the speech frontend, per-branch adapters, time alignment | The A→B increment is *one encoder*: A is already jointly trained, so B's marginal claim is frontend specialization ([03](03-candidates.md#-a-is-stage-one-of-b-not-a-fallback)) | time |
| 5 | **Codec-aware stage (S3)** | ★ Strong evidence, and it is a schedule not an architecture | time |
| 6 | Teacher ensemble + distillation | Buys accuracy we cannot otherwise run | time |
| 7 | **C-lite**, low-band member, ensemble members | Upside, not foundation | time |

🔴 **Rungs 3 and 4 are one continuous build, not a plan-and-a-backup.** B strictly contains A, so
rung 3 is stage one of rung 4 and nothing in it is discarded
([03](03-candidates.md#-a-is-stage-one-of-b-not-a-fallback)). A is one forward pass, one set of
weights, no cross-encoder alignment, and the best score comparability for the 0.45 head — and since
A is *already* jointly trained, the case for B over A rests on a **single citation** about frontend
specialization. If engineer-days run short, **A is not a degraded submission — it is a defensible
one**, and the A-vs-B ablation is 2nd-stage report material either way.

⚠️ **Promote B over A only under the pre-committed rule** — P1–P6 in
[validation/03 §3](../validation/03-decision-protocol.md), with the inconclusive-case tiebreaker
(lower fold variance → better worst-cell EER → better SHADOW → lower runtime). 🔷 That tiebreaker
leans toward **A** on a tie, which is the correct default: fewer moving parts, more headroom.

⚠️ Whatever is reached, the rungs below it must stay runnable and packaged. A half-finished rung 6
that cannot be zipped scores nothing, while a working rung 3 scores.

---

## 5. Validation is already built — use it

[`metrics/`](../../metrics/AGENTS.md) ships and is verified end to end: 60 tests, `eer` fuzzed
against exact rational arithmetic over 2,991 cases with zero genuine disagreements.

🔴 **Never compute EER in training code.** Import it. Three details in the official estimator are
load-bearing (`drop_intermediate=False`, FAKE as `pos_label=1`, `argmin` over `|fpr − fnr|` rather
than interpolation) and a reimplementation that "cleans up" any of them silently disagrees with the
leaderboard.

🔴 **Never pool raw OOF scores across folds** — measured **0.1705 for a true 0.100**. Use the mean
of per-fold metrics (`metrics.aggregate.fold_mean`).

⚠️ Sanity thresholds, from [survey/10](../survey/10-open-questions.md). If local CV shows these,
suspect a leak rather than success:

| Head | Suspicious if |
|---|---|
| Music fake, unseen generator | **< 3% EER** — published cross-generator is 46.4% |
| Voice fake, unseen generator | **< 1% EER** — ASVspoof 5 best is ~4% |
| Any head | perfect separation on a random split → re-split by generator |

---

## 6. Two open dependencies

⚠️ **Version skew.** The suite runs under Python 3.12 / numpy 2.5.3 / pandas 3.0.5 locally against
the server's **3.11.15 / 1.26.4 / 2.0.3**. Training code should target the server's versions from
the start rather than discovering the gap at packaging time
([`PROGRESS.md`](../../PROGRESS.md)).

🔴 **The music-family shortfall blocks fold construction.** At only 5 music generator families, a
family-disjoint 5-fold puts *one* family per validation fold and the sealed PROBE slice cannot be
carved out at all — on the highest-weighted component head (0.27). The recorded decision is to
raise the floor to **≥8 families**, which costs Phase-B synthesis
([validation/01](../validation/01-split-scheme.md)). **No music-branch training result is
trustworthy until this is resolved.**
