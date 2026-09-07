# 06 — Train Big, Ship Small

The mechanism that makes foundation-model quality fit on an L4. Four techniques, ordered by
evidence strength and by how much they cost us in accuracy.

🔷 **The framing:** with 8×H200 for training and one L4 for inference, model size at *training*
time is free and model size at *inference* time is the scarce resource. Every technique here
converts the first into the second.

| Technique | Evidence | Accuracy cost | Inference saving |
|---|---|---|---|
| **Layer truncation** | ☆ 🔴 4 layers of XLS-R-300M **match the full model** | ~none, if probed | **~3–6×** |
| **Adapters / LoRA** | ☆ meta-learned LoRA **8.84 → 5.30% EER** at ~1.1% trainable params | **negative** — better than full FT | zip size, not runtime |
| **Distillation** | ☆ 0.898 vs 0.876 with stop-gradient | ⚠️ the measured effect is *positive*; the real risk is **teacher-domain mismatch** (44.1 kHz Demucs residuals at 16 kHz) and a frozen student backbone (§3) | unbounded |
| **fp16 / compile** | engineering | none | ~2× from fp16. ⚠️ `torch.compile` probably does **not** pay here — see §4 |

---

## 1. Layer truncation — the largest free win

☆ *Probing-Guided Layer Selection from SSL Speech Models* (2025,
[arXiv](https://arxiv.org/abs/2606.30791)). Our index's summary of it
([papers/INDEX](../papers/INDEX.md)) — not a quotation from the paper:

> 🔴 **4 probed layers of XLS-R-300M match the full model, 1.34M trainable params.**

☆ Corroborated by *Comprehensive Layer-wise Analysis of SSL Models for ADD* (2025), which maps
which wav2vec2 / HuBERT / WavLM layers are spoof-discriminative
([papers/INDEX](../papers/INDEX.md)).

🔴 **A gap between the cited result and the use we make of it.** The finding is that *4 **probed**
layers* match the full model — a claim about **which layers' representations feed the head**, not
about **where those layers sit**. Truncation needs them to be *early*. If the probed set includes,
say, layer 18, we must run 18 layers to reach it and the 3–6× saving evaporates, taking the
affordability of two frontends with it.

⚠️ Layer-selection and truncation are therefore **not the same technique**, and this directory has
been treating them as one. The probing sweep ([09 B3](09-open-questions.md)) must report the
*depth* of the selected layers, not just the accuracy at those layers — and the fallback if useful
layers turn out to be deep is layer-selection *without* truncation, which saves head parameters but
not inference time.

⚠️ **This is also a card-level entry** in [papers/INDEX](../papers/INDEX.md) — the number comes from
an abstract, not a primary read — and the **entire inference budget rests on it**. Two truncated
frontends fit only if it holds. It is the highest-priority promotion to a deep read
([09 D](09-open-questions.md)), and the probing sweep in [09 B3](09-open-questions.md) measures it
on our own data regardless, which is the real safeguard.

🔷 Why this should be true rather than surprising: the upper layers of a speech SSL model
specialize toward phonetic and semantic content, which is precisely the information a spoofing
countermeasure does *not* need. Anti-spoofing lives in low- and mid-level acoustic detail.

**Consequence for us.** Truncating XLS-R-300M from 24 layers to ~8–9 cuts the encoder to roughly
a third. That is what makes **two specialist frontends affordable at once**
([02](02-pretrained-catalog.md#the-case-for-two-frontends)) — a design that would otherwise be
priced out by [01 §2](01-design-envelope.md#2-the-compute-ceiling).

⚠️ **The evidence is for speech SSL models on speech anti-spoofing.** Whether the same layer
economy holds for a general-audio encoder on AI-*music* detection is unmeasured. 🔷 We should not
assume it transfers; the probing sweep must be run separately per frontend
([09 B3](09-open-questions.md)).

**How to choose the depth.** Linear probes per layer on our own validation pool, per head. Cheap
on 8×H200, and it produces a **layer-importance plot that is directly reportable** for the
2nd-stage 결과 해석 section.

---

## 2. Adapters — specialization without duplication

☆ Meta-learned LoRA: **average EER 8.84 → 5.30%** versus full fine-tuning, at **~1.1% trainable
parameters**. ☆ Mixture of Low-Rank Adapter Experts: OOD EER 8.55 → 6.08%.
☆ Multi-Scale Convolutional Adapter: cheaper than a full LoRA sweep.
Meta-learned LoRA is in [papers/05](../papers/05-generalization.md); the other two are card-level
entries in [papers/INDEX](../papers/INDEX.md).

🔴 The striking part is that LoRA **beat** full fine-tuning rather than approximating it. 🔷 The
plausible reason is the same one that motivates one-class methods: full fine-tuning on a
generator-limited corpus overfits the generators it saw, and constraining the update to a low-rank
subspace limits how far the frontend can specialize to them. If that reading is right, the benefit
should be *larger* for us than for the paper, since cross-generator generalization is our binding
constraint (46.4% EER published on music).

**Consequence for us.** Freeze both frontends; buy branch specialization with adapters. One copy
of each frontend's weights in the zip serves all branches. 🔷 This is what makes a two-frontend,
three-branch design fit in 10 GB at all
([01 §1.2](01-design-envelope.md#12-everything-loads-offline-from-model)).

⚠️ Frozen frontends and joint training are in tension: PC-Mix's and CompSpoof's joint gains came
from branches genuinely co-adapting. Adapters are how we keep co-adaptation while keeping the
frontend shared and frozen. **That the gain survives adapter-only joint training is an assumption
we should test**, not one to build on silently ([09 B4](09-open-questions.md)).

---

## 3. Distillation — and its honest risk

☆ The Kaggle result: **0.898 with distillation vs 0.876 without**, using embedding MSE against a
frozen teacher plus a stop-gradient
([kaggle/06 §3](../kaggle/06-notebook-code.md)). ⚠️ Attributed to a discussion thread — treat the
number as indicative.

```python
Loss = BCE(0.5·clip + 0.5·frame_max) + α · MSE(student_emb, teacher_emb)   # α = 1.0

h = self.backbone(x)
distill_emb = self.distill_head(h)      # GAP + Linear → teacher embedding dim
h_cls       = h.detach()                # ← the classification head does NOT update the backbone
```

The stop-gradient is the part that matters: without it the classification and distillation losses
fight over the backbone, and the notebook says so explicitly.

🔴 **But that mechanism assumes a trainable backbone, and §2 freezes ours.** In the source recipe
`h_cls = h.detach()` hands the backbone to the distillation loss *alone*. If the frontend is frozen
there is nothing for distillation to own, and the borrowed 0.898-vs-0.876 number is not evidence
for what we would be doing. Two honest options, and we must pick one rather than eliding it:

| Option | What distillation shapes | Cost |
|---|---|---|
| **Frozen frontend + adapters** (as §2) | adapters and heads only — a **weaker claim** than the source's | keeps one copy of weights in the zip; co-adaptation limited to adapters |
| **Unfreeze the truncated frontend** | the backbone, as in the source recipe | changes the 10 GB reasoning if branches need separate copies, and forfeits the LoRA-beats-full-FT result |

🔷 Our reading: the truncated frontend is small enough that unfreezing it during distillation is
affordable, and the two are not exclusive — distil into an unfrozen truncated frontend, then freeze
it and attach per-branch adapters for the joint stage. ⚠️ That is a *design proposal, not a
measured result*, and [09 B4](09-open-questions.md) already flags the adapter/joint-training
assumption as untested. Do not repeat the source's number as if it applied unchanged.

**Our teachers**, all training-only, none shipped:

| Teacher | For | Why it can be this big |
|---|---|---|
| XLS-R-1B / 2B | voice branch | ★ The AT-ADD runner-up's 0.3B+1B+2B fusion is "likely over budget" *at inference* — not at training |
| SSLAM / EAT-large | music branch | Highest AudioSet mAP found (0.502); mixture-pretrained |
| HT-Demucs v4 **residuals** | music branch | ⭐ ArtifactNet's Phase 1 — separation as a *teacher*, never in the inference path ([papers/02](../papers/02-music-detection.md)) |

🔷 The Demucs-as-teacher line is the one worth flagging. [03 G](03-candidates.md#g--frozen-separator--per-stem-detectors--rejected)
rejects separation at inference on four grounds; **none of them apply to a teacher.** The 44.1 kHz
domain mismatch still degrades the residuals we distil, so this is not free — but it costs runtime
we do not have to pay.

⚠️ **The honest risk.** ★ CtrSVDD's baselines are a warning against distilling into the wrong
student: with a graph-attention backend, **raw waveform 13.75% vs mel-spectrogram 25.19% EER** —
mel was nearly twice as bad ([survey/09](../survey/09-challenge-playbooks.md)). 🔷 So the student
should **not** be a log-mel CNN, however convenient. Keep a learned waveform-domain frontend and
compress it by truncation (§1); use distillation to recover what truncation costs, not to change
representation family.

⚠️ Two entries in our own index on this axis — DK-CAST and FTDKD — are marked *unverified, locate
before citing* ([papers/07](../papers/07-foundation-distillation.md)). Do not lean on them.

---

## 4. Precision and compilation

fp16 or bf16 inference and `torch.compile` are ~2× for no accuracy cost worth measuring. ★ The
BirdCLEF precedent went further (ONNX → fp16 OpenVINO) but that is CPU-oriented and we have a GPU
([kaggle/05 G7](../kaggle/05-transferable-playbook.md)).

🔷 **Our call on `torch.compile`: skip it unless measurement says otherwise.** Compilation happens
inside the 60-minute budget on first call, and with a corrected margin of ~3.7–10×
([07 §2](07-runtime-budget.md#2-estimated-line-items-)) we do not need the speedup. fp16 alone is
the safe win.

⚠️ Two further cautions specific to our submission contract:
- ❌ Anything that makes output depend on batch composition violates rule 2.4. Assert
  batch-invariance ([01 §1.4](01-design-envelope.md#14-rule-24--per-file-independence-)).
- ⚠️ fp16 **worsens the saturation risk** in [04 §7](04-heads-and-pooling.md#7-the-output-contract-).
  Run the model in fp16 if we like; compute and emit the final scores in **float64**.

---

## 5. The order to apply them

1. **Probe layer importance per frontend, per head** → set truncation depth. Largest win, cheapest
   to measure, and it sets the whole inference budget.
2. **Freeze frontends, attach adapters** → branch specialization inside one set of weights.
3. **Add the teacher ensemble with stop-gradient distillation** → recover truncation loss and
   import the ensemble we cannot run ([05 §4](05-multi-model.md#4-the-main-answer-ensemble-at-training-time-ship-one-model)).
4. **fp16 + compile**, measured against the real 60-minute clock.

🔷 Steps 1–3 are all *training-time* work on hardware we have in surplus. That is the point: the
compression story costs H200 hours, which are free, to buy L4 seconds, which are not.
