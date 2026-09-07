# 01 — Detecting AI-Generated Content

Competitions whose *structure* matches ours: no training data provided, generalization to unseen
generators, heavy public→private shift.

---

## LLM - Detect AI Generated Text (Kaggle, 2023–24) ⭐ closest structural analogue

**Setup**: classify student essays as human-written or LLM-generated. Organizers provided almost
no training data. Test set written by unseen models. Massive CV/LB divergence.
[competition](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)

### 1st place — `rbiswasfc` ★ [repo](https://github.com/rbiswasfc/llm-detect-ai)

**Data generation was the solution.** They fine-tuned a wide set of LLMs with a causal-LM
objective on the in-domain corpus (PERSUADE) so the generated text matched the *domain*, not just
the task:

> Mistral · LLaMA-13B · Falcon · MPT · OPT · GPT-2 · Bloom · Pythia · TinyLlama

plus **instruction-tuned LLMs used to generate adversarial essays**.

**Three complementary model types**, not one:
1. (Q)LoRA-fine-tuned LLM classifiers
2. `deberta-v3-large` trained with a **ranking loss**
3. An **embedding model trained with supervised contrastive loss**, using KNN neighbours from
   test-set samples

**Transfer to us** 🔴:
- Fine-tuning generators *on the target domain* before generating fakes is the key move. Our
  analogue: clone real speakers from our real-voice pool (T2/T3 in [data/05](../data/05-synthesis-plan.md))
  and prompt music generators with tags describing our real tracks — not generic prompts.
- **Ranking loss + supervised contrastive** on generated/real pairs is exactly what we proposed;
  a Kaggle winner independently arrived at both. Confirms [data/05 paired design](../data/05-synthesis-plan.md#using-the-pairs-in-training).
- Adversarial generation (deliberately make hard fakes) is a step we had not planned. Worth adding.

### Efficiency-prize / high-rank lesson ☆ [discussion 471898](https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/471898)

> Advanced NLP/LLM models weren't working on the public LB; the hypothesis was that the
> competition needed **a dataset of similar distribution**. … **Adding other data caused severe
> data drift**, which further increased the CV/LB gap.

The winning final submission was a *simple* baseline over carefully curated, distribution-matched
data. Time went into **finding the right data**, not the model.

**Transfer to us** 🔴: more data is not automatically better. Every pool addition must be checked
for drift against our test-chain-normalized distribution. This directly justifies the shortcut
audit in [data/07](../data/07-eda-plan.md).

---

## Deepfake Detection Challenge (DFDC, 2019–20)

$1M prize, 100k+ clips from 3,426 paid actors, multiple generation methods. Winner scored 65.18%
on the black-box set — a huge drop from public LB, the canonical "generalization to unseen
manipulations" cautionary tale.

### 1st place — Selim Seferbekov ★ [repo](https://github.com/selimsef/dfdc_deepfake_challenge)

**Architecture**: MTCNN face detection → EfficientNet-B7 (ImageNet + **Noisy Student** pretrained),
380×380 input, 5 models with different seeds, synchronized BN, separate fake/real loss terms
within each batch.

**Augmentations** (the generalization strategy):

| Type | Detail |
|---|---|
| Compression | `ImageCompression(quality_lower=60, quality_upper=100, p=0.5)` |
| Noise/blur | Gaussian noise, blur |
| Geometric | h-flip, rotate ±10°, scale ±20%, shift ±10% |
| Color | brightness/contrast, fancy PCA, hue/saturation |
| Grayscale | p=0.2 |
| **Structured dropout** | Cutout-like removal of face parts, inspired by GridMask — forces the model off any single artifact region |
| Isotropic resize | **varied interpolation methods** |

**🔴 The aggregation finding**: frame-by-frame classification beat more complex temporal models,
and a confidence-based aggregation heuristic — not a mean — did the heavy lifting:

> mean of predictions >0.8 if more than 11 frames exceed it and they are >40% of frames;
> mean of predictions <0.2 if >90% of frames are below it; otherwise plain mean.

This alone moved a standalone B5 from **0.25 → 0.22**.

**Transfer to us** 🔴:
- **Structured dropout is the anti-shortcut tool.** Our analogue: randomly mask frequency bands or
  time regions so the model can't rely on one band. Stronger than plain SpecAugment because it is
  *targeted at artifact regions*.
- **Varied interpolation during resize** ↔ our varied resamplers/codecs. Same idea: don't let one
  processing path become the signal.
- **Our segment→file pooling should be a tuned heuristic, not a mean.** DFDC's confidence-gated
  rule is a concrete template for `FILE_FAKE_PROB` ([data/02](../data/02-label-taxonomy.md)).

---

## Adjacent: dedicated audio-deepfake challenges (not Kaggle, same rigor)

### RADAR Challenge 2026 — Robust Audio Deepfake Recognition under Media Transformations
★ [arXiv 2605.09568](https://arxiv.org/pdf/2605.09568) · APSIPA Grand Challenge

Applies exactly our degradation set — **compression, resampling, additive noise, reverberation** —
and evaluates multilingual (English, Singapore English, Mandarin, Taiwanese Mandarin, Japanese,
Vietnamese), >100k utterances. Finding: systems trained on clean audio **degrade substantially**
under these transforms; the stated remedies are training with the target transformations and
evaluating across conditions rather than on clean audio only.

### SAFE — Synthetic Audio Forensics Evaluation ★ [arXiv 2510.03387](https://arxiv.org/pdf/2510.03387)

Real speech (LibriVox etc.) vs. **contemporary TTS**: Cartesia, ElevenLabs, Kokoro, Parler-TTS,
MetaVoice, OpenAI TTS. Tests **audio laundering**: Ogg/OPUS/semantic-codec compression, resampling
and bandwidth reduction, environmental noise, temporal modification.

Findings: strong on matched conditions, **significant degradation on unseen TTS**;
*"domain mismatch represents the primary challenge"*; **codec-aware preprocessing improves
robustness**; no single approach generalizes uniformly.

### ESDD 2026 — Environmental Sound Deepfake Detection
☆ [arXiv 2508.04529](https://arxiv.org/pdf/2508.04529) — non-speech deepfake detection, relevant
to the non-voice half of our problem.

**Transfer to us**: these three independently confirm that **codec/bandwidth augmentation is the
single highest-value robustness intervention** for audio deepfake detection — the same conclusion
reached in [survey/08](../survey/08-augmentation.md) and [data/06](../data/06-augmentation-spec.md).
