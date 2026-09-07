# 04 — Component-Level, Partial & Localized Detection

The axis that matches our two-fake-head structure most exactly. Two 2025–26 papers solve nearly
our problem and **reach different architectures** — that disagreement is the most useful finding
in this file.

---

### ⭐ D4 · PC-Mix: Partial-Component Audio Spoofing Detection under Mixed Conditions (2026)
[arXiv](https://arxiv.org/html/2607.10345) · ★

**Field** — Localized spoofing in speech, environmental sound, or both, within a mixture. 16 kHz.

**Contribution** — **Three parallel branches on the raw mixture, no separation**: speech branch,
env-sound branch, mix branch (distinguishes constructed mixtures from originals). Each emits
multi-resolution frame scores **and** an utterance score. Two-stage training: independent, then
joint with component losses masked to constructed mixtures (`λ_s = λ_e = λ_m = 1`).

Labels: 5 utterance classes (original / bona-bona / spoof-bona / bona-spoof / spoof-spoof) and 4
frame classes, at **40/80/160/320/640 ms** resolutions. 126,586 clips / 140.65 h. Train SNR 8 dB,
eval 10 dB.

**Result** 🔴 — cross-condition transfer (Table VI), utterance EER:

| Detector | On mixed audio | On separated streams |
|---|---|---|
| Speech | **29.35%** | **51.38%** ← separation much worse |
| Env sound | 44.80% | 22.89% ← separation better |

Matched training on PC-Mix (Table VII) narrows it: speech 8.53% (mixed) vs 7.83% (separated);
env 3.57% (mixed) vs 7.43% (separated).

**Joint-training ablation** (Table VIII), independent → joint:
- Env utterance: EER 3.59→3.12%, **F1 76.41→92.67 (+16.26)**, **ACC 69.36→92.17 (+22.81)**
- Speech utterance: EER 8.72→7.86%
- **5-class utterance ACC 69.40 → 85.12 (+15.72)**

**For us** ⭐ — The branch layout maps one-to-one onto our `VOICE_FAKE` / `MUSIC_FAKE` / `FILE_FAKE`
heads, at our sample rate, with our label structure. **Joint training across branches is the
single largest reported gain.** Their stated reason for avoiding separation —
*"source separation may introduce artifacts or remove environmental-sound spoofing cues"* —
matches the hybrid-stems evidence.

**Limitations (stated)** — frame-level localization much harder than utterance-level; env branch
degrades most under **background-domain shift** (E3: 0.67% → 9.15% EER); macro-F1 lags weighted-F1
from class imbalance.

---

### ⭐ D5 · CompSpoof: Dataset and Joint Learning Framework (2025)
[arXiv](https://arxiv.org/html/2509.15804v2) · ★ · dataset itself → [data/11](../data/11-source-inventory.md)

**Contribution** — The opposite architecture: **UNet separation jointly trained** with anti-spoofing.
Four models — mixture detector (XLSR-AASIST), UNet complex-mask separation in the STFT domain
(plus an adaptive soft-mask to suppress speech leakage), and dedicated XLSR-AASIST detectors per
component. Independent for 4 epochs, then **joint from epoch 5**. Segment predictions aggregate by
majority vote.

**Result** 🔴 — eval F1:

| Configuration | Overall F1 |
|---|---|
| Direct 5-class baseline | 0.827 |
| Separation **without** joint learning | **0.668** ← worse than not separating |
| Separation **+ joint learning** | **0.908** |

Segment level with JL: speech 0.863, env 0.849. Without JL: speech 0.720, env 0.718.

**For us** 🔴 — This is the precise refinement of "should we separate?". **Separation alone hurts
(0.668 < 0.827); separation plus joint training wins (0.908).** The failure mode is *naive
pipelining*, not separation itself — consistent with the hybrid-stems 38–94.7% FPR result.

**Limitation (stated)** — *"environment anti-spoofing consistently performs worse than speech
anti-spoofing"*, suggesting XLSR-AASIST is a poor fit for non-speech components. For us that
argues for a **separate music/general-audio frontend** rather than reusing the speech SSL encoder
on the music head.

---

## 🔴 The synthesis

| Claim | Evidence |
|---|---|
| **Never separate-then-detect as a pipeline** | CompSpoof 0.668 vs 0.827 · PC-Mix speech 51.38% vs 29.35% EER · hybrid-stems 38–94.7% FPR |
| Either skip separation (3 branches on the mixture) **or** train it jointly end-to-end | PC-Mix vs CompSpoof — both beat the naive middle |
| **Joint training across component branches is the largest single gain** in both papers | PC-Mix +15.72 ACC · CompSpoof 0.668→0.908 F1 |
| Don't reuse a speech SSL encoder for the non-speech component | CompSpoof stated limitation |
| Multi-resolution frame + utterance supervision is the standard | PC-Mix, PartialSpoof |
| ⭐ **A third option: use a separator as a residual *teacher*, keeping it out of the inference path** | ArtifactNet distils Demucs v4 residuals in Phase 1 → [02](02-music-detection.md) |

This supersedes the flat "don't separate" guidance in [survey/03](../survey/03-sota-singing-mixed.md).
PC-Mix's three-branch design is the cheaper of the two to try first.

---

Remaining papers on this axis (localization, boundary-aware attention, one-class localization,
temporal pyramids) are listed in [INDEX](INDEX.md).
