# 01 — Speech Deepfake Detection (voice-fake head)

Most mature of the four literatures. Five ASVspoof challenge cycles. Treat as solved-ish
methodology; our work is data, not architecture.

## Canonical pipeline (stable since ~2022)

```
raw waveform → SSL frontend (frozen or LoRA) → attention/graph backend → score
```

## SOTA leaderboard

| System | Venue | ASVspoof21 LA | 21 DF | In-the-Wild | Code |
|---|---|---|---|---|---|
| ★ **Fake-Mamba** | ASRU 2025 | **0.97%** | **1.74%** | **5.85%** | [github](https://github.com/xuanxixi/Fake-Mamba) |
| ☆ HierCon | 2025 | – | 1.93% | 6.87% | – |
| ★ XLSR-Mamba | 2024 | – | – | 6.71% | [paper](https://arxiv.org/pdf/2411.10027) |
| ★ XLSR-MamBo | ACL 2026 Findings | – | – | – | [github](https://github.com/saki-ciallo/XLSR-MamBo) |
| ★ XLS-R + SLS | ACM MM 2024 | – | – | – | [paper](https://dl.acm.org/doi/10.1145/3664647.3681345) |
| ☆ Wav2DF-TSL | 2025 | – | – | – | [paper](https://arxiv.org/html/2509.04161) |

HierCon reports **+22.5% relative over XLS-R+SLS**. Note the trend: In-the-Wild has moved
6.87 → 6.71 → 5.85 in ~18 months. **Backbone innovation yields ~1 point/year. Data yields more.**

## ASVspoof 5 (2024) — the reference for unseen generators

★ Crowdsourced data, many speakers, diverse acoustics, adversarial attacks included.

| | minDCF | EER |
|---|---|---|
| Baselines | >0.7 | **>29%** |
| Top-5 submissions | <0.5 | **<15%** |
| Best individual results | 0.115 / 0.1348 | **4.04% / 5.02%** |

★ Organizers' own conclusions:
- *"most of the well-performing submissions use features extracted by pre-trained
  self-supervised learning (SSL) models"*
- *"submissions using an ensemble of sub-systems tend to perform better"*

⚠️ The **29% baseline EER** is the number to internalize: a competent-looking system with the
wrong data is barely better than chance on unseen generators.

## What generalizes / what doesn't

| Lever | Evidence | Strength |
|---|---|---|
| SSL frontend | ASVspoof 5, AT-ADD, every SOTA paper | ⭐⭐⭐⭐⭐ |
| Augmentation breadth | ☆ RawBoost best among AWGN/RawBoost/vocoded/RIR | ⭐⭐⭐⭐ |
| Ensembling / fusion | ASVspoof 5, AT-ADD top-5 | ⭐⭐⭐ |
| Segment-level supervision | PartialSpoof | ⭐⭐⭐ |
| Backbone (Mamba/Conformer/AASIST) | ~1 pt/yr on In-the-Wild | ⭐⭐ |

## Known failure modes / shortcuts

| Shortcut | Note |
|---|---|
| ★ **Vocoder/codec fingerprints** | "Neural Encoding Detection is Not All You Need" (2026): detectors latch onto frequency distortion + quantization patterns of *specific* synthesis families and collapse on others |
| ☆ **Silence / leading-trailing padding** | Classic ASVspoof artifact ("The Impact of Silence on Speech Anti-Spoofing") |
| **Bandwidth / sample-rate** | See [02](02-sota-music.md) — same disease in music, more acute |

## Codec-based fakes — mandatory coverage

★ **Codecfake** (1,058,216 samples; reals from LibriTTS + VCTK; fakes from **7 neural codecs**:
SoundStream, SpeechTokenizer, FunCodec, EnCodec, AudioDec, AcademicCodec, DAC).

> Codec-trained ADD models show a **41.406% reduction in average EER** vs vocoder-trained models
> on the Codecfake test set.

Rationale: LLM-style TTS generates directly from discrete neural codecs, **skipping the vocoder
stage entirely**. A vocoder-artifact detector is structurally blind to it.

⚠️ Tension to manage: train *on* codec artifacts (Codecfake) but do not *depend* on them
(2604.16700). Resolution is breadth of generator families, not one artifact type.

## Telephone channel (our test set contains it)

☆ ASVspoof 2021 **LA** passed audio through telephony/VoIP with varied coding and transmission.
Effects: 8 kHz sampling loses everything above 4 kHz; plus codec + packet-loss artifacts.

- **UR channel-robust system** (ASVspoof 2021): augmented with three transmission degradations —
  landline, cellular, VoIP (8 kHz bandwidth, OPUS codec). [paper](https://arxiv.org/pdf/2107.12018)
- **ADD-C**: newer test set for VoLTE/VoIP conditions with codec combinations + packet loss.
  [paper](https://arxiv.org/html/2504.12423v1)

→ See [08-augmentation.md](08-augmentation.md) for the recipe.

## Partial / segment-level spoofing

★ **PartialSpoof**: fake segments embedded in bona fide utterances, labeled at **20–640 ms**
resolutions. Countermeasure trains segment-level **and** utterance-level labels simultaneously
(multi-resolution) on an SSL frontend → **0.77% utterance EER** (PartialSpoof), 0.90% (ASVspoof19 LA).
[paper](https://arxiv.org/pdf/2204.05177) · [code](https://github.com/nii-yamagishilab/partialspoof)

**Why it matters here**: our files mix voice and music **sequentially**, and a fake component may
occupy a small fraction of a 60s file. Multi-resolution segment supervision + max/attention
pooling is the established answer. Whole-file mean pooling dilutes.

## Resources

- [awesome-fake-audio-detection](https://github.com/john852517791/awesome-fake-audio-detection) — maintained paper/code list
- [AUDDT benchmark toolkit](https://arxiv.org/pdf/2509.21597) — evaluate any pretrained model across many datasets
- [deepfake-total.com](https://deepfake-total.com/related_work/) · [Fraunhofer AISEC related work](https://deepfake-demo.aisec.fraunhofer.de/related_work/)
- Surveys: [Audio Deepfake Detection: A Survey](https://arxiv.org/pdf/2308.14970) · [A Survey on Speech Deepfake Detection](https://arxiv.org/pdf/2404.13914)
