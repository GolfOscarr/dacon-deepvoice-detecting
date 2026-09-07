# 02 — AI-Generated Music Detection (music-fake head)

**Weight 0.27 — second highest in the metric. Least-solved literature. Our biggest opportunity.**

## 🔴 The 16 kHz finding

Published AI-music detectors are dominated by **production-pipeline artifacts**, not musical ones.
★ From *The AI Music Arms Race* (TISMIR), on 30,000 tracks (10k Suno / 10k Udio / 10k MSD):

| Artifact | Value |
|---|---|
| Suno bitrate | fixed **192 kbps** |
| Udio bitrate | fixed **320 kbps** |
| AI music sample rate | nearly all **48 kHz** (consistent with 24→48 kHz upsampling) |
| MSD (real) | ~140.9 kbps, mixed 44.1/48 kHz |
| Suno spectral centroid | **1,091 Hz** vs 1,501 Hz real |
| Most indicative features | *"standard deviation of the skewness of the mel bands"*, bark band kurtosis |

**And they are destroyed by resampling:**
- IRCAM Amplify *"misclassifies all Suno samples when downsampled to 22.05 kHz"* — despite
  F1 0.976/0.988 at baseline.
- All classifiers degrade below a 5 kHz low-pass cutoff, and collapse above a 10–12 kHz high-pass.

**Our test set is standardized to 16 kHz → Nyquist 8 kHz → every one of those cues is gone.**

Three consequences:
1. Off-the-shelf detectors (IRCAM, `lofcz/ai-music-detector`, `intrect/artifactnet`) are
   untrustworthy here. Do not use as a drop-in music head without 16 kHz validation.
2. **Our training data MUST pass the identical 16 kHz chain** (same resampler, same container
   re-encode) or we learn a shortcut that does not exist at test time.
3. It is a leveling event → careful work gains the most ground here.

## Cross-generator generalization is catastrophic

| Finding | Number | Source |
|---|---|---|
| ★ MERT-AASIST trained on one generator, tested on others | **46.4% EER** | MusicDET |
| ★ Systems trained on Suno/Udio, tested on Boomy | **3–12% detection** | TISMIR |
| ★ Udio → Suno transfer | F1 0.940–0.972 | TISMIR |
| ★ Suno → Udio transfer | F1 **0.629–0.778** | TISMIR |
| ★ In-distribution (CLAP embeddings + SVM) | F1 **>0.96** | TISMIR |

Transfer is **asymmetric**. In-distribution numbers are meaningless as a progress signal.

## MusicDET — the strongest published result, and its fragility

★ [MusicDET: Zero-Shot AI-Generated Music Detection](https://arxiv.org/html/2605.18072v1) (2026)

**Method**: band-wise **normalizing flows** over STFT energy spectrograms, trained **only on real
music**, combined by a global flow to a Gaussian prior. No pretrained audio model. Generator-
agnostic by construction.

| Benchmark | MusicDET | Baselines |
|---|---|---|
| FakeMusicCaps | **4.51% EER** (class-conditional: **0.89%**) | 17.61–23.27% |
| SONICS | **2.89% EER** | 2.66–18.98% |

⚠️ **Robustness section is a direct warning for us**:
- pitch shift → **+40 percentage points EER**
- **MP3 @ 64 kbps → +37 points**

Our test data is MP3/WAV/FLAC at 16 kHz. That is exactly the failure regime. **Do not adopt
zero-shot density modeling without testing it under our own degradation chain first** — but the
real-only training idea is still worth stealing, since it sidesteps generator overfitting.

## Other notable work

| Work | Contribution | Relevance |
|---|---|---|
| ★ [SONICS / SpecTTTra](https://arxiv.org/abs/2408.14080) (ICLR 2025) | Long-range temporal dependency modeling for full songs; +8% F1 over ViT on long songs, 38% faster, 26% less memory. [code](https://github.com/awsaf49/sonics) | Our clips cap at 60s → mild effect, but argues against single-crop classification |
| ☆ [Broadcast monitoring](https://arxiv.org/pdf/2602.06823) (2026) | Trains on pristine **and** degraded variants: MP3/AAC, radio processing, loudness normalization | Closest published setup to degraded real-world AI-music detection |
| ☆ [Fake Music Detection Under Audio Augmentations](https://arxiv.org/pdf/2507.10447) | Systematic degradation study; conclusion: **train with augmented audio** | Confirms 08 |
| ☆ [Segment Transformer](https://arxiv.org/pdf/2509.08283) | Detection via music structural analysis | Structure-level cues survive resampling better than spectral fingerprints — worth a look |
| ☆ [Explainable detection of machine generated music](https://www.nature.com/articles/s41598-026-42133-7) (Sci Rep 2026) | Interpretability | Feeds the 2nd-stage report (15 pts) |

## Off-the-shelf detectors (treat as baselines to beat, not as components)

| Model | Notes |
|---|---|
| ☆ [`lofcz/ai-music-detector`](https://huggingface.co/lofcz/ai-music-detector) | "fakeprint" method, **resamples to 16 kHz** — the one that may actually survive our chain. Verify license. |
| ☆ [`intrect/artifactnet`](https://huggingface.co/intrect/artifactnet) | 4.2M params, claims generalization across 22 AI music generators. **CC BY-NC 4.0** — usable per DACON's answer, but verify redistribution. |
| IRCAM Amplify | Commercial. F1 0.976/0.988 baseline; **breaks at 22.05 kHz**. |

## Datasets → see [06-datasets.md](06-datasets.md)

SONICS (97k songs), FakeMusicCaps (~27.6k clips / 5 TTM models), M6, MUSDB18-HQ (stems).
⚠️ SONICS reals are YouTube-sourced; FakeMusicCaps reals are MusicCaps (YouTube-linked).
