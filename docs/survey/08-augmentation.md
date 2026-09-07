# 08 — Augmentation & Channel Simulation

Second-biggest lever after the SSL frontend. Also the **cheapest compliance win**: DACON
confirmed on-the-fly augmentation intermediates need **not** be shipped — originals + code +
config + **seed** suffice. So: persist sources, augment on the fly with fixed seeds.

## What the winners actually used (AT-ADD top systems)

**Signal-level**: MUSAN noise · RIR reverberation · codec / re-encoding artifacts · quantization ·
dynamic-range compression · pitch shifting · time stretching · resampling · replay simulation

**Segment-level**: random cropping · bonafide segment concatenation · silence insertion

## RawBoost — the strongest single family

☆ In a systematic comparison of AWGN / RawBoost / vocoded / RIR augmentation, **RawBoost
configurations were the most effective.** It models nuisance variability from encoding,
transmission, microphones and amplifiers, plus linear and nonlinear distortion, directly on the
raw waveform. [code](https://github.com/TakHemlata/RawBoost-antispoofing)

## 🔴 Telephone-channel simulation (test set explicitly contains it)

The **UR channel-robust ASVspoof 2021 system** generalized well by simulating three transmission
degradations: **landline, cellular, VoIP** — reduced bandwidth to 8 kHz with **OPUS**.
[paper](https://arxiv.org/pdf/2107.12018)

Recipe to implement (all available via preinstalled `ffmpeg` / `torchaudio`):

| Stage | Options |
|---|---|
| Band-limit | downsample 16k → **8k** → upsample back to 16k (this is the key transform) |
| Telephony codecs | G.711 μ-law/A-law, **AMR-NB**, **OPUS** (narrowband/wideband), GSM, G.722 |
| Packet loss | random frame drop / repeat (ADD-C models this) |
| Media codecs | MP3 (multiple bitrates incl. **64 kbps**), AAC, OGG/Vorbis, FLAC round-trip |
| Level | loudness normalization, DRC, clipping |
| Channel | RIR convolution, MUSAN noise at varied SNR |

⚠️ **MP3 @ 64 kbps cost MusicDET +37 EER points** ([02](02-sota-music.md)). Our test set is
MP3/WAV/FLAC. Codec augmentation is not optional for the music head.

## 🔴 The 16 kHz normalization chain

Whatever the organizers did to standardize to 16 kHz, our training data must experience the
same thing. Otherwise we learn cues that don't exist at test time.

- Resample **everything** to 16 kHz with a fixed, documented resampler (`soxr` is preinstalled).
- Round-trip through the same container formats the test set uses (MP3 / WAV / FLAC).
- Handle mono **and** stereo (test set has both per-sample).
- ⚠️ Do **not** let source sample rate, bitrate, or channel count leak as a label cue. Randomize
  them independently of the real/fake label.

## Label-preserving vs label-changing — the boundary

Per the competition rules, these are **REAL-preserving** (must be trained as REAL):
quality enhancement, **noise removal**, volume adjustment — anything that does not regenerate
the voice/music component.

These are **FAKE**: vocoder resynthesis, neural codec resynthesis, TTS, VC, SVS, TTM.

⚠️ Neural denoisers/enhancers are themselves generative and leave similar traces. **Include
denoised-real audio labeled REAL** or we will systematically false-positive on clean-up
processing. This is likely a deliberate trap in the test set.

## Tools

| Tool | Use |
|---|---|
| [RawBoost](https://github.com/TakHemlata/RawBoost-antispoofing) | Raw-waveform nuisance simulation |
| [audiomentations](https://github.com/iver56/audiomentations) / [torch-audiomentations](https://github.com/iver56/torch-audiomentations) | General augmentation, GPU-capable |
| `torchaudio` effects/codec | ★ Preinstalled (2.7.1). Applies effects, filters, RIR and codecs — has a documented "noisy speech over phone" recipe |
| `ffmpeg` | ★ Preinstalled system package. All codec round-trips |
| MUSAN / RIRS_NOISES | Noise + reverberation corpora ([06](06-datasets.md)) |

## Inference-time (not training) — multi-crop

AT-ADD winners: **4–5 temporal crops**, aggregated by **median/mean pooling** or weighted fusion.
Consistently improved robustness. Cheap. ⚠️ But for *fake-component* detection, "any part fake →
fake" argues for **max/attention** pooling, not median. Test both — they may differ per head.
