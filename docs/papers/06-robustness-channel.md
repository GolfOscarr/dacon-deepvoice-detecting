# 06 — Robustness: Codec, Channel, Bandwidth, Laundering

Our test set is 16 kHz, MP3/WAV/FLAC, with telephone-channel samples. This axis is the one whose
findings we can act on immediately.

---

### ⭐ D9 · Low Pass Filtering and Bandwidth Extension Against Codec Variabilities (2022) `[foundational]`
[arXiv](https://arxiv.org/abs/2211.06546) · ★ abstract-level

**Contribution** — Directly targets narrowband/codec mismatch. Core finding: *"using the
low-frequency subbands of signals as input can mitigate the negative impact introduced by codecs
on the countermeasure systems."* Two low-pass filters with different cut-offs were applied.

**Result** — **EER reduced by up to 25% relative** under codec conditions.

**For us** 🔴 — The most directly applicable finding on this axis. Our audio is *already*
band-limited to 8 kHz by the organizers, and some of it further to telephone band. Deliberately
**restricting the model's input to low-frequency subbands** is a codec-robustness strategy that
costs nothing and is exactly aligned with what our data actually contains. Worth an early ablation:
full 0–8 kHz vs a restricted low band.

⚠️ Older paper; retained under the `[foundational]` exception because the finding is condition-
specific rather than architecture-specific, so it has not been superseded.

---

### ⭐ D12 · Measuring the Robustness of Audio Deepfake Detectors (2025) — `card`
[arXiv](https://arxiv.org/abs/2503.17577) · ☆ abstract only

**10 detectors × 18 corruption types**, grouped into three categories. Headline finding: models are
*"vulnerable to audio modifications and compression, especially neural codecs."*

**For us** ⚠️ — the per-corruption numbers are not in the abstract and I could not extract them
from the listing. **Fetch the full text before relying on any specific figure.** The qualitative
finding — neural codecs are the worst corruption class — matters because our own T3 resynthesis
plan ([data/05](../data/05-synthesis-plan.md)) uses neural codecs as a *fake generator*, while this
says they are also the most damaging *corruption*. Those two roles must not be confounded in the
corpus, or codec artifacts become a label shortcut.

---

## High-value cards on this axis

| Finding | Paper |
|---|---|
| **41.4% relative EER reduction**, codec-trained vs vocoder-trained | [Age of Advanced TTS](https://arxiv.org/abs/2601.20510) |
| **EER improves 10–15 points at 10→0 dB SNR** after multi-condition fine-tuning | [Noise-Aware ADD](https://arxiv.org/abs/2512.13744) |
| **35 augmentation types across 11 categories** (codecs, noise, reverb, filtering, VoIP sim) — usable as our augmentation library | [Proteus](https://arxiv.org/abs/2606.29544) |
| 6 real speech codecs benchmarked: AMR-WB, EVS, IVAS, OPUS, Speex-WB, SILK | [ADD-C](https://arxiv.org/abs/2504.12423) |
| 31 open codecs as training proxy → detects 17 unseen CoSG systems | [CodecFake+](https://arxiv.org/abs/2501.08238) |
| Proxy-to-wild gap: codec-resynthesis training data ≠ real wild deepfakes | [Mitigating Proxy-to-Wild](https://arxiv.org/abs/2606.07494) |
| RawBoost: **27% relative improvement** over raw E2E baseline `[foundational]` | [RawBoost](https://arxiv.org/abs/2111.04433) |
| ASVspoof 5 top systems: SpecAugment + RawBoost + codec compression + speed perturbation | [SZU-AFS](https://arxiv.org/abs/2408.09933) |

⚠️ **The proxy-to-wild caveat is important for us.** Our `S-S1` T3 twins are codec-resynthesis —
exactly the "proxy" that paper warns does not match wild deepfakes. Keep T3 as a *supervision
signal and leak detector*, not as the whole fake pool.
