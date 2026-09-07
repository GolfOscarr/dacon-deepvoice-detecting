# 06 — Augmentation Method Catalog (tiered)

Implemented as **on-the-fly, seeded transforms** over the four pools — never as a pre-rendered
corpus. That is both a training win (unbounded variety from a small pool) and a compliance win:
DACON confirmed (#417280) that augmentation intermediates need not be submitted if we provide
originals + code + config + seed.

Tier legend in [07](07-eda-plan.md#tier-legend-used-in-05-06-07).

## 🔴 The governing rule

> **Every transform must be applied independently of the label.** For every transform `T` and
> label `L`: `P(T | L) = P(T)`.

If only fakes get concatenated, only reals get denoised, or fake music is systematically louder,
the model learns the *transform*, not the artifact. Local CV soars; the leaderboard doesn't move.
Enforce it in code — sample transforms from a label-independent RNG stream — and **verify it** via
the shortcut audit ([07 E-S2](07-eda-plan.md)).

## Pipeline order (per training sample)

```
0. draw target duration in [4, 60] s → the timeline everything is placed on
1. sample cell (1–9)                 → which pools to draw from
2. draw components (split-safe)      → labels are fixed here, and nowhere else
     …including any MixUp partner    → see below
3. structural composition            → overlap (gain ratio) or sequence (crossfade)
4. label-independent signal aug      → the menu below, minus MixUp
5. TEST-CHAIN NORMALIZATION          → always last
```

Steps 3–5 are shared by every cell. **Labels come from steps 1–2 only.**

⚠️ **Duration is drawn first, not cropped last.** An earlier version cropped to `U(4,60)` s as a
final step, which renders audio only to discard it — expensive given the codec round-trips in
`A-S3` — and makes frame-level targets ([architecture/04 §4](../architecture/04-heads-and-pooling.md))
a post-hoc re-slice instead of exact by construction. Drawing the timeline first makes component
placement and frame targets the same arithmetic.

### ✅ The resampler is `scipy.signal.resample_poly`

⚠️ This stage runs in the **shipped** preprocess path, so train/test symmetry depends on it
([pipelines/03](../pipelines/03-transforms.md)). `soxr` was named here originally, but the
requirement is *"one fixed resampler applied identically in train and test"*, not a specific
library — and `scipy` is already a dependency while `soxr` is not. Adding a shipped dependency
costs against the submission's offline-install budget
([architecture/01](../architecture/01-design-envelope.md)) and buys quality we cannot measure.

⚠️ **`A-B2` (varied resampler kernels) is unbuildable as written** — it lists `soxr` HQ/VHQ,
`librosa` and `torchaudio` kernels, none of which are installed. If that augmentation is wanted,
respecify it against `resample_poly` window/filter variants.

🔴 **Revisit once `G1` lands.** If the dummy-file forensics reveal which resampler the organizers
used, *matching them* is worth more than kernel quality, and the stage is written with the
resampler injectable for exactly that reason.

### 🔴 MixUp belongs in step 2, not step 4

`A-A1` and `A-A2` were listed below as step-4 augmentation while simultaneously specifying
*"update labels for whatever the mixed-in audio contributes"* and *"labels `np.maximum`"*. Those
cannot both hold: step 4 is by definition label-independent, and **labels come from steps 1–2
only.** Mixing another file in changes which components are present and whether they are
generated, so it is a **component draw**, not a signal transform.

Treating MixUp as an extra draw in step 2 keeps the invariant true, keeps the `P(T | L) = P(T)`
audit valid over transforms, and makes the union-label rule a consequence of the cell arithmetic
rather than a special case bolted onto an augmentation. The `p`, `Beta`/`alpha` and `max`-label
parameters in `A-A1`/`A-A2` are unchanged — only the stage they run in.

---

## Tier S

| # | Method | What it does / why | Parameters & output |
|---|---|---|---|
| **A-S1** | 🔴 **Test-chain normalization** — resample to 16 kHz with one fixed resampler (**`scipy.signal.resample_poly`** — decided; see below), round-trip through MP3/WAV/FLAC, emit mono **and** stereo, all randomized **independently of label** | Whatever the organizers did to standardize, our data must experience it. Otherwise we learn cues that don't exist at test time. This is the single highest-leverage step in the whole pipeline ([survey/02](../survey/02-sota-music.md)) | Parameters come from **E-S1** `signal_chain.yaml`. ⚠️ Never let source sr, original bitrate, or channel count correlate with label |
| **A-S2** | **Label-independent RNG discipline** — one transform-RNG stream keyed on `hash(sample_id, epoch, global_seed)`, never on label | Mechanism enforcing the governing rule; also gives byte-reproducibility for the 2nd-stage submission | CI test: same seed → byte-identical rendered sample ([09 R9](09-risks-and-checks.md)) |
| **A-S3** | **Codec round-trip** | MP3 at **64 kbps** cost MusicDET **+37 EER points** ([survey/02](../survey/02-sota-music.md)); the test set is MP3/WAV/FLAC. Confirmed as the top robustness intervention by RADAR 2026 and SAFE ([kaggle/01](../kaggle/01-ai-content-detection.md)) | MP3 {64, 96, 128, 192}, AAC, OPUS, OGG, FLAC round-trip. p ≈ 0.5 |
| **A-S4** | **Telephone chain** — 16k → **8k** → 16k, plus G.711 μ/A-law, AMR-NB, OPUS-NB, GSM, G.722; optional packet loss | The test set explicitly contains 전화채널 audio. The UR channel-robust ASVspoof 2021 system generalized by simulating landline/cellular/VoIP at 8 kHz with OPUS ([survey/01](../survey/01-sota-speech.md)) | p ≈ 0.2. Log which chain was applied for per-condition error analysis |

---

## Tier A

| # | Method | What it does / why | Parameters & output |
|---|---|---|---|
| **A-A1** | 🔴 **Cross-domain MixUp** — mix self-generated clean audio with **real degraded** audio (ASVspoof21 LA telephony, MUSAN, real Jamendo tracks) | ★ `[BC2026 Distilled-SED]` runs "Focal–Soundscape MixUp" explicitly to *"bridge the domain gap between clean focal audio and noisy real-world soundscapes."* **This is legal domain bridging where pseudo-labeling is not** — it never touches DACON's test set | `p=0.5`, `Beta(0.4, 0.4)`. 🔴 **Runs in step 2, not step 4** (see above) — the mixed-in audio is a component draw, and labels update from it |
| **A-A2** | **Same-type MixUp, hard/union labels** | ★ `[BC2026]` `MIXUP_HARD=True` → `lb = max(lb1, lb2)`; ★ `[BirdCLEF 2024, 3rd]` "mixed labels are the max of the labels of the two audios". Three independent sources agree on `max` — and it matches our OR-over-components semantics | `MIXUP_PROB=0.5`, `MIXUP_ALPHA=0.4`, labels `np.maximum`. 🔴 **Step 2, not step 4** (see above) |
| **A-A3** | 🔴 **Low-SNR-skewed component mixing** | ★ `[G2Net 2021, 3rd]`: models generalize **low-SNR → high-SNR but not the reverse**; they sampled `SNR ~ max(N(3.6,1),1)` for 2–8 bps. The hybrid-stems paper confirms detection tracks stem energy (vocals 65–80% TPR vs accompaniment 97–98%) | **Changed from uniform**: skew voice/music gain ratio toward the quiet end rather than `U(−15,+15)`. Log the ratio per sample for the accuracy-vs-SNR curve (**E-B6**) |
| **A-A4** | **Sequential composition with sigmoid crossfade** | The rules say voice and music may appear **순차적으로**. ★ `[Freesound 2019, 1st]` `SigmoidConcatMixer` gives a *"smooth (sigmoid-based) transition from one audio-clip to another over time"* — better than hard cuts, which become a splice shortcut | 2–4 segments, order permuted, sigmoid crossfade (plus some hard cuts and silent gaps). ⚠️ **Must be applied to REAL-REAL pairs at the same rate** |
| **A-A5** | **RawBoost** | ☆ Most effective single augmentation family in a systematic comparison against AWGN / vocoded / RIR ([survey/08](../survey/08-augmentation.md)). Models encoding, transmission, microphone and nonlinear distortion on the raw waveform | Linear, nonlinear and stationary variants. p ≈ 0.5 |
| **A-A6** | **Additive noise (MUSAN noise + speech partitions)** | ★ `[BC2026]` uses SNR **10–30 dB** at p=0.5. ⚠️ **Never MUSAN's *music* partition** — it flips `MUSIC_PRESENT` to 1 | `AUG_NOISE_SNR_DB_RANGE = (10, 30)`, p=0.5 |
| **A-A7** | **Gain jitter** | ★ `[BC2026]` ±6 dB, "simulates varying recording distances". Note how **modest** this is — MixUp does the heavy lifting, not signal mangling | `AUG_GAIN_DB_RANGE = (-6, +6)`, p=0.5 |
| **A-A8** | **Time shift** | ★ `[BC2026]` ±0.5 s, "simulates temporal misalignment" | p=0.5 |
| **A-A9** | **SpecAugment** | ★ `[BirdCLEF'25 B0]`: 1–3 masks, each 5–20 frames/bins, p=0.5 each for time and frequency. ★ `[Freesound 2019, 1st]`: 2 masks, 15% freq / 20% time, p=0.5 | Either parameterization; ablate |
| **A-A10** | **RIR convolution** | Standard channel simulation; ☆ birdcall community used p=0.2 with a **dry/wet mix control** | RIRS_NOISES, p ≈ 0.3 |
| **A-A11** | **Silence / padding edits** | Inoculates the classic ASVspoof silence-statistics shortcut ([survey/01](../survey/01-sota-speech.md)); ★ `[BC2026]` uses "silence insertion" as a segment-level aug | Trim or pad leading/trailing, p ≈ 0.2 |

---

## Tier B

| # | Method | What it does / why | Parameters |
|---|---|---|---|
| **A-B1** | **Structured frequency/time dropout** (GridMask-style) | ★ `[DFDC 2020, 1st]` dropped structured parts of faces so the model couldn't rely on one artifact region. Our analogue: mask whole frequency bands / time regions | Larger, structured blocks than SpecAugment |
| **A-B2** | **Varied resampler / interpolation choice** | ★ `[DFDC 2020, 1st]` used "diverse interpolation methods during resizing to generalize across video processing variations" — so no single processing path becomes the signal | Rotate among `soxr` HQ/VHQ, `librosa`, `torchaudio` kernels |
| **A-B3** | **Channel operations** — mono↔stereo, L/R imbalance, mono-duplication | Test set has both mono and stereo per sample. ⚠️ Must not correlate with label (see **E-B4**) | p ≈ 0.3 |
| **A-B4** | **Loudness / DRC / clipping** | Real-world processing chains; broadcast-monitoring work shows ignoring these destroys deployment performance ([survey/02](../survey/02-sota-music.md)) | p ≈ 0.3 |
| **A-B5** | **Spectrogram brightness/contrast** | ★ `[BirdCLEF'25 B0]`: `gain ~ U(0.8,1.2)`, `bias ~ U(-0.1,0.1)`, then clamp to [0,1] | p=0.5 |
| **A-B6** | **RandomResizedCrop on the spectrogram** | ★ `[Freesound 2019, 1st]`, scale 0.8–1.0, ratio 1.7–2.3, p=0.33. The author: *"random resize crop helps a lot, but I can't explain why"* | p=0.33 |
| **A-B7** | **Augmentation strength conditioned on source cleanliness** | ☆ `[Bengali.AI 2023, top]`: heavy augmentation for clean read speech, lighter for already-degraded spontaneous audio. Our pools are very heterogeneous — LibriTTS vs ASVspoof21 LA telephony | Per-source `aug_strength` multiplier in the ledger |

---

## Tier C

| # | Method | Note |
|---|---|---|
| **A-C1** | ⚠️ **Pitch shift** | Pitch shift is *itself* a resynthesis operation that imprints phase-vocoder artifacts on REAL audio — manufacturing label noise. It cost MusicDET **+40 EER points** ([survey/02](../survey/02-sota-music.md)), and ★ `[BirdCLEF playbook 2026]` places tempo perturbation in "case-by-case". **Use as inoculation only: p ≈ 0.05, strictly label-independent**, never the sole difference between a real and a fake sample |
| **A-C2** | **Time stretch** | Same family of concern. p ≈ 0.1, small factors |
| **A-C3** | **Packet-loss simulation** | Modeled by ADD-C for VoLTE/VoIP; a refinement of **A-S4** |
| **A-C4** | **Freq-MixStyle** | ★ `[BC2026]` shipped it set to **0.0** — the authors disabled it. Low priority |

---

## Tier X — rejected

| Method | Why not |
|---|---|
| **Music as "background noise"** | Adding music flips `MUSIC_PRESENT` to 1, and if AI-generated, `MUSIC_FAKE` too. Musical backgrounds must go through **step 3 (composition)** where labels are updated — never through step 4 |
| **Superimposing real + fake of the same type** | Overlaying masks the very artifact we want to detect — that is label noise, not augmentation. **Concatenate in time instead** (the PartialSpoof setting, 0.77% EER with multi-resolution labels) |
| **Heavy warping / unrealistic distortion** | ★ `[BirdCLEF playbook 2026]` "use with care" tier: augmentations must not change class semantics "so much that the offline metric becomes noisy" |
| **Cross-file normalization at inference** | Rule 2.4 — 파일 단위 독립 예측 |

---

## Inference-time (not augmentation, but adjacent)

| # | Method | Source |
|---|---|---|
| **I-1** | **Multi-crop, 4–5 crops**, aggregated by median/mean | ★ AT-ADD top systems ([survey/09](../survey/09-challenge-playbooks.md)) |
| **I-2** | **`0.5 · clip + 0.5 · frame_max` blend** — matches our "any segment fake ⇒ component fake" semantics | ★ `[BC2026 Distilled-SED]` ([kaggle/06](../kaggle/06-notebook-code.md)) |
| **I-3** | **Gaussian temporal smoothing** across windows, kernel `[0.1, 0.2, 0.4, 0.2, 0.1]` | ★ `[BC2026]`. ✅ Legal (within-file only). ⚠️ Fights the frame-max semantics — test per head |
| **I-4** | **Confidence-gated aggregation** rather than a plain mean for `FILE_FAKE_PROB` | ★ `[DFDC 2020, 1st]` — worth 0.25 → 0.22 on a single model |
