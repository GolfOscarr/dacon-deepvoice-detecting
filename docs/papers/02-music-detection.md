# 02 — AI-Generated Music Detection

Our highest-leverage axis: metric weight **0.27**, thinnest literature, and the head where the
16 kHz constraint bites hardest. Full listing in [INDEX](INDEX.md).

---

### ⭐ D1 · Assessing AI-generated music detection in real-world broadcast monitoring (2026)
[arXiv](https://arxiv.org/html/2608.07359) · ★ read from primary text

**Field** — AI-music detection under real broadcast degradation. Directly tests our `G1` question.

**Contribution** — Evaluates two CNN detectors (Afchar-style) across three escalating conditions,
comparing clean-trained vs broadcast-trained. Training data: FMA-medium (human) + Suno v3.5 subset
of SONICS (AI).

**Result** — F1 / ROC-AUC:

| Condition | CNN Clean | CNN Broadcast |
|---|---|---|
| Clean foreground music (22.05 kHz, 353 kbps) | 0.992 / 0.998 | 0.992 / 0.996 |
| Synthetic TV broadcast (speech mixed, −30…+30 dB SNR) | 0.342 / 0.909 | 0.661 / 0.926 |
| **Real TV broadcast (8 kHz + 40 kbps AAC-LC)** | **0.186 / 0.707** | **0.472 / 0.775** |

Authors' conclusion: *"current training approaches on CNN-based detectors remain insufficient for
reliable AI-generated music detection in broadcast monitoring."*

**For us** 🔴 — Three things.
1. **Matched-condition training more than doubled worst-case F1** (0.186 → 0.472). That is our
   `A-S1` test-chain normalization, empirically validated.
2. 🔴 **AUC degrades far less than F1** (0.998 → 0.775 vs 0.992 → 0.186). The F1 collapse is
   largely *threshold* collapse. **We are scored on EER and ROC-AUC — pure ranking** — so their
   headline catastrophe overstates our risk. AUC 0.775 at worst-case is poor but far from chance.
3. Our condition is milder than theirs: 16 kHz (Nyquist 8 kHz) vs their 8 kHz sampling
   (Nyquist 4 kHz). We sit between their CFM and RTB rows.

**Limitation** — band-limiting was never ablated separately from AAC-LC compression; only the
cumulative effect is measured. So we cannot attribute the drop to sample rate alone.

---

### ⭐ D2 · ArtifactNet: Detecting AI-Generated Music via Forensic Residual Physics (2026)
[arXiv](https://arxiv.org/html/2604.16254v2) · ★ full text read · ONNX at [HF](https://huggingface.co/intrect/artifactnet) (CC BY-NC 4.0)

> ⚠️ **Correction to my earlier note.** I first flagged this as "the strongest music-head candidate
> found," on the abstract. **The full text says it does not work at our sample rate.** Verdict
> below is now: *not adoptable, but the ideas are.*

**Method** — Three parts, **4.0M params total**, ~16 MB on disk:
1. **ArtifactUNet** (3.6M) predicts a multiplicative mask on the STFT magnitude, `r = m ⊙ X`,
   with the mask **bounded to [0, 0.5]** via `m = 0.5·σ(z)` — an inductive prior that "forensic
   artifacts constitute at most half the total signal energy at any time-frequency bin."
2. **7-channel HPSS forensic features**: `mel_res · mel_H · mel_P · Δ · Δ² · log(H/P) · spectral_flux`
3. **CNN classifier** (0.4M): 3×3 Conv-BN-ReLU-Pool → AdaptiveAvgPool → FC, on 4 s segments,
   song-level **median** verdict at threshold 0.5

**Training** — three phases over 20,374 tracks (12,495 AI + 7,879 real):
`P1` knowledge distillation of **Demucs v4 residuals** (L1 + multi-resolution STFT loss) →
`P2` UNet fine-tuned end-to-end by BCE with the **classifier frozen** →
`P3` **codec-aware** training on 4-way variants (WAV, MP3 128k, AAC 128k, Opus 128k).

**Results**

| Benchmark | ArtifactNet (4.0M) | SpecTTTra (18.7M) | CLAM (194M) |
|---|---|---|---|
| SONICS full test (n=23,288), F1 | **0.9993** (FPR 0.09%) | 0.8874 (FPR 17.97%) | 0.7652 (FPR 67.16%) |
| ArtifactBench test (n=2,263), F1 | **0.9829** | 0.7713 | 0.7576 |
| Sanity protocol, subsets failed /28 | **4** | 23 | 15 |

Out-of-distribution TPR by generator group: AIME (9 gen.) **97.4%** vs SpecTTTra 57.7%;
Latest CDN (4 newest) **94.2%** vs 50.4%. Real-source FPR: SONICS 0.0%, MoM 0.3%,
FMA+YouTube hard negatives 5.4%.

The paper is **honest that SONICS is in-distribution** for it ("the generator families are not
new"), so the ArtifactBench and AIME/CDN rows are the meaningful ones.

**Codec robustness (Table X, at 44.1 kHz)** — MP3 128k **+0.0 pp**, MP3 320k −1.1 pp,
Opus 128k +0.9 pp, Opus 192k +0.0 pp vs WAV. Essentially codec-invariant.

## 🔴 Why we cannot adopt it

| Blocker | Evidence |
|---|---|
| **Wrong sample rate** | Stated limitation: *"Our forensic features operate on 44.1 kHz residuals targeting high-frequency RVQ artifacts. Benchmarks distributed at reduced sample rates (e.g., 16 kHz) attenuate the forensic signal."* The ONNX build requires 44.1 kHz input. **No evaluation at 16 kHz is provided.** |
| **Cannot retrain it** | *"training code and raw weights are not publicly released"* — only a pre-compiled ONNX inference build. So we cannot adapt it to 16 kHz even if we wanted to |
| ⚠️ **Patents pending** | *"Patent applications covering the bounded-mask residual extraction and codec-invariant training methods are pending (KR + PCT)"* — Korean patent applications, in a Korean government competition whose rules assign winning-work copyright to the host. Worth a legal look before reimplementing the bounded-mask method, not just before using the weights |

## ⭐ What to steal anyway

| Idea | Evidence it matters | Where it lands |
|---|---|---|
| 🔴 **Codec-aware training phase** | Phase 2 → Phase 3 cut FMA-hard-negative FPR from **98.7% → 8.0%**, and cross-codec probability delta 0.95 → 0.16 (**−83%**) | Validates `A-S3` codec round-trip as Tier S ([data/06](../data/06-augmentation-spec.md)) |
| 🔴 **Bounded mask** | Ablation: with an unbounded `[0,1]` sigmoid the UNet converges to mean mask ≈ 1.0 (vs 0.18 bounded) and residual energy exceeds 95% of input — i.e. **it degenerates into passing the input through**. A residual extractor without a bound does nothing | Any residual/denoise-channel design, incl. our `P-C1` ([data/10](../data/10-preprocessing-and-filtering.md)) |
| 🔴 **Separation model as a residual *teacher*, not a separator** | Phase 1 distils Demucs v4 **residuals**. Separation is never in the inference path | A **third way** to use separation that neither PC-Mix nor CompSpoof covers — see [04](04-component-partial.md) |
| **7-channel HPSS feature stack** | Cheap, hand-specified, reimplementable in an afternoon | Music-head input representation |
| **Song-level median over 4 s segments** | Their aggregation choice | Compare against our clip+frame-max blend |

## ⚠️ An open question their Table XI raises

Effective bandwidth of the extracted **residual**: AI generators cluster near **291 Hz average**
(Suno v3.5 170, Riffusion 219, Stable Audio 237, Udio 245, MusicGen 255) versus **1,996 Hz for
human music** — a 6.9× separation.

Those are low numbers, seemingly well inside our 0–8 kHz range, which sits oddly against the
stated need for 44.1 kHz and "high-frequency RVQ artifacts." The likely resolution is that
*bandwidth* here measures the residual's spectral **concentration**, not its **location** — a
narrow band sitting high in the spectrum. I could not settle it from the text.

**This is exactly what `E-A1` should measure** ([data/07](../data/07-eda-plan.md)): if
residual-concentration separates AI from human music *below 8 kHz*, the mechanism survives our
band-limit even though this implementation does not.

**Other stated limitations** — latest-generation Udio TPR only 87% (quiet/transition segments have
harmonic-percussive ratios converging with real music); single-pass Demucs laundering drops TPR
99.0% → 94.0%, and multi-pass/adaptive laundering is unevaluated; a future generator without RVQ
quantization "may not exhibit the same residual signature"; **per-channel feature ablations are not
yet published** ("to be completed before the TASLP camera-ready").

---

### ⭐ D3 · Finding the noise: Zero-shot AI Music Detection ("fakeprint") (2026)
[arXiv](https://arxiv.org/html/2607.25530) · ★

**Contribution** — "Fakeprint": checkerboard artifacts left by **deconvolution layers**. STFT →
mean power spectrogram → residual local peaks → normalize. Real music shows scattered random
peaks; synthetic shows consistent structure. Detection is **one-class** (NMF dictionary +
Gaussian-blur reconstruction-error comparison), trained without seeing the generator.

**Result** — Supervised separability (logistic regression on fakeprints), EER: Suno **1.3%**, Udio
**0.2%**, ElevenLabs 2.0%, Mureka 8.7%; Encodec 0.1%, DAC 0%, Musika 0%, GrifMel 1.0%.
Zero-shot @10% FPR: FMA-AE real 90.0% / synthetic 99.4%. Failures: Mubert 5%, Mureka 48.5%.

**For us** ⚠️🔴 — **Do not adopt.** The paper explicitly **excludes SONICS as unsuitable "due to
16 kHz cutoff"** and resamples everything to 44.1 kHz. The method depends on high-frequency
structure our test set does not contain. Note the tension: the HF model
[`lofcz/ai-music-detector`](https://huggingface.co/lofcz/ai-music-detector) claims to implement
fakeprint *at 16 kHz* — resolve that contradiction before trusting either.

The transferable idea is the **one-class / real-only** framing, which sidesteps generator
overfitting — see D6 and the one-class family in [05](05-generalization.md).

---

### D11 · Probing Token Spaces under Generator Shift (2026) — `card`
[arXiv](https://arxiv.org/pdf/2606.08663) · ☆ no extractable table numbers

Probes EnCodec / DAC / XCodec token representations; trains on Suno+Udio, tests on Riffusion,
DiffRhythm, YuE. Conclusion: token-level representations **do not transfer uniformly** across
generators; some codec spaces generalize better than others. Worth a re-read at PDF level if we
pursue a codec-token frontend.

---

## Cross-cutting for the music head

| Finding | Source |
|---|---|
| Published detectors ride **production artifacts** (Suno 192 kbps, Udio 320 kbps, 48 kHz upsampling) that 16 kHz destroys | TISMIR arms race |
| Cross-generator is catastrophic: MERT-AASIST **46.4% EER**; Boomy transfer **3–12%** | MusicDET, TISMIR |
| Codec-aware training cuts cross-codec drift **83%** | D2 |
| Matched-condition training doubles worst-case F1 | D1 |
| **Ranking metrics degrade far less than thresholded F1** | D1 |
