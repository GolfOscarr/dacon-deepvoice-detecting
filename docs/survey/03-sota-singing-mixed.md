# 03 — Singing Voice & Mixed Audio (the 혼합 case)

Covers: sung vocals as "voice", vocals-over-accompaniment mixtures, and **the separation question**.

## 🔴 Should we separate before detecting? — Direct evidence: NO (not naively)

★ [*Detection of AI-generated stems within hybrid human-AI music*](https://arxiv.org/html/2607.26874v1) (2026).
The closest published analogue to our decoupled `VOICE_FAKE` / `MUSIC_FAKE` labels.

**Setup**: MUSDB18-HQ (150 tracks, isolated vocals + accompaniment) → EnCodec autoencoding
(24 kbps, 48 kHz) per stem → **four mixture types** (gen-vocals+real-accomp, real-vocals+
gen-accomp, both-gen, both-real) → **5 gain levels (−12…+12 dB)** → 3,000 mixtures.

**Naive separate-then-detect fails:**

| Case | FPR |
|---|---|
| Generated-vocal detection @ 0 dB | **38%** |
| Accompaniment, hybrid vs fully-generated | **94.7%** |

> ★ *"Artifacts associated with an AI-generated stem are not reliably recovered by generic source
> separation systems"* — they **spread across all separated sources.**

**What worked instead** — separation used for *features*, not as preprocessing:

```
full mixture ──→ 2s chunk-level AI detector ──┐
                                              ├──→ per-stem MLP (5 hidden layers, 32 units)
ht-demucs ──→ per-band SNR estimates ─────────┘    (16 × 1 kHz bands, or 5–16 kHz focus band)
track-level = average of chunk predictions
```

| Target | Oracle SNR | Separation-based SNR |
|---|---|---|
| Generated **accompaniment** | ~98% TPR / ~1% FPR | ~97% TPR / ~5% FPR |
| Generated **vocals** | ~80% TPR / ~26% FPR | **~65% TPR / ~40% FPR** |

**Two transferable findings:**
1. **Detection tracks stem energy.** Quiet background music under speech, or a voice buried in a
   mix, are the hard cases → **mixing gain must be a controlled variable in our training data.**
2. **Vocals are harder than accompaniment** (sparse, lower energy in the detection band).
   Combined with the metric weights, this compounds: the music head is both higher-weighted
   *and* more tractable.

**Authors' stated limits**: two-stem only (multi-stem is combinatorial); MSS-based SNR adds
compute and degradation; detectors remain vulnerable to pitch shift, time stretch, and
sample-rate conversion.

## SingFake — separation is ambivalent, not positive

★ [SingFake](https://arxiv.org/pdf/2309.07525) (2023). Deepfake songs from user-generated content
sites. Pipeline: **Demucs** vocal separation → VAD → segment both mixtures and vocals.

- Speech countermeasures improve substantially when trained on SingFake — **either** on separated
  vocals **or** on song mixtures.
- ★ Critical caveat, authors' framing: separation *"can create artifacts that obscure the
  differences between bonafide and deepfake vocals."*

That caveat is exactly why SVDD split into two tracks.

## SVDD 2024 — the inaugural challenge

★ Two tracks: **CtrSVDD** (controlled, clean unaccompanied vocals) and **WildSVDD** (SingFake-
derived, in-the-wild with instrumental interference).
[challenge paper](https://arxiv.org/abs/2408.16132) · [eval plan](https://challenge.singfake.org/SVDD_Challenge_2024_Eval_Plan_v0.3.pdf)

**CtrSVDD**: 47 teams submitted, 37 beat baselines, **top team 1.65% EER**.

☆ The controlled→wild gap was substantial; laboratory performance did not survive background
music, compression and noise. *(My secondary source for WildSVDD specifics was unreliable —
re-read the challenge paper before quoting numbers. See [10](10-open-questions.md).)*

### CtrSVDD dataset (★ [paper](https://arxiv.org/html/2406.02438))

| | |
|---|---|
| Total | 220,798 clips / 307.98 h @ **16 kHz** / 164 singers / avg 5.02 s |
| Bonafide | 32,312 clips (47.64 h) |
| Deepfake | 188,486 clips (260.34 h) |
| Train | 12,169 bona + 72,235 fake — 59 singers, 8 methods |
| Dev | 6,547 bona + 37,078 fake — 55 singers, 8 methods |
| Eval | 13,596 bona + 79,173 fake — 48 singers, **6 unseen methods** |
| License | **CC BY-NC-ND 4.0** ⚠️ see below |
| Download | [train/dev](https://zenodo.org/records/10467648) · [eval](https://zenodo.org/records/10742049) · [baseline](https://github.com/SVDDChallenge/CtrSVDD2024_Baseline) |

**Real sources**: Opencpop, M4Singer, KiSing, ACE-Studio (Mandarin); Ofuton-P, Oniku Kurumi,
Kiritan, JVS-MuSiC (Japanese).

**Generators** — 7 SVS: XiaoiceSing, VISinger, VISinger2, NNSVS, Naive RNN, DiffSinger, ACESinger.
7 SVC: NU SVC (diffusion + ContentVec) and 6 Soft-VITS-SVC variants (WavLM, ContentVec, MR-HuBERT,
WavLabLM, Chinese HuBERT, source-filter HiFi-GAN).

**Baselines** (graph-attention backend, 5 frontends):

| Frontend | EER |
|---|---|
| Raw waveform | **13.75%** |
| LFCC | 16.15% |
| Mel-spectrogram | 25.19% |
| Spectrogram | 25.50% |
| MFCC | 26.67% |

Raw waveform and LFCC clearly best. Degradation concentrated on unseen methods A12 (DiffSinger)
and A14 (ACESinger).

⚠️ **CC BY-NC-ND**: the **ND (NoDerivatives)** term is a genuine problem for us — augmented
copies and possibly trained models could be read as derivatives, and DACON requires shipping the
training files. **Legal review needed before use.** Same concern for anything ND-licensed.

## Practical implications for our design

| Question | Current best answer |
|---|---|
| Separate before detecting? | **No.** Detect on the mixture; optionally use separation to derive per-band SNR features. |
| Is ht-demucs even applicable? | Trained at 44.1 kHz; our audio is 16 kHz mono/stereo incl. telephone → likely degraded. Preinstalled (`demucs==4.0.1`) so it costs nothing to test, but budget runtime. See [10](10-open-questions.md). |
| Vocals vs accompaniment difficulty | Vocals harder. Expect `VOICE_FAKE` to lag `MUSIC_FAKE`. |
| Training data structure | Must include the **4-way voice×music real/fake grid at varied mixing gains**, or the two heads collapse into one entangled axis. |
| Sung vs spoken voice | Competition labels **vocals as voice**; a song = mixed. Sung-voice deepfakes need CtrSVDD-style coverage, not just TTS. |
