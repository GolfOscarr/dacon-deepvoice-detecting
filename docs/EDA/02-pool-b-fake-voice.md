# 02 — Pool B: Fake Voice

**50.9 GiB, 3 sources.** `wavefake` (26.9, 9 vocoder directories over **two** voices — LJSpeech and JSUT — 117,983 files; see B0b), `mlaad` v9
(5.1, 175 TTS families across 535 `language × generator` directories, capped at 30 files/dir =
16,025 files ≈ 30.1 h), and `ctrsvdd` (18.9, 260 h of **sung** fake voice from 14 SVS/SVC methods).
Target ~70 h, **maximizing generator count** ([data/04](../data/04-sources.md)).

⚠️ **CtrSVDD belongs here, not in pool D.** It is fake *singing*, and
[data/02](../data/02-label-taxonomy.md) is explicit: *"Voice includes spoken and sung (the rules
classify vocals as voice)"*, while *"Instrumental = 반주·악기음 with no vocals"*. Calling it a
fake-music source and routing it to pool D would assert `music_fake = 1` on a vocal recording and
`voice_present = 0` on a person singing.

Pool B carries `VOICE_FAKE` (0.18) and most of the file head's positive mass. Its EDA has one
unusual property: **we hold the real counterpart of one of its two sources**, which makes a paired
design possible that nothing else in the corpus supports.

---

## B0 — CFAD registered: 11 generators, and the duration confound is not only cross-corpus

Fetched, joined, extracted and probed 2026-09-12. CFAD is the best-structured source in the corpus:
real and fake are separated **by directory**, and each side names its origin one level down, so the
grouping key is the publisher's own rather than a path guess.

| | n | hours | groups |
|---|--:|--:|--:|
| `cfad-real` → pool A | 38,600 | 57.5 | **6** Chinese corpora |
| `cfad-fake` → pool B | 73,700 | 69.5 | **11** generators |

All 16 kHz mono wav, so it adds no format diversity — and being natively 16 kHz it is on the
unresampled side of the split described in [00 §4c](00-harness.md#4c---r1s-premise-measured-on-pool-e-it-holds).

🔴 **Duration separates real from fake at AUC 0.725 inside a single publisher's dataset.** Median
4.61 s real against 3.11 s fake; every one of the 11 generators sits between 2.66 s and 3.82 s
while the real corpora span 2.10 s (`magicconversa`) to 9.12 s (`thchs30`).

| real corpus | median | | generator | median |
|---|--:|---|---|--:|
| thchs30 | 9.12 s | | world | 3.82 s |
| selfrecording | 5.20 s | | lpcnet | 3.48 s |
| aishell1 | 4.52 s | | gl | 3.45 s |
| magicread | 4.18 s | | hifigan | 3.22 s |
| aishell3 | 3.21 s | | wavenet | 3.03 s |
| magicconversa | 2.10 s | | fasthifigan | 2.66 s |

This matters because it is the first evidence that the duration confound is **not** merely an
artifact of mixing publishers. FakeMusicCaps at 10.0 s against `fma_small` at 30.0 s could be
dismissed as two archives with different conventions; here one publisher's own real and fake halves
differ, so the crop policy that [03 C5](03-pool-c-real-instrumental.md) and
[04 D2](04-pool-d-fake-instrumental.md) call for has to hold **within** a source as well as across
sources. ⚠️ Caveat: the halves are not the same utterances (38,600 real against 73,700 fake), so
part of the gap is utterance selection rather than generation. Separating those needs the
utterance-level join, which CFAD's protocols support and which is not done yet.

**Three of CFAD's four shipped versions are not registered.** `codec_version` (4 codecs) and
`noisy_version` (5 SNRs) are the same 115,800 utterances transformed by the publisher — under R2 a
codec and additive noise are render-time transforms applied to train and test alike, not more
corpus, and as rows they would be duplicates carrying a fixed fingerprint. `partiallyfake` (3,500
files) is excluded from the fake half because it is *partially* spoofed: pool B asserts the voice is
generated throughout, and those files contain bona fide segments. They belong to the frame head.

---


## B0b — 🔴 WaveFake fetched: 117,983 files, **two** grouping atoms, and a duplicated half

Fetched, extracted and probed 2026-09-13. 26.9 GB, 3 objects, sha256 verified.

**The census reads 117,983 — WaveFake's published count — and a naive walk finds 134,266.**
The difference is an archive defect the E1 duplicate sweep named on the first run: the
`common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech/` directory holds 16,283 files and
repeats **all 16,283 byte-identical** under a nested `generated/` inside itself. Measured three
independent ways — 16,283 top-level `.wav`, 16,283 nested, 16,283 names in common, and 16,283
sha256 pairs in the sweep. Counting both weights those prompts double in pool B and puts an
identical pair on either side of any fold.

It is excluded in `configs/eda.yaml` with its reason, as a **deduplication of a packaging defect**
— the same shape as FakeMusicCaps' `__MACOSX` sidecars, and deliberately not a licence decision
(R2: every exclusion names which kind it is).

| directory | files | voice |
|---|--:|---|
| `ljspeech_*` × 7 — melgan, melgan_large, multi_band_melgan, full_band_melgan, parallel_wavegan, hifiGAN, waveglow | 13,100 each | LJSpeech |
| `common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech` | 16,283 | LJSpeech |
| `jsut_*` × 2 — multi_band_melgan, parallel_wavegan | 5,000 each | JSUT |

🔴 **Nine of the ten directories are one speaker, so WaveFake has two grouping atoms, not ten.**
The vocoder is a stratification axis; the voice is the grouping one. `eda.groupkeys` gives the nine
LJSpeech directories **`ljspeech`'s own key**, so 137,366 files across two sources and two pools
share one group — 13,100 real recordings in pool A and 124,266 vocoded ones in pool B.

That sharing is B1's precondition rather than a nicety. The whole value of this pool is that we
hold the real counterpart; a fold table that puts `LJ001-0001` in train and
`ljspeech_melgan/LJ001-0001_gen` in validation measures a model recognising an utterance it has
already heard, and reports it as vocoder detection. Grouping by vocoder directory would do exactly
that, seven times over.

⚠️ And the honest consequence: **2 atoms from 117,983 files.** WaveFake cannot supply fold
rotation however large it is, and `G-EDA3` now says so in the clause reserved for sources whose key
was read and is genuinely short — beside `ljspeech` and `sonics`, and apart from the sources nobody
has looked at yet.

---

## 🔴 B0c — S tier: **41% of pool B is shorter than the test set's minimum**

Measured 2026-09-14 over 6,000 decoded files (`eda report`).

| p05 | median | p95 | in 4–60 s | **under 4 s** | has a 4 s span |
|--:|--:|--:|--:|--:|--:|
| 2.03 | 4.725 | 13.76 | 0.586 | **0.414** | 0.103 |

The test set is 4–60 s. Two fifths of this pool cannot fill its minimum window at all, and only
10.3% contain a non-silent span of 4 s — the lowest of any pool, and a direct constraint on the
sampler rather than a curiosity. Pool A's equivalents are 0.174 and 0.182
([01 A9](01-pool-a-real-voice.md)).

This is the pool's own version of the duration problem: where pools C and D differ from each other
([06 X1d](06-cross-pool.md#-x1d--x1-on-the-decoded-audio-the-duration-shortcut-survives-the-chain)),
pool B differs from the **test set**. The voice heads themselves stay clean — `voice_fake` collapses
to 0.401 under an archive holdout — so what this costs is usable material, not label integrity.

---

## B1 — 🔴 WaveFake ↔ LJSpeech pair reconstruction — Tier S, and the highest-value item in this pool

**Compute.** WaveFake is LJSpeech re-synthesized by 7 vocoders (MelGAN, Parallel WaveGAN,
Multi-band MelGAN, Full-band MelGAN, HiFi-GAN, WaveGlow, and the JSUT set). Its filenames carry the
**LJSpeech utterance id**. Join on that id against our own `ljspeech` rows and populate `pair_id`
— one id per utterance, shared by the real file and all 7 of its fakes. Then, per vocoder, compute
the **mean log-mel difference** `E[mel(fake) − mel(real)]` over pairs, at both planes
([00 §2](00-harness.md)).

**Why it is meaningful.** Every other real/fake comparison in this corpus is confounded: different
speakers, different text, different recording chain, different archive. **This one is not.** Same
speaker, same utterance, same source recording — the *only* thing that differs is the vocoder. That
is precisely the paired-negative construction SETI's organizers were forced into (on-target vs
off-target through an identical channel) and that [data/05](../data/05-synthesis-plan.md) calls the
T3 twin design, ☆ `[SETI 2021]`. It gives an unconfounded estimate of the artifact.

**What we discover.** Three things, none of which requires training a model:

1. **What the vocoder artifact actually looks like**, per family, as a mel-difference image. This is
   E-A8's matched-pair gallery — *"the most legible figure available for the 2nd-stage report
   (결과 해석, 15 pts)"* — and it comes free with the join.
2. 🔴 **Whether it survives 16 kHz.** Compute the difference at the `native` plane and again at the
   `chain` plane. Vocoder artifacts are classically concentrated in the upper spectrum; LJSpeech is
   22.05 kHz, so everything above 8 kHz is discarded by the competition's own standardization. The
   fraction of the artifact's energy that survives the cut is a direct, cheap measurement of the
   voice head's difficulty — and the exact analogue of E-A1 for the voice side.
3. **Which vocoder families are near-duplicates of each other.** The correlation matrix between the
   7 difference images says whether MelGAN and Multi-band MelGAN are one artifact family or two —
   which is the `artifact_family` assignment, and the fold-disjointness axis depends on getting it
   right. ⚠️ [validation/01](../validation/01-split-scheme.md) splits on **artifact family, not
   model name**, precisely because model names over-count.

**⚠️ The trap this also exposes.** If WaveFake's 7 vocoders are the LJSpeech utterances and
LJSpeech is also in pool A, then **the same underlying recording sits on both sides of the label**.
That is fine — desirable, even — but it *must* be linked by `dup_group`/`pair_id` so the fold
builder keeps the pair together, or the model sees utterance 07-0123 as REAL in TRAIN and as FAKE in
VAL and the split leaks in the subtlest possible way. This is the RIRS/MUSAN lesson
([data/12](../data/12-acquisition-status.md)) in a form that hashing **cannot** catch, because the
files are not byte-identical — they are the same content through different synthesis.

**Output.** `pair_id` populated in the manifest; `wavefake_artifact_gallery/`;
`vocoder_family_correlation.csv`; the surviving-energy fraction per vocoder at 16 kHz.
**Gate.** **G-EDA4** — no `pair_id` group may straddle a fold boundary.

---

## B1b — ✅ Answered 2026-09-16: the artifact does **not** die at 16 kHz

**Population.** 500 utterances drawn at seed 0 from the **13,100 that resolve to all eight files**
(the real one plus all seven vocoders) — every LJSpeech utterance is completely paired, so the draw
was unconstrained. 4,000 files decoded on both planes, **0 decode failures**, 7.41 audio-hours in
10.7 min. Artifacts under `eda/out/_shared/b1/`; re-read with `python -m eda.cli b1`.

⚠️ This is a **separate selection**, not the S-tier draw. Only 128 of 13,100 utterances are
incidentally pairable inside `sample.json`, which is far too thin for a per-vocoder mean, and the
recorded draw is part of the corpus definition ([00 §3](00-harness.md)) — so B1 wrote its own
artifact and left it alone.

🔴 **The trap this experiment nearly fell into.** WaveFake ships **three** filename conventions and
announces none of them: `_gen.wav` in five directories, `_generated.wav` in `ljspeech_hifiGAN`, and
a bare `.wav` in `ljspeech_waveglow`. The natural join resolves five of seven and silently drops
HiFi-GAN and WaveGlow — a five-column table where the plan asked for seven, failing as a *thin
result* rather than an error. `eda.analyze.pairs` therefore refuses an utterance that does not
resolve to all eight files, and two mutants hold the two odd suffixes in place.

### B1b-i — What survives the chain: **82–115%**

Energy of the `[128]` difference image, `chain` against `native`. The mel bank is pinned to an
absolute 0–8000 Hz on both planes ([`eda/extract/vectors.py`](../../eda/extract/vectors.py)), so the
two images are band-for-band comparable and the ratio subtracts the same thing.

| vocoder | energy native | energy chain | **surviving** | peak band (native) |
|---|--:|--:|--:|--:|
| `parallel_wavegan` | 364.97 | 301.36 | **0.826** | 1 (28 Hz) |
| `waveglow` | 274.37 | 247.95 | **0.904** | 1 |
| `multi_band_melgan` | 198.02 | 183.20 | **0.925** | **114 (5.9 kHz)** |
| `hifiGAN` | 163.18 | 134.11 | **0.822** | 1 |
| `melgan` | 54.74 | 57.22 | **1.045** | 1 |
| `melgan_large` | 41.85 | 48.18 | **1.151** | 1 |
| `full_band_melgan` | 26.72 | 22.35 | **0.837** | 1 |

🔴 **This overturns the expectation B1 was written to test.** [B1](#b1---wavefake--ljspeech-pair-reconstruction--tier-s-and-the-highest-value-item-in-this-pool)
predicted that *"vocoder artifacts are classically concentrated in the upper spectrum"* and that the
16 kHz cut would therefore be the voice head's central difficulty. It is not. **The artifact lives
below 8 kHz and arrives at the model essentially intact.** Per-band, the dominant term for five of
seven vocoders is **sub-speech rumble below 72 Hz** that the vocoder invents and LJSpeech does not
have — `+5.58 dB` for Parallel WaveGAN, `+5.50 dB` for WaveGlow, `+2.37 dB` for MelGAN. Nothing
about a 16 kHz resample touches 28 Hz.

⚠️ **Two ratios exceed 1.0 and that is not a bug.** For `melgan` and `melgan_large` the chain
*deepens* the artifact. The mechanism is visible in the per-band table and corroborated by B1b-ii:
both vocoders are already short of energy above 8 kHz, so the resampler's near-Nyquist rolloff
tilts their top bands further negative (`melgan` bands 120–127 go `−0.115 → −0.411 dB`), and energy
is a sum of squares. The chain does not remove this artifact; it slightly sharpens it.

### B1b-ii — Above the cut, and why the sign matters

`hf_ratio_8k` on the **native** plane, paired `fake − real`, 500 pairs each. Real mean `0.0289`.

| vocoder | fake mean | **Δ mean** | Δ std |
|---|--:|--:|--:|
| `parallel_wavegan` | 0.0165 | **−0.0124** | 0.0188 |
| `melgan_large` | 0.0187 | **−0.0102** | 0.0131 |
| `full_band_melgan` | 0.0208 | **−0.0081** | 0.0143 |
| `melgan` | 0.0231 | **−0.0058** | 0.0125 |
| `hifiGAN` | 0.0263 | **−0.0026** | 0.0171 |
| `multi_band_melgan` | 0.0272 | **−0.0017** | 0.0149 |
| `waveglow` | 0.0425 | **+0.0136** | 0.0210 |

There **is** signal above the cut, and the chain throws all of it away. But ⚠️ **the sign is not
consistent**: six vocoders under-produce high frequency and WaveGlow over-produces it by more than
any of them under-produce. A "fake audio is missing its top octave" heuristic — which is the
folk rule this corpus was most likely to absorb — would be **actively wrong on WaveGlow**, and
every Δ here is smaller than its own standard deviation. This is a population effect, not a
per-file test.

### B1b-iii — Artifact families: the names are wrong

Correlation between difference images, native plane (chain is in
`b1_family_correlation_chain.parquet` and tells the same story).

| | fbm | hifi | mel | mel-L | mbm | pwg | wg |
|---|--:|--:|--:|--:|--:|--:|--:|
| `full_band_melgan` | 1.000 | 0.705 | 0.418 | 0.375 | 0.659 | 0.633 | 0.714 |
| `hifiGAN` | 0.705 | 1.000 | 0.268 | 0.291 | 0.508 | 0.641 | 0.688 |
| `melgan` | 0.418 | 0.268 | 1.000 | **0.895** | **−0.186** | 0.411 | 0.571 |
| `melgan_large` | 0.375 | 0.291 | **0.895** | 1.000 | **−0.086** | 0.345 | 0.514 |
| `multi_band_melgan` | 0.659 | 0.508 | **−0.186** | **−0.086** | 1.000 | 0.279 | 0.393 |
| `parallel_wavegan` | 0.633 | 0.641 | 0.411 | 0.345 | 0.279 | 1.000 | 0.742 |
| `waveglow` | 0.714 | 0.688 | 0.571 | 0.514 | 0.393 | 0.742 | 1.000 |

🔴 **This is the direct evidence [validation/01](../validation/01-split-scheme.md) was asserting
without any.** That page splits folds on **artifact family, not model name**, on the argument that
model names over-count. The measurement says the names are worse than over-counting — they are
*anti-correlated with the truth* in the one place it matters most:

* `melgan` ↔ `melgan_large` correlate at **0.895** — one family, as the names suggest.
* `melgan` ↔ `multi_band_melgan` correlate at **−0.186**. They share a name and have **opposite**
  artifacts: MelGAN's signature is low-frequency rumble (band 1), Multi-band MelGAN's is a
  mid-high bump at **band 114, 5.9 kHz** — the only vocoder in the set whose peak is not at the
  bottom, and plausibly its sub-band crossovers. A fold builder that grouped "the MelGANs" by name
  would put two opposite artifacts in one family and split the one real family it had.

**Working family assignment**, at a 0.70 cut: `{melgan, melgan_large}` · `{multi_band_melgan}` ·
`{full_band_melgan, hifiGAN, waveglow, parallel_wavegan}` — noting that the third is a weak
grouping (0.63–0.74) and is the one to revisit when the fold table exists.

### B1b-iv — What this changes

1. **The voice head is more tractable than assumed.** Its artifact survives the competition's own
   standardization at 82–115%. The pessimism in
   [X1d](06-cross-pool.md) about voice-head collapse (`voice_fake` 0.401 under an archive holdout)
   is about *confounding*, not about signal availability — this measurement removes every confound
   and the signal is there.
2. 🔴 **A high-pass is not free.** Five of seven vocoder signatures are concentrated below 72 Hz.
   Any render-time DC-removal or rumble filter would delete the largest single discriminative
   feature this corpus has for pool B. [`eda/planes.py`](../../eda/planes.py) currently applies
   neither, and this is now a reason to keep it that way — the question it says is
   *"answered with the corpus rather than assumed"* is answered here, for the voice side: **do not
   high-pass.**
3. **`artifact_family` is measured, not named.** See B1b-iii.

**Still open.** `pair_id` is populated for the 500 drawn utterances in `b1_pairs.parquet`, not for
all 13,100 in the manifest. **G-EDA4** stays `na` until a fold table exists (Phase 2), and whether
the full population needs `pair_id` is a fold-builder decision, not an EDA one. ⚠️ The trap B1
flagged is unchanged and still live: LJSpeech sits in pool A and its re-synthesis sits in pool B, so
the same underlying recording is on both sides of the label and **hashing cannot catch it**.

---

## B2 — MLAAD generator and language inventory 🔴 Tier S

**Compute.** Walk the `fake/<language>/<generator>/` tree from `_meta/selection.json` and the S3
key listing (structure was deliberately preserved — flattening it was one of the five acquisition
defects). Per leaf: file count, total hours, native sample rate, container. Aggregate to
`language × generator`, then to `generator`, then propose an `artifact_family` mapping that groups
architecturally identical generators (all XTTS variants, all VITS variants, all Bark variants).

**Why it is meaningful.** `generator` is the DOSS domain key and the fold-disjointness axis at the
same time. The cap is per leaf directory *because* the leaf is the domain — a global cap would have
kept all of the first few generators and none of the rest, the opposite of what generator diversity
needs. But the **family** grouping above it is a judgement call that nothing has yet made, and
`build_folds` consumes families, not directories.

**What we discover.**
- The true count of independent **artifact families** in pool B, against the ≥6 needed for a 5-fold.
  535 directories may collapse to far fewer families; 175 model names certainly do.
- 🔴 **Whether Korean fake voice exists in our pool at all.** MLAAD spans 54 languages. The
  competition is Korean-hosted and "Korean slice size" is an open decision in
  [PROGRESS](../../PROGRESS.md) with no evidence attached. This census is the evidence. If Korean
  fake voice is absent or thin, self-generated Korean TTS moves up the priority order — and we hold
  Zeroth-Korean transcripts to drive it ([data/05](../data/05-synthesis-plan.md)).

**Output.** `mlaad_inventory.csv` (leaf → hours, sr, container), `artifact_family_map.yaml`.
**Gate.** **G-EDA3** on the family count.

---

## B3 — Format census against pool A 🔴 Tier S

**Compute.** The A1 table, for pool B, and the joint `P(FAKE | container, native_sr, codec, bitrate)`.

**Why it is meaningful.** This is the other half of A1 and it is where the corpus-identity confound
either appears or does not. 🔷 The expectation: MLAAD's fakes carry whatever each TTS model emitted
(commonly 16, 22.05 or 24 kHz wav), WaveFake inherits LJSpeech's 22.05 kHz wav, and Common Voice —
88 of pool A's 111 GiB — is 32 kHz mp3. If that holds, **"was ever an mp3" is close to "is REAL"**,
with 0.45 of the metric riding on it.

**What we discover.** The single number that decides whether the codec stage is optional or
mandatory: the metadata-only AUC on the file head.

**What it changes.** 🔴 If the confound is real, the **S3 `codec_aware` stage stops being an
augmentation and becomes a correctness requirement** — every pool must be pushed through the same
distribution of codec round-trips so that codec history carries no label information. That stage
already exists and already works on real audio (the `_PAD_TOL = 576` fix), but it is currently
justified as robustness rather than as de-confounding. ⚠️ It is also the **decode-bound** stage
(14.4 samp/s, render 75%), so promoting it has a measured throughput cost.

---

## B4 — Generation-failure and degenerate-output screen 🔴 Tier S (F-S4)

**Compute.** Over every MLAAD file (16 k files — cheap enough for 100%) and the WaveFake sample:
`rms_dbfs` (silence), duration vs the directory median (truncation), and a **looping/babble
detector** — the maximum of the mel-envelope autocorrelation at lags between 0.5 s and half the
file length, normalized. Flag `> 0.9` as probable loop. Plus `statistics_T` below the 0.8 quantile
of its own generator.

**Why it is meaningful.** F-S4 is Tier S in [data/10](../data/10-preprocessing-and-filtering.md)
and is written as if it applies only to self-generated audio. It does not: MLAAD is 175 TTS
families run at scale, and TTS at scale fails in exactly these ways — silence, looping, babble,
truncation, wrong language. Those files carry `label_voice_fake = 1` with **no voice to judge**.

**What we discover.** The per-generator failure rate, which is also a quality ranking of the
generators — and 🔴 an *asymmetric* one: a family with a 20% failure rate contributes 20% of its
rows as label-noise concentrated in one fold group.

**What it changes.** Drops, with the reason and the count recorded per family (R2: a drop is a
distribution-shift decision). And the `label_confidence` tier — public-corpus fakes are *asserted*,
not *exact*, and these screens are what justify the tier boundary that F1/F2's per-tier loss and
two-stage clean→noisy training consume.

⚠️ **Do not simply drop silent fakes without checking pool A's silent rate.** If fakes are screened
for silence and reals are not, silence becomes a REAL cue — the asymmetric-filter trap in R2,
inverted. Screen both pools with the identical threshold.

---

## B5 — Duration and speaker-proxy census ⚠️ Tier A

**Compute.** Duration histogram per generator; MLAAD v9 is ~6.75 s/file on average over the full
release, so check whether the capped 30-per-directory selection preserved that. Count files below
the 4 s sampler floor. For WaveFake, the speaker is LJSpeech's single speaker — record it as such.

**Why it is meaningful.** Two distinct problems. First, the same 4 s floor as
[A5](01-pool-a-real-voice.md): rows below it are invisible to the sampler while still counting
toward the hours budget, which is how `rirs_isotropic` contributed nothing while appearing in the
file tally. Second, 🔴 **WaveFake is one speaker and one text corpus** — 117 k files that are
**one** grouping atom for speaker-disjointness and, in effect, one content distribution. Its file
count radically overstates its diversity contribution, exactly as LJSpeech's does on the real side.

**What we discover.** The effective, floor-corrected, group-corrected size of pool B — which is
almost certainly much smaller than 32 GiB suggests and is what the DOSS cap should be computed
against.

---

## B7 — 🔴 CtrSVDD: the sung/16 kHz confound, and the ND duty — Tier S

**Compute.** The [A1](01-pool-a-real-voice.md)/[B3](#b3--format-census-against-pool-a--tier-s)
format census restricted to pool B, split by `sung` vs `spoken`. Then `P(sung | native_sr)` and
`P(native_sr | source)`.

**Why it is meaningful.** CtrSVDD is **already 16 kHz**, and it is the only sung source in pool B.
MLAAD and WaveFake are 22.05/24 kHz and get downsampled, so they carry a resampler transition band
that CtrSVDD does not ([A2](01-pool-a-real-voice.md)). The result: within pool B, **"is sung" and
"is natively 16 kHz" are perfectly confounded**, and either one is recoverable from the near-Nyquist
shelf alone. A model asked to detect sung fakes could instead be detecting the absence of a
resampler.

**What we discover.** Whether the two can be separated at all inside our corpus, and how much of
pool B's sung slice would survive a resampler-randomising transform. 🔷 The likely answer is that
the confound can only be broken by resampling *everything* through the same randomised chain, which
is [A2](01-pool-a-real-voice.md)'s recommendation arriving from a second direction.

**⚠️ And the compliance half.** CtrSVDD is **CC BY-NC-ND**. Usable only in the form
[#417333 A5](../competition/05-talkboard-qa.md) grants: train from the **originals**, augment at
training time, ship 원본 파일 + 코드 + seed, never a redistributed derivative.
[`scripts/sources.yaml`](../../scripts/sources.yaml) carries that as `caveat_compliance`; the
acquisition record's own `verified_note` and `caveat` are **empty**, so the provenance file alone
does not carry the condition. Its `test_set.zip` also comes from Zenodo record `10742049` while the
licence was read on `10467648` — two records, one verdict.

---

## B6 — Cross-generator separability screen ⚠️ Tier B (cheap form of E-A9)

**Compute.** Per generator family, the mean `ltas[128]` and `band_energy[8]` at the `chain` plane.
Then a family × family distance matrix (symmetric KL over the normalized LTAS), and a
leave-one-family-out linear probe on those summary statistics alone.

**Why it is meaningful.** E-A9 as specified trains a small model per family and is a Phase-3 item.
The **spectral-summary** version is a same-day screen that answers most of the same question: which
families are trivially separable from the rest (low marginal value — they will be learned from two
examples) and which sit on top of each other (spend the budget there). ★ `[survey/02]` reports
**46.4% EER cross-generator** for music, i.e. near-chance, so the expectation is that this matrix is
far less structured than intuition suggests.

⚠️ **A linear probe on summary statistics is a screen, not E-A9.** A family that is inseparable in
LTAS may be trivially separable to an SSL frontend. Use it to *rank* where to spend, never to
conclude that a family is redundant.

**Output.** `generator_difficulty_screen.csv`, ranking families by distance to their nearest
neighbour — which is also a principled way to pick which families to hold out for the sealed PROBE
slice.
