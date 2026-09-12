# 02 — Pool B: Fake Voice

**50.9 GiB, 3 sources.** `wavefake` (26.9, 7 vocoders over LJSpeech + JSUT), `mlaad` v9
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
