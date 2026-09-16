# 01 — Pool A: Real Voice

**111.0 GiB, 5 sources.** `common-voice-en` (88.1), `libritts-r` (10.0), `zeroth-korean` (9.6),
`ljspeech` (3.0), `common-voice-ko` (0.2). Target ~70 h ([data/04](../data/04-sources.md)); we hold
far more than that, so pool A's EDA is mostly a **selection** problem, not a coverage problem.

Pool A is the entire real-voice class. It supplies cell 1 whole-file rows and the voice half of
cells 5 and 6. Whatever is systematically true of these five archives and *not* true of pool B
becomes "REAL" to the voice head — and to the file head, which carries 0.45.

---

## A1 — Native-format census 🔴 Tier S

**Compute.** `ffprobe` over 100% of pool A: container, codec, native sample rate, channels,
bitrate, encoder/LAME tag, and for mp3 the presence of a Xing/Info gapless header. Cross-tabulate
against `source_name`. Then the same table for pool B ([02](02-pool-b-fake-voice.md)) and compute
`P(REAL | container, sr, codec)`.

**Why it is meaningful.** These five corpora were produced by four different recording and
distribution chains, and — 🔷 based on each publisher's documented format, to be confirmed by this
very census — they do not overlap: Common Voice ships crowd-sourced **mp3**, LibriTTS-R ships
restored **24 kHz wav**, Zeroth-Korean ships **16 kHz flac**, LJSpeech ships **22.05 kHz wav**. If
pool B's fakes arrive predominantly at 22.05/24 kHz wav, then *container × native rate* is close to
a label, and a metadata-only classifier will find it before any acoustic model does.

**What we discover.** Whether the REAL/FAKE axis is confounded with the archive axis, and by how
much. This is the input to the E-S2 shortcut audit ([06](06-cross-pool.md)), and it is the one
measurement that can be taken today, without decoding anything, on the full population.

**Output.** `format_census.csv`; the metadata-only AUC per head.
**Gate.** Feeds **G-EDA2** (metadata AUC < 0.60). Above that, the fix is not to delete the column —
it is to *equalize the chain*: push every pool through the same codec round-trip distribution, which
`training/render.py`'s `codec_aware` stage already implements and which S3 already exercises.

---

## 🔴 A9 — S tier: duration, and a 22.6 dB level spread across the corpus

Measured 2026-09-14 over 6,426 decoded files (`eda report`).

| p05 | median | p95 | in 4–60 s | under 4 s | has a 4 s span |
|--:|--:|--:|--:|--:|--:|
| 2.47 | 6.96 | 411.27 | 0.761 | 0.174 | 0.182 |

17% is shorter than the test set's floor and the p95 is nearly seven minutes — `musan-speech` ships
whole LibriVox recordings while `cfad-real` ships utterances. Only 18% hold a non-silent 4 s span.

🔴 **Level is a second confound, and no metadata audit could have seen it.** Across the whole
corpus, median RMS on the chain plane spans **22.6 dB** — `fma` at −15.0 dBFS, `rirs-isotropic-noise`
at −37.6 — and the pattern is per-publisher rather than per-pool:

| source | median peak | median RMS | files with clipping |
|---|--:|--:|--:|
| `musan-speech` | **−0.000265 dBFS** | −21.3 | **87.1%** |
| `musan-music` | −0.000265 | −16.2 | 88.5% |
| `musan-noise` | −0.000265 | −19.8 | 86.0% |
| `ljspeech` | −5.32 | −23.95 | 0.0% |
| `zeroth-korean` | −13.46 | −31.46 | 0.8% |
| `cfad-real` | −11.76 | −30.21 | 5.8% |

All three MUSAN partitions sit at the same peak to six decimals: **peak-normalised to full scale**,
which is a fingerprint of how MUSAN was published and has nothing to do with what the audio is. A
model given un-normalised level can read the publisher off it.

⚠️ This is an argument for normalising, and for choosing the stage deliberately — the numbers above
are *after* the 16 kHz chain, which is the only level a normalisation decision can act on.

---

## A2 — Near-Nyquist rolloff profile 🔴 Tier S

**Compute.** At the `chain` plane (16 kHz), per source: mean energy in 7.0–7.5, 7.5–7.8 and
7.8–8.0 kHz bands relative to the 1–4 kHz band; `near_nyquist_ratio`; and the LTAS overlay from
6 kHz to Nyquist. Also at `native`, for the same files.

**Why it is meaningful.** After resampling to 16 kHz, a file that was **natively 16 kHz** carries
real energy all the way to 8 kHz. A file downsampled from 22.05, 24 or 32 kHz carries the
resampler's **transition band** instead — a characteristic shelf or notch in the last few hundred
Hz whose shape identifies the resampler and its parameters. Zeroth-Korean is the only natively
16 kHz source in pool A; everything else is downsampled. So this measurement separates
`zeroth-korean` from the rest **on a property that has nothing to do with speech**, and if Korean
coverage is uneven across REAL and FAKE, it separates the label too.

**What we discover.** (a) Whether our own resampling has written a source fingerprint into the
corpus, which is a processing defect we introduce, not one we inherit; (b) the magnitude, which
decides whether B6 — ★ `[DFDC 2020, 1st]` *"vary the interpolation/resampling method itself, so no
single processing path becomes the signal"* — must be promoted from an augmentation idea to a
required step in the render chain.

**What it changes.** If the shelf is visible: register a **resampler-choice transform** in
`training/registries.py` drawing uniformly over `{soxr_hq, soxr_vhq, polyphase, linear}` per
sample, and apply it on the test path too (R2 — it is a transform, so it must be symmetric).
⚠️ `registries.py` measures time-invariance at registration; a resampler-choice step is
time-invariant and will register, but it changes the drawn stream, so it needs an I1b re-run.

**Output.** `rolloff_by_source.csv` + the LTAS overlay figure. The figure is also 2nd-stage report
material (결과 해석, 15 pts).

---

## A3 — Speaker inventory and grouping-atom count 🔴 Tier S, **no second chance**

**Compute.** Per source, extract the publisher's own speaker key at its native granularity —
Common Voice `client_id` from `validated.tsv`; LibriTTS-R the speaker directory (and its
`SPEAKERS.txt` gender/subset fields); Zeroth the speaker directory; LJSpeech is **one speaker**.
Count distinct speakers, files per speaker, hours per speaker. Then run
`training.folds.build_folds` feasibility at 5, 4, 3 and 2 folds using `speaker_ref_id` as the
grouping atom.

**Why it is meaningful.** [data/08](../data/08-build-plan.md) records this as *the one Phase-B
decision with no second chance*: `source_name` at corpus granularity collapses every speaker into
one fold group, and the smoke corpus proved the consequence — a manifest with corpus-granularity
`source_name` **validates clean** and then makes `build_folds` infeasible at every fold count
([data/12](../data/12-acquisition-status.md)). The information exists only in the publisher's
directory layout, and it is destroyed the moment files are flattened.

**What we discover.** The real number of independent voice groups, which is what decides whether
the repo's 5-fold design is achievable at all or whether it stays at the 2 folds the smoke corpus
was stuck on. ⚠️ **LJSpeech contributes exactly one group regardless of its 13,100 utterances** —
a fact that is invisible in a file count and decisive in a fold table.

**Output.** `speaker_inventory.csv`, `fold_feasibility_A.md`.
**Gate.** **G-EDA3** — ≥6 independent speaker groups per role, or the fold count drops and the
caveat travels into the ledger.

---

## A4 — Common Voice English subset design ⚠️ Tier S (blocking on corpus size)

**Compute.** From `validated.tsv`: the joint distribution of `client_id` × `accent`/`locale` ×
`up_votes`/`down_votes` × `gender`/`age` where present; clip duration; and the per-client clip
count. Then propose a subset: `N` clients × `M` clips, seeded, stratified by accent, and write the
chosen ids to `cv_en_subset.json`.

**Why it is meaningful.** 88.1 GiB against a ~20 h budget is a **13× over-supply**, and taking it
whole reproduces the MLAAD mistake in the other direction: Common Voice English would become ~60%
of a 70 h pool A, making "crowd-sourced mp3 English" the dominant REAL texture. DOSS is explicit
that per-domain capping beats volume, and ★ `[BirdCLEF 2024, 3rd]` caps at 500 per class for the
same reason.

**What we discover.** Whether accent and locale give us any handle on the **unknown test-set
language composition** — an open decision in [PROGRESS](../../PROGRESS.md) that currently has no
evidence attached to it at all. Common Voice is the only source in the corpus carrying accent
metadata.

**What it changes.** The `domain_cap` value in `configs/run_default.yaml` (currently 500) and the
corpus row count. 🔴 The subset, like the MLAAD cap, is **part of the corpus definition** — seed
it and record it, because change the seed and it is a different corpus.

---

## A5 — Duration census against the 4 s floor and the 60 s ceiling ⚠️ Tier S

**Compute.** Duration histogram per source, at 0.5 s resolution. Then three counts: files below
`duration_range[0]` = 4.0 s (invisible to the sampler), files above 60 s (the competition ceiling),
and files in 20–60 s. Repeat *after* silence trimming to see how much of the shortfall is silence.

**Why it is meaningful.** The smoke corpus's most limiting defect was exactly this and it was
discovered late: *"durations reach 4–10 s, not 4–60 s… `segmentation: whole_file` at 60 s is still
untested on real audio"* ([PROGRESS](../../PROGRESS.md)). Read speech is utterance-length. Common
Voice clips are ~5 s, LibriTTS-R utterances a few seconds. **If nothing in pool A reaches 20 s,
then no composed sample can reach 20 s either, and the whole-file segmentation path — which the
`fps` work, the frame arithmetic and the entire duration-bias analysis exist to serve — cannot be
exercised.**

**What we discover.** Whether V-B2 concatenation is *required* rather than optional to reach the
test set's upper duration range, and how many source files a 60 s sample must be built from.
⚠️ If concatenation becomes required, it must be applied symmetrically across labels or the splice
boundary becomes the cue — ★ PartialSpoof measures EER **worst at zero concatenation boundaries**,
so the direction of this trap is already known and counter-intuitive.

**Output.** `duration_hist_A.csv`; the count of files reaching each of the 4 / 10 / 20 / 60 s marks.

---

## A6 — Silence statistics vs pool B ⚠️ Tier A

**Compute.** `lead_silence_s`, `tail_silence_s`, `silence_ratio` (−50 dB energy segmentation) per
source; then the two-sample comparison **against pool B**, per head, as an AUC.

**Why it is meaningful.** Leading/trailing silence is *the* known ASVspoof shortcut
([survey/01](../survey/01-sota-speech.md)): detectors trained on that corpus learned silence
duration rather than synthesis artifacts. Our pool A is studio-trimmed (LibriTTS-R, LJSpeech) and
crowd-sourced with long lead-ins (Common Voice); our pool B is TTS output, which typically has
near-zero trailing silence. **This is the highest-probability single shortcut in the corpus** and
it is measurable in an afternoon with no decode beyond an energy envelope.

**What we discover.** Whether the shortcut is present and how strong. AUC here is directly
comparable to the E-S2 gate.

**What it changes.** [data/10](../data/10-preprocessing-and-filtering.md) P-A2 already argues
*against* trimming — trimming kills the shortcut and the genuine cue together — and prefers
**inoculation** via the `A-A11` silence-edit augmentation. This measurement is what decides whether
`silence_lead_s` / `silence_tail_s` (both currently **0.0** in `configs/run_default.yaml`, so no RNG
draw happens at all) must become non-zero. ⚠️ Turning them on changes the drawn stream and needs an
I1b re-run.

---

## A7 — Loudness, DC and clipping census ⚠️ Tier A

**Compute.** `lufs_integrated`, `peak_dbfs`, `crest_factor`, `dc_offset`, `clipping_ratio` per
source, both planes. Report the between-source variance against the within-source variance.

**Why it is meaningful.** Level is the cheapest corpus-identity cue that exists: LJSpeech is
normalized, Common Voice is not, LibriTTS-R was restored and re-levelled. P-A1 is explicitly marked
⚠️ because loudness normalization *removes a corpus-identity shortcut* and *may remove a genuine
cue* (generators have characteristic loudness) at the same time, and the doc says the choice is
"only defensible if the organizers normalized too" — which is E-S1's job to determine.

**What we discover.** The size of the between-source level gap, i.e. how much P-A1 is actually
worth. If between-source variance dwarfs within-source variance, normalization is mandatory
regardless of what the dummy files say.

**What it changes.** `P-A1` on/off, and its parameterization in the render chain. 🔴 Note that
`normalize` in the render path is **unparameterized until E-S1 lands**, so A-S1 — the
highest-leverage augmentation step — is structurally present and doing nothing today.

---

## A8 — Pool-membership verification ⚠️ Tier A

**Compute.** Silero VAD (thresholds 0.5 and 0.4) and PANNs tagging over the S-tier sample. Flag
files with `vad_speech_ratio < 0.2` (labelled voice, no voice found) and files whose top PANNs tag
is Music.

**Why it is meaningful.** Pool A rows assert `label_voice_present = 1`. A row that carries no
audible voice teaches the voice head from an absent component — [data/10](../data/10-preprocessing-and-filtering.md)
P1's *label-evidence sufficiency*, which is the only legitimate form of "this file is too noisy."

**What we discover.** The rate of unsupported labels per source, and whether any pool A source
contains music (Common Voice contains singing and background music in a nonzero fraction of
crowd-sourced clips).

**What it changes.** F-A1 says **reassign, do not drop** — a pool-A file with music in it is a
cell-5 sample, not garbage, and reassignment *adds* a hard positive rather than shrinking the pool.
Files with no evidenced voice get a validity mask (V-A2) or, if nothing survives, a drop with the
reason recorded.

---

## A1b — ✅ Answered 2026-09-16 (with B3 and C6): the format census, and where it separates the pools

`eda/out/_shared/census_format_census.parquet`, written by `python -m eda.cli roles`. One row per
source over all **382,068** census rows. A1 asks for it over pool A; [B3](02-pool-b-fake-voice.md)
asks for pool B's *against* pool A's, so it is one table and the comparison is a sort.

| source | pool | n | hours | container | codec | fmt | rate | ch | encoder present |
|---|---|--:|--:|---|---|---|--:|--:|--:|
| `cfad-real` | A | 38,600 | 57.5 | wav | pcm_s16le | s16 | 16k | 1 | 0% |
| `ljspeech` | A | 13,100 | 23.9 | wav | pcm_s16le | s16 | 22.05k | 1 | 0% |
| `musan-speech` | A | 426 | 60.4 | wav | pcm_s16le | s16 | 16k | 1 | 0% |
| `zeroth-korean` | A | 22,720 | 52.9 | **flac** | flac | s16 | 16k | 1 | 0% |
| `cfad-fake` | B | 73,700 | 69.5 | wav | pcm_s16le | s16 | 16k | 1 | 0% |
| `mlaad` | B | 16,006 | 34.6 | wav | pcm_s16le | s16 | 22.05k | 1 | 0% |
| `wavefake` | B | 117,983 | 198.7 | wav | pcm_s16le | s16 | 22.05k | 1 | 0% |
| `fma` | C | 8,000 | 66.6 | **mp3** | mp3 | **fltp** | **44.1k** | **2** | **100%** |
| `musan-music` | C | 660 | 42.6 | wav | pcm_s16le | s16 | 16k | 1 | 0% |
| `fakemusiccaps` | D | 27,605 | 77.3 | wav | **pcm_f32le** | **flt** | 16k | 1 | 0% |
| `compspoof-env-bonafide` | E | 13,172 | 14.6 | wav | pcm_s16le | s16 | 16k | 1 | 0% |
| `musan-noise` | E | 930 | 6.2 | wav | pcm_s16le | s16 | 16k | 1 | 0% |
| `rirs-isotropic-noise` | E | 92 | 0.8 | wav | pcm_s16le | s16 | 16k | **8** | 0% |
| `sonics` | cell 8 | 49,074 | 1,970.6 | **mp3** | mp3 | fltp | 16k | 1 | **100%** |

### ✅ B3's answer: pools A and B are format-indistinguishable

**Every pool-A and pool-B source is `pcm_s16le`/`s16` mono wav, except `zeroth-korean`'s flac.**
The rates are 16 kHz and 22.05 kHz on *both* sides. This is the good news B3 was looking for and did
not expect: there is **no format confound between real and fake voice**, so the voice heads' problem
is not a container fingerprint. It is consistent with [X2b](06-cross-pool.md)'s `voice_fake` being
the weakest metadata head (AUC 0.834 against `music_fake`'s 1.000).

### 🔴 …and the two places where format *is* the label

1. **`fakemusiccaps` is the only `pcm_f32le`/`flt` source in the corpus.** One column,
   `sample_fmt`, identifies pool D with certainty. That is the mechanism behind X2b's
   `music_fake` AUC of **1.000** — not a subtle statistical edge, a single categorical value.
2. **`fma` is the only 44.1 kHz source, the only stereo one, and one of two mp3 sources.** Four
   independent columns each isolate pool C.

So the music heads' metadata leak is not one confound but two, sitting on opposite sides of the
label, and neither exists between pools A and B.

### 🔴 C6's answer: the mp3 question is about **bit rate**, not about mp3

`census_codec_provenance.parquet`. Only two sources are lossy, and they differ by **7×**:

| source | container | n | bit rate p05 / median / p95 | Xing | LAME |
|---|---|--:|--:|--:|--:|
| `fma` | mp3 | 7,997 | 138.5k / **265.7k** / 320.0k | 100% | 50.5% |
| `sonics` | mp3 | 49,074 | 33.9k / **36.8k** / 40.1k | 100% | 100% |
| `zeroth-korean` | flac | 22,720 | 97.2k / 121.1k / 141.3k | 0% | 0% |
| everything else | wav | — | 256k – 2,048k, near-constant per source | 0% | 0% |

⚠️ **`sonics` at 36.8 kbps against `fma`'s 265.7 kbps is the cell-8-vs-pool-C boundary wearing a
codec.** Both are mp3, so "is it mp3" separates nothing; the bit rate separates them completely and
non-overlappingly (fma p05 138.5k against sonics p95 40.1k). SONICS is AI songs and FMA is real
music, so this is a **third** metadata route to the music label, independent of the two above.
Anything that reads `bit_rate` reads the label.

⚠️ Three `fma` rows have no container at all — the probe failures already recorded in `probe_error`.
They are counted here, not dropped (R2).

⚠️ `encoder` is present for 100% of both mp3 sources and 0% of everything else, so its
**missingness** is the fingerprint rather than its value — which is exactly what `build_design`'s
`_isna` indicators encode, and why the census reports `encoder_present` rather than only
`n_encoders`.

---

## A6b — 🔴 Answered 2026-09-17: the silence shortcut is real, and it **survives pairing**

`lead_silence_s`, `tail_silence_s` and `silence_ratio` over the 12,426 decoded pool-A and pool-B
rows, as an A-vs-B AUC on the `voice_fake` head. Gate: **AUC < 0.60**. No decode — the columns were
already in `signal.parquet`.

| column | native | chain |
|---|--:|--:|
| **`lead_silence_s`** | **0.6116** | **0.6087** |
| `silence_ratio` | 0.5561 | 0.5569 |
| `tail_silence_s` | 0.5311 | 0.5325 |

Only **leading** silence clears the gate, and the render chain does not touch it — 0.6116 → 0.6087.

### 🔴 The corpus-wide number is the *weakest* reading, not the strongest

A6 predicted a between-corpus artefact: studio-trimmed real against TTS output. The controls say
otherwise.

| comparison | what is held constant | `lead_silence_s` AUC |
|---|---|--:|
| corpus-wide A vs B | nothing | 0.6116 |
| **within `cfad`** | **one publisher ships both halves** | **0.7167** |
| **`ljspeech` ↔ `wavefake`** | **same utterance, same speaker, one vocoder apart** | **0.6391** |
| cross-publisher only | CFAD and the LJSpeech pair removed | **0.7336** |

**The shortcut is stronger inside a publisher than across the corpus.** The corpus-wide 0.61 is
*diluted* by pool A's own heterogeneity — `ljspeech` has a median lead of **0.00 s**, exactly like
the TTS sources — so mixing publishers hides the effect rather than manufacturing it.

🔴 **And it survives the strictest control the corpus can offer.** The `ljspeech ↔ wavefake`
comparison is [B1](02-pool-b-fake-voice.md)'s paired design: same utterance, same speaker, same
source recording, one vocoder apart. At **AUC 0.639** on leading silence alone, the vocoder is
changing the silence — reconstruction puts low-level noise where the original had digital silence.
That is a **synthesis artefact**, not a corpus artefact, and no amount of publisher balancing
removes it.

| source | pool | lead median | lead p90 | tail median |
|---|---|--:|--:|--:|
| `zeroth-korean` | A | 0.50 | 0.80 | 0.85 |
| `cfad-real` | A | 0.35 | 1.36 | 0.35 |
| `musan-speech` | A | 0.05 | 1.10 | 0.00 |
| `ljspeech` | A | **0.00** | 0.00 | 0.05 |
| `mlaad` | B | 0.10 | 0.45 | 0.40 |
| `cfad-fake` | B | 0.05 | 0.40 | 0.25 |
| `wavefake` | B | **0.00** | 0.20 | 0.10 |

### ⚠️ This refines "the voice heads are clean"

[06 X1d](06-cross-pool.md) reports `voice_fake` collapsing to **0.401** under an archive holdout, and
[RESULTS](RESULTS_FOR_ANALYSIS.md) carries that as *"the voice heads are clean"*. Both remain true
and neither covers this: X1d is a **multivariate** model measured under an **archive** holdout, and
this is a **single column** measured under a **same-utterance** control. The precise statement is
that the voice heads are clean of *publisher* confounds and are **not** clean of the silence
artefact, which is the one thing that survives pairing.

### What it changes — turn the silence augmentation on

[data/10 P-A2](../data/10-preprocessing-and-filtering.md) already argues **against trimming**:
trimming kills the shortcut and the genuine cue together. It prefers **inoculation** through the
`A-A11` silence-edit augmentation — and `silence_lead_s` / `silence_tail_s` are both **0.0** in
`configs/run_default.yaml`, so no RNG draw happens at all today.

🔴 **This measurement is what P-A2 said would decide it, and it decides for non-zero.** The
artefact clears the gate in every configuration measured, is strongest where the controls are
tightest, and survives the render chain untouched.

⚠️ Turning them on changes the drawn stream and needs an **I1b re-run**. And ⚠️ **the augmentation
must be symmetric** (R2): applied to pool A and pool B alike, or it becomes the cue it is meant to
inoculate against.
