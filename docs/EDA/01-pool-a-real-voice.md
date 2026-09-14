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
