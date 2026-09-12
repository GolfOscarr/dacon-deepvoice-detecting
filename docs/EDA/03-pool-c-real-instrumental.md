# 03 — Pool C: Real Instrumental

**12.6 GiB acquired.** `fma` (7.5 — `fma_small` 8,000 tracks + metadata) and `mtg-jamendo`
(5.1 — one `raw_30s` shard of 100). Plus MUSAN's `music/` partition, which the smoke corpus
already routes here as `musan-music-{fma,jamendo,hd}`. Target ~45 h ([data/04](../data/04-sources.md)).

Pool C is the real half of the **0.27-weight** music head. It is also the pool where the label we
assert — "instrumental, no vocals" — comes from an **uploader-supplied tag** rather than from
measurement.

---

## C1 — 🔴 Licence-allowlist intersection — Tier S, blocking, and nobody has computed it

**Compute.** Load `fma_allow.csv` and `jamendo_allow.csv` from
`s3://<bucket>/dacon-deepfake-detection/data/licences/`. Intersect with the track ids **actually
present** in `fma_small` (8,000 tracks) and in the single `raw_30s_audio-00.tar` shard. Report the
surviving count, and the surviving count per genre and per artist.

**Why it is meaningful.** [data/12 §4](../data/12-acquisition-status.md) is explicit: *"The archive
being cleared does not clear a track. Only ids in `fma_allow.csv` / `jamendo_allow.csv` may enter
the corpus, and `fma_small` is an 8,000-track subset, so intersect the allow list with whatever
subset is actually downloaded."* The allow rates are known over the **full** corpora — FMA 56.6%,
Jamendo 65.9% — but **the intersection with what we downloaded has never been taken**, and there is
no reason the rates are uniform across the subset: `fma_small` is a genre-balanced 8,000-track
sample, and licence choice correlates with genre and with artist.

**What we discover.** The true, licence-corrected size of pool C. 🔷 If the rates hold, ~4,500 FMA
tracks × 30 s ≈ **37 h**, plus the Jamendo shard — which lands near the 45 h target with no
headroom at all. If they do not hold, pool C may be the binding constraint on the music head and
nobody would currently know.

**What it changes.** Whether to pull more Jamendo shards (each is ~5.4 GiB, and the caveat in
`sources.yaml` says *"one or two shards is the right order, not the set"*). That decision needs
this number first.
**Gate.** **G-EDA1** — every pool-C row's track id is in the allow list. A row that is not is a
licence violation, not a data-quality issue.

---

## C2 — 🔴 Silero VAD sweep: is "instrumental" actually instrumental? Tier A (E-A5)

**Compute.** Silero VAD at thresholds **0.5 and 0.4** over every allowed pool-C track, plus PANNs
tags. Record `vad_speech_ratio`, `vad_spans`, and the PANNs scores for `Singing`, `Speech`,
`Music`, `Rapping`. Classify each track: clean instrumental / contains singing / contains speech.

**Why it is meaningful.** [data/04](../data/04-sources.md) flags Jamendo's "instrumental" tag as
uploader-supplied and untrustworthy, and FMA's genre tags say nothing about vocals at all. A pool-C
track containing vocals asserts `label_voice_present = 0` while carrying a voice — it poisons the
**presence** head directly and, once composed with a pool-B fake voice, produces a cell-7 sample
with two voices and one label.

**What we discover.** The contamination rate, per source and per genre.

**⚠️ Two calibration problems, both already flagged in this repo.** PANNs **conflates singing with
Music** — an open item in [PROGRESS](../../PROGRESS.md) that says *"check PANNs fires
`VOICE_PRESENT` on a sung song before trusting the day-one presence baseline; one song answers
it."* This EDA pass is where that one song gets played. And Silero VAD is trained on **speech**, not
singing: it will under-detect sung vocals, which is exactly the failure mode that matters here. So
run **both**, treat agreement as confident and disagreement as the manual-listen queue, and
calibrate on a hand-labelled set of ~50 tracks before trusting either threshold.

**What it changes.** F-A1's rule is **reassign, do not drop**: a Jamendo track with vocals is a
legitimate **cell-5** sample (mixed, both real) and cell 5 is the cell the composition trap most
needs non-composed examples of. Quarantining it would throw away the scarcest material in the
corpus. Output `pool_c_quarantine.csv` with a *reassignment* column, not a drop column.

---

## C3 — Artist / album grouping-atom inventory 🔴 Tier S, no second chance

**Compute.** From FMA's `tracks.csv` (`artist_id`, `album_id`) and Jamendo's `raw.tsv`
(`ARTIST_ID`, `ALBUM_ID`): distinct artists, tracks per artist, tracks per album. Run `build_folds`
feasibility at 5/4/3/2 using artist as the grouping atom.

**Why it is meaningful.** Same structure as [A3](01-pool-a-real-voice.md), same irreversibility.
Two tracks by the same artist share instrumentation, mixing, mastering chain and often the same
session — an artist straddling a fold boundary is the music-side equivalent of a speaker straddling
one. ⚠️ And an **album** is tighter than an artist: same mastering, often the same day. Group at
album level within artist, and let the union-find in `folds.grouping_atoms` collapse them.

**What we discover.** Whether artist-disjoint folds are feasible on 4,500-ish tracks, and the
artist concentration — FMA has a long tail of artists with one track and a head with dozens.

---

## C4 — 🔴 Genre census — and the C↔D matching problem — Tier A

**Compute.** Genre distribution from FMA's `genre_top`/`genres_all` and Jamendo's 195 tags, over
the **allowed** subset only. Then hold it against pool D's prompt/caption distribution
([04 D5](04-pool-d-fake-instrumental.md)) and compute the divergence.

**Why it is meaningful.** This is the pool-C finding with the largest expected value, and it is
structural rather than acoustic. Pool C is FMA + Jamendo: independent/electronic-heavy, Creative
Commons, self-released. Pool D is FakeMusicCaps: **MusicCaps captions**, which describe YouTube
audio — a very different genre and instrumentation distribution. If real music is
indie/electronic and fake music is whatever MusicCaps describes, then **genre predicts
`MUSIC_FAKE`**, with 0.27 of the metric riding on a confound that has nothing to do with synthesis.

This is [data/02](../data/02-label-taxonomy.md)'s composition trap generalized: that document
worries about *composedness* predicting the label and solves it with the `f_c` equality rule. The
same argument applies to **every** structural property that differs between the pools feeding
opposite sides of a label — and genre is one nobody has checked.

**What we discover.** The divergence, and the overlapping region. If the overlap is large, cap both
pools to it. If it is small, this is a hard argument for the self-generated pool D: ACE-Step
generation should be **prompted from pool C's own genre tags**, which is exactly why
[data/04](../data/04-sources.md) notes Jamendo's *"195 genre tags usable as TTM prompts"* — the
mechanism is already identified, this measurement is what makes it required rather than clever.

**What it changes.** The pool-D generation prompt distribution, and possibly a genre cap on pool C.

---

## C5 — Clip boundary and duration morphology ⚠️ Tier A

**Compute.** Duration histogram; and for each track the **onset and offset envelope shape** over
the first and last 0.5 s — does the clip begin at full level (a hard cut from the middle of a
track) or fade in? Classify: hard-cut / fade / natural start.

**Why it is meaningful.** FMA's `fma_small` is 30 s excerpts taken from track centres; Jamendo's
`raw_30s` likewise. **Both begin and end mid-phrase, at full level, with no fade.** A generated
pool-D clip is a complete generation with a beginning and an end. 🔷 If every REAL music sample
starts with a hard cut and every FAKE one starts from silence, the first 100 ms of the file
separates the classes perfectly, at 0.27 weight, and no spectral analysis would ever reveal it
because it is an *editing* artifact rather than an acoustic one.

**What we discover.** Whether the hypothesis holds, and the exact onset-level distribution per pool.

**What it changes.** A crop policy applied identically to C and D — draw every music component from
a random offset inside the file with the same crop length distribution, so onset morphology carries
no label information. ⚠️ This is a **transform**, so R2 applies: it is symmetric by construction
only if both pools go through it.

---

## C6 — Codec provenance and the mp3 question ⚠️ Tier A

**Compute.** `ffprobe` census over 100% of pool C: codec, bitrate, encoder/LAME tag, Xing header
presence, and the mp3 pre-echo signature at the `chain` plane. Compare against pool D.

**Why it is meaningful.** FMA ships mp3 at various bitrates; Jamendo `raw_30s` is mp3; MUSAN's
music partition is a re-encode of FMA and Jamendo material. Pool D will be **wav straight from a
generator**. So "was ever mp3" ≈ "is real music", the same confound as [B3](02-pool-b-fake-voice.md)
but on the head that carries more weight.

**What we discover.** The bitrate distribution — which doubles as the parameterization for the
codec stage. [data/10](../data/10-preprocessing-and-filtering.md) rejects filtering on bitrate
("bandwidth correlates with the telephone subset we most need"), so the answer is never to drop
low-bitrate tracks; it is to make bitrate history uninformative.

**What it changes.** 🔴 Mandatory mp3 round-trip on pool D before it can be used, with a bitrate
distribution **drawn from this census** rather than chosen. `training/render.py`'s `codec_aware`
stage is the implementation and `_PAD_TOL = 576` already handles LAME's trailing padding.

---

## C7 — MUSAN `music/` as pool C, not pool E ⚠️ Tier S (a correctness check)

**Compute.** Confirm every `musan/music/*` row is assigned pool **C** and every `musan/noise/*` row
pool **E**, and that no pipeline path feeds `music/` into the noise role.

**Why it is meaningful.** `scripts/sources.yaml` marks this Critical: *"MUSAN ships a `music`
partition. Never feed it in as noise — it would put real music into the noise role and corrupt the
music-present labels."* A pool-E draw asserts `label_music_present = 0`. Real music in that role
teaches the music-presence head that music is absent, and that head is 0.05 — but the same rows
also become the "noise" component of composed cells, so the damage is not contained.

**Why it is in the EDA rather than assumed.** The smoke corpus gets this right today. That is a
property of one builder script, not of the corpus, and **the next builder is where it breaks**.
⚠️ See also [05 E1](05-pool-e-noise.md): MUSAN's music subdirectories are literally named
`fma`, `jamendo` and `hd` — and we are separately acquiring FMA and Jamendo.
