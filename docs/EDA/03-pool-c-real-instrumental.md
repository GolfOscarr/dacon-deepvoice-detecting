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

## 🔴 C2b — ANSWERED: "instrumental" is not instrumental, and 13.7% is a floor

Run 2026-09-15 over all 2,660 decoded pool-C files, Silero VAD on the chain plane
(`eda report`, `eda.analyze.signal.content_report`).

| source | asserts | median speech | p90 | **speech ≥ 0.20** |
|---|---|--:|--:|--:|
| `fma` | `voice_present = 0` | 0.000 | **0.418** | **274 / 2,000 = 13.7%** |
| `musan-music` | `voice_present = 0` | 0.000 | 0.137 | 62 / 660 = 9.4% |

**Pool C contradicts its own assertion on 336 of 2,660 files — 12.6%.** The median
track really is instrumental, but the top decile of FMA is 42% speech, and every
one of those files tells the model there is no voice in audio that has one.

🔴 **And 13.7% is a lower bound, not an estimate.** Silero detects *speech*.
Measured on SONICS, whose publisher reports `no_vocal = False` for all 49,074
rows, it finds speech in only **28.1%** of files that certainly contain vocals
([00 §4e](00-harness.md#4e---what-the-vad-can-and-cannot-evidence)). Sung FMA
tracks are therefore invisible to it, and FMA is a music archive — the true
fraction carrying voice is higher, plausibly much higher.

⚠️ **This is pool C's whole job at stake.** It is the negative class for the
`music_fake` head, which carries **0.27** of the metric. A mislabelled negative
is worse here than a missing one.

### The threshold question C2 asked, answered too

C2 asks for 0.5 **and** 0.4 so the gap between them can be read. Measured, the gap
is **0.000 on every pool-C source** and at most 0.010 anywhere in the corpus. The
VAD is not equivocating: these files are confidently speech or confidently not,
and the second threshold buys nothing here. It stays recorded because the cost is
one column and the reasoning is only valid for *this* corpus.

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

## 🔴 C8 — S tier: 30 s against pool D's 10 s, and the only source the chain really cuts

Measured 2026-09-14 over 2,660 decoded files (`eda report`).

| | p05 | median | p95 | in 4–60 s | over 60 s |
|---|--:|--:|--:|--:|--:|
| pool C | 29.977 | **30.003** | 296.06 | 0.757 | **0.242** |

FMA ships 30 s excerpts and MUSAN's music partition ships whole tracks, so a quarter of this pool is
longer than anything the test set contains. Against pool D's flat **10.000 s**
([04 D8](04-pool-d-fake-instrumental.md)) that makes duration a near-perfect real/fake separator for
the music heads — the finding
[06 X1d](06-cross-pool.md#-x1d--x1-on-the-decoded-audio-the-duration-shortcut-survives-the-chain)
quantifies and the one the crop policy has to answer.

🔴 **`fma` is the source the 16 kHz chain cuts hardest**, and by a wide margin:

| source | native bandwidth | chain | lost | `hf_ratio_8k_native` |
|---|--:|--:|--:|--:|
| `fma` | 18,389 Hz | 8,000 Hz | **10,389 Hz** | 2.8e-03 |
| `musan-music` | 7,766 Hz | 7,766 Hz | 0 | 9.4e-12 |

So within one pool, one source loses more than half its spectrum to the chain and the other loses
nothing — which is a *domain* split inside pool C, not only a cross-pool one.

⚠️ And `fma` is the corpus's loudest, most clipped source: median peak **−0.63 dBFS**, median RMS
**−15.0 dBFS**, **24.7%** of files with at least one clipped sample, and a DC offset reaching 0.553.
`musan-music` is peak-normalised to full scale (median peak −0.000265 dBFS, **88.5%** clipping),
which is a publisher fingerprint rather than a property of music.

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

---

## C5b — 🔴 Answered 2026-09-16: the boundary shortcut is **real, and it runs backwards**

Measured over all **58,885** drawn files on the chain plane — every partition, not only C and D,
because the crop policy C5 proposes is a *transform* and R2 makes a transform a symmetry
obligation. 1 decode failure, ~5 min at 76 audio-hours/min (`python -m eda.cli envelope --run`).

### The hypothesis, and what actually happened

[C5](#c5--clip-boundary-and-duration-morphology--tier-a) predicted that **real** music is cut from
track centres and so begins at full level, while a **generated** clip is a complete piece and begins
from silence — and that the first 100 ms would therefore separate the classes at 0.27 weight
through an *editing* artifact no spectral analysis could surface.

**The separation is there. The direction is inverted.**

| partition | `hard_cut` | `fade` | `natural` | median onset deficit |
|---|--:|--:|--:|--:|
| **C** real instrumental | **12.1%** | 16.6% | **71.3%** | **43.35 dB** |
| **D** fake instrumental | **73.1%** | 15.7% | **11.2%** | **0.90 dB** |
| A real voice | 11.7% | 56.6% | 31.7% | 22.24 dB |
| B fake voice | 16.3% | 45.9% | 37.9% | 24.60 dB |
| E noise | 86.7% | 7.1% | 6.2% | 0.25 dB |
| cell 8 (SONICS) | 27.8% | 56.2% | 16.0% | 37.95 dB |

`onset_level_deficit_db` is how far below its **own** median level a clip opens, so 0 is a hard cut
and a large number is a quiet start. **Pool D opens at its own level; pool C opens 43 dB below it.**

### 🔴 `onset_level_deficit_db` separates C from D at **AUC 0.869**

| column | C vs D AUC |
|---|--:|
| **`onset_level_deficit_db`** | **0.8692** |
| `onset_class == hard_cut` | 0.8052 |
| `onset_lead_silence_s` | 0.7514 |
| `onset_rise_s` | 0.7064 |
| `offset_level_deficit_db` | 0.5986 |

Against the **0.60** gate. The interquartile ranges do not overlap — C p25 **34.53 dB** against D
p75 **6.93 dB**:

| | p05 | p25 | median | p75 | p95 |
|---|--:|--:|--:|--:|--:|
| C | −1.12 | **34.53** | 43.35 | 72.63 | 88.01 |
| D | −8.27 | −2.21 | 0.90 | **6.93** | 61.35 |

🔴 **This is a second shortcut of the same magnitude as duration, on the same head, and independent
of it.** Duration holds AUC 0.852 after a whole-publisher holdout
([06 X1d](06-cross-pool.md)); this holds 0.869. One is the file's *length*, the other its first
**20 milliseconds** — no crop that fixes one automatically fixes the other, and a model that lost
duration would still have this.

**The mechanism, stated as inference rather than measurement:** excerpting a track from its centre
leaves a discontinuity, and the standard remedy is a short de-click ramp at the cut. The **real**
pool therefore carries the edit and the **generated** pool carries none, because a generation simply
begins. C5's instinct — *"an editing artifact rather than an acoustic one"* — was exactly right; the
sign was assumed rather than measured, and the assumption was wrong.

### Two alternative explanations, ruled out before publishing

1. ⚠️ **mp3 decoder padding.** FMA is pool C's only mp3 source and encoder delay would counterfeit
   this precisely. **Refuted per source:** `musan-music`, a **wav** source, shows the largest
   deficit in the corpus at **77.04 dB**, against `fma`'s 39.71. Both pool-C sources carry it, so
   the container does not explain it.
2. ⚠️ **The classifier measuring loudness compression rather than editing.** A compressed signal has
   its first frame near its own median by construction, so `hard_cut` could have meant "heavily
   limited". **Refuted:** `corr(onset_level_deficit_db, crest_factor_db_chain)` is |r| ≤ 0.33 in
   every partition and under 0.15 in most.

### What it changes

🔴 **The crop policy moves from optional to mandatory.** Drawing every music component from a random
offset inside the file is the only thing that removes this, and R2 makes it symmetric by
construction only if **both** pools go through it — C5 said so and the number now says how much it
costs to skip.

⚠️ **`onset_*` and `offset_*` are `leakage_risk` columns**, not features, and belong in
[`roles.md`](06-cross-pool.md) as such the moment anything consumes them.

⚠️ **Pool E is 86.7% `hard_cut` and pool A/B are fade-dominated**, which is why this was measured
everywhere. A crop policy tuned on C-vs-D alone would be applied to a corpus where the voice pools
sit at 11.7% and 16.3% — close to each other, and nothing like either music pool.

### Limits of the classification, stated

* The thresholds (`HARD_CUT_DB` 6 dB, `FADE_MIN_S` 50 ms, `BOUNDARY_S` 0.5 s) are **a reading, not a
  measurement**. `onset_level_deficit_db` and `onset_rise_s` are stored beside the class so anyone
  who disagrees can re-derive it without a re-decode — and the AUC above is computed on the
  **scalar**, not the class, for that reason.
* ⚠️ **A digitally silent file has a flat envelope**, so its deficit against its own median is 0 and
  it reads as `hard_cut` — silence would otherwise count as evidence for the very asymmetry this
  tests. 93 pool-E rows are excluded on `envelope_ref_dbfs <= -100` and **counted** in the report.
* The `fade` class for speech is really "gradual onset": a talker ramping up over more than 50 ms is
  indistinguishable here from a deliberate fade. It does not affect the C-vs-D reading.

---

## C1b — ✅ Answered 2026-09-16: the allowlist is **stale**, not restrictive — and the split is 23 / 2,884

`G-EDA1/allowlist/fma` fails on **2,907 of 8,000** pool-C files, and that number has been carried as
*"a licence decision, open"*. It reads like a licence problem. It is not one.

Measured against the licence ledger on
`s3://hyeonseop-s3/dacon-deepfake-detection/data/licences/`:

| bucket | files of the 2,907 | what it means |
|---|--:|---|
| **DENY** | **23** | the licence restricts provision. Genuinely unusable |
| **ND or unlisted** | **2,884** | ⚠️ **already resolved as usable** |

🔴 **The ND call was answered and the allowlist was never regenerated.**
[`scripts/filter_track_licences.py`](../../scripts/filter_track_licences.py) deliberately parked the
ND family in a third bucket and said so in its own docstring — *"the call is a legal read (the same
one CtrSVDD is waiting on), not something this script should decide."* That read came back **yes**:
[#417333 A5](../competition/05-talkboard-qa.md) — *"CC BY-NC-ND data may be used, augmentation
included, provided the work is reproducible from 원본 파일 + 코드"* — resolved **2026-09-08**, the
same day `fma_allow.csv` was generated at 20:08.
[data/01](../data/01-rules-check.md) and [survey/10 V2](../survey/10-open-questions.md) both record
it as closed; the artifact was simply never rebuilt.

### ⏭️ Deferred, with a trigger — **not** an open licence question

Regenerating is ~10 minutes: merge `allow ∪ derivatives_barred` into `fma_allow.csv` keeping the
`verdict` column so the ND provenance stays visible, then re-run `eda gates`. It would take
`G-EDA1/allowlist/fma` from **fail** to pass on ~7,977 of 8,000.

**It is deferred because the benefit is volume, and volume is not pool C's constraint.**

| | |
|---|---|
| what it adds | **+57%** allowed files (5,093 → ~7,977) |
| what it does **not** add | a publisher. Pool C is two sources — `fma` 66.6 h, `musan-music` 42.6 h — and stays two |
| what it does **not** dilute | 🔴 neither shortcut. Duration (30.003 s) and the 43 dB onset deficit ([C5b](#c5b---answered-2026-09-16-the-boundary-shortcut-is-real-and-it-runs-backwards)) are properties of **FMA's excerpting**, and the recovered files are FMA excerpts too |
| what it does **not** change | any EDA number. The S-tier draw takes 2,000 `fma` files, well under the 5,093 already allowed |

⚠️ And pool C is not volume-starved: **109.2 h of real music against pool D's 77.3 h**. The corpus
is fake-heavy on this head, so adding to the real side is the less useful direction.

**Trigger: do it when pool C volume becomes the binding constraint on the music head — not before.**
The cost is the ~10 minutes plus accepting A5's condition as a *pipeline obligation*: the 2nd-stage
submission must ship 원본 파일 + 코드 and never a redistributed derivative. That is already
[data/06](../data/06-augmentation-spec.md)'s design, so it costs nothing architecturally, but it
binds the submission format.

### 🔴 The lead this actually produced: `mtg-jamendo`

The ND resolution's value is not fma. **`mtg-jamendo` would be pool C's *third* publisher** — which
is the constraint that binds — and its allowlist carries the identical staleness: 36,728 `allow`
rows, with a `derivatives_barred` bucket roughly 60% that size again. A5 approximately doubles what
Jamendo would contribute.

It is **not on disk**, so that is an *acquisition* decision rather than EDA work, and
`G-EDA1/mtg-jamendo` correctly reads `na` until it lands.

⚠️ For the record, because it is the obvious next guess and it is wrong: **ND does not unblock
CtrSVDD.** Its blocker is a label-file join — 128,029 rows are 109,313 deepfake and **18,716
bonafide**, and the label lives only in `train.txt`/`dev.txt` keyed by utterance id
([`configs/eda.yaml`](../../configs/eda.yaml)). Nothing about licensing changes that.
