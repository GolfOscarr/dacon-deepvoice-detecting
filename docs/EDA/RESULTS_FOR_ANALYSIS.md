# EDA results — an index for an analyst

**You are reading this because you have been asked to analyse this corpus's EDA.** This page is a
map, not a summary: it says what was measured, where each result physically lives, which numbers are
authoritative, and — most importantly — **which readings of them are wrong**. Section 5 is the one
that will save you the most time.

**Written 2026-09-15, rewritten 2026-09-16 and 2026-09-17.** The EDA is complete: **all 34 runnable tasks** of
[10](10-final-plan.md) have written answers. ⚠️ The 2026-09-15 version of this page said the same
thing on the strength of [09](09-next-steps.md)'s eight steps, and **that was wrong** — 09 was a
remaining-work plan, never the task inventory. [10](10-final-plan.md) is the inventory; read its
§1 before believing any completeness claim, including this one.

> **The one-line version.** The corpus has **two independent shortcuts of near-equal strength on the
> music head**: **duration** (AUC 0.852 after a whole-publisher holdout) and the **first 20 ms**
> (AUC 0.869 between pools C and D). Both survive the 16 kHz render chain, neither is acoustic, and
> only the sampler can fix either. Everything else is detail.

---

## 1 — Read in this order

| if you want | read |
|---|---|
| the conclusions, with numbers | **[`data_memo.md`](data_memo.md)** — the exit document, one page |
| why each measurement exists | [`00-harness.md`](00-harness.md) |
| the shortcut analysis | [`06-cross-pool.md`](06-cross-pool.md) — X1, X1b, X1c, X1d, X6b |
| one pool's findings | [`01`](01-pool-a-real-voice.md)–[`05`](05-pool-e-noise.md) by pool letter |
| **the task inventory, and what is out of scope** | **[`10-final-plan.md`](10-final-plan.md)** — 33 answered, 7 blocked, 2 Phase 2 |
| what was decided not to do, and why | [`09-next-steps.md`](09-next-steps.md) — ⚠️ superseded by 10 |

⚠️ **Do not treat the prose as the source of truth.** Every figure in those documents came from a
parquet file listed in §3. Where they disagree, **the artifact wins** — and §6 tells you which
documents are dated snapshots rather than live claims.

---

## 2 — The corpus

**382,068 files · 2,676 audio hours · 14 runnable sources · 16,700 grouping atoms.**
`probe_ok` 382,065 — the 3 failures are known-unreadable pool-C files, recorded rather than dropped.

| partition | role | files | hours | S-tier rows | how sampled |
|---|---|--:|--:|--:|---|
| A | real voice | 74,846 | 194.8 | 6,426 | 2,000/source |
| B | fake voice | 207,689 | 302.8 | 6,000 | 2,000/source |
| C | real instrumental | 8,660 | 109.2 | 2,660 | 2,000/source |
| D | fake instrumental | 27,605 | 77.3 | **27,605** | full |
| E | non-musical sound | 14,194 | 21.6 | **14,194** | full |
| cell 8 | whole-file AI songs | 49,074 | **1,970.6** | 2,000 | 2,000, spread over 5 generators |

🔴 **One source is 74% of the audio.** SONICS' 49,074 songs are 1,970.6 of the 2,676 hours. Counting
*files* hides this completely; anything you reason about in "corpus proportions" must say which unit.

**Labels per head** (M tier; `not scored` means the metric ignores that head for that row):

| head | metric weight | scored | positive | rate |
|---|--:|--:|--:|--:|
| `voice_present` | 0.05 | 382,068 | 331,609 | 0.868 |
| `music_present` | 0.05 | 382,068 | 85,339 | 0.223 |
| `voice_fake` | 0.18 | 331,609 | 256,763 | 0.774 |
| `music_fake` | **0.27** | 85,339 | 76,679 | **0.899** |

⚠️ The file head — **0.45 of the metric** — is *not* auditable from this corpus. A composed file's
fake status does not exist until the sampler draws a spec; that is X5's question, over the spec
stream.

---

## 3 — Where the results physically are

All paths are relative to the repository root. Load with pandas / numpy; nothing here needs the
`eda` package, though §7 gives the convenience loaders.

### Per partition — `eda/out/{A,B,C,D,E,cell8}/`

| file | rows | what it is |
|---|--:|---|
| `files.parquet` | the full partition | **M tier**, 100% of files, 30 columns |
| `signal.parquet` | the S-tier draw | **S + C tiers**, 90 columns |
| `vectors.npz` | same rows as `signal.parquet` | **V tier**, 6 arrays of `(n, 128)` |
| `sample.json` | — | which files were drawn, the seed, and a fingerprint |

🔴 **`vectors.npz` is keyed by position, not by join.** Its `file_id` array is in the same order as
`signal.parquet`'s rows, and the merge refuses to write if they ever disagree. Do not sort either
one independently and then zip them.

```
vectors.npz keys:  file_id (n,) object
                   ltas_native, ltas_chain                   (n, 128) float32
                   mel_band_skew_native, mel_band_skew_chain (n, 128) float32
                   mel_band_kurt_native, mel_band_kurt_chain (n, 128) float32
```

### Corpus-wide — `eda/out/_shared/`

| file | shape | what it answers |
|---|---|---|
| `gates.parquet` | 13 × 3 | every gate's verdict and its reason — **6 pass · 4 fail · 3 `na`** |
| `shortcut_audit.parquet` | 4 × 7 | **X1**: can metadata alone predict each head? |
| `grouping_report.parquet` | 14 × 8 | per source: the grouping key, its kind, how many atoms |
| `depth_profile.parquet` | 29 × 7 | per source and path depth: cardinality |
| `duplicates.parquet` | 129 × 30 | **E1 pass 1**: byte-identical files |
| `content_duplicates.parquet` | 868 × 12 | **E1 pass 2**: LTAS-similar pairs — ⚠️ see §5.4 |
| `signal_duration.parquet` | 6 × 12 | duration and silence per partition, vs the test window |
| `signal_bandwidth.parquet` | 14 × 8 | what the 16 kHz chain removes, paired per file |
| `signal_level.parquet` | 14 × 9 | loudness, headroom, DC, clipping per source |
| `signal_content.parquet` | 14 × 11 | **G-EDA6**: VAD evidence vs what each pool asserts |
| `roles.md` | 118 rows | **X6**: every column's role. ⚠️ generated, not committed — `eda roles` |
| `column_roles.parquet` | 118 × 7 | the same, as a table |
| `envelope.parquet` | 58,885 × 14 | **C5**: onset/offset morphology, chain plane |
| `envelope_report.parquet` | 6 × 14 | **C5** read: the class mix per partition |
| `screen_degenerate.parquet` | 6 × 10 | **B4/D6**: the degenerate-output rate per partition |
| `screen_cell9.parquet` | 3 × 7 | **E5**: cell-9 viability |
| `screen_separability_generator.parquet` | 41 × 7 | **B6**: per-generator identifiability |
| `screen_metadata_leak.parquet` | 4 × 8 | **X2**: metadata vs chain-plane AUC, and the drop |
| `census_format_census.parquet` | 14 × 13 | **A1 / B3**: native format per source |
| `census_codec_provenance.parquet` | 15 × 10 | **C6**: the mp3 question |
| `census_generator_census.parquet` | 231 × 10 | **D2**: per generator |
| `census_near_nyquist_by_generator.parquet` | 159 × 9 | **D3**: both planes, per generator |
| `census_mlaad_inventory.parquet` | 534 × 6 | **B2**: MLAAD language × generator |
| `b1/` | 8 files | **B1**: the paired vocoder experiment |

---

## 4 — What was measured, and the findings

**56.0 M individual measurements** over four tiers. Every S and V column exists **twice** — `_native`
(the file as published) and `_chain` (after the 16 kHz resample the competition mandates) — because
each file is decoded once and measured on both, so it is its own control.

| tier | population | extractors | per-file values |
|---|--:|---|--:|
| M metadata | 382,068 | `identity`, `ffprobe`, `mp3_header` | 18 |
| S signal | 58,885 | `level` (9), `spectral` (14), `timing` (6) | 29 × 2 planes |
| V vectors | 58,885 | `ltas`, `mel_band_skew`, `mel_band_kurt` | 768 + 2 |
| C content | 58,884 | Silero `vad` | 7, chain plane only |
| **E envelope** | **58,884** | `envelope` (C5) | 9, chain plane only |

### The findings, ranked by what they change

**1 · Duration is the shortcut, and the render chain cannot touch it.**
`music_present` scores **AUC 0.852** after holding out a whole publisher, against a gate of 0.60.
Top features: `longest_valid_span_s` 0.943, `duration_s_decoded` 0.922. The cause is structural —
pool D is **10.000 s for every file** (p05 = p50 = 10.000, p95 = 10.242) and pool C is 30.003 s,
while the test set is 4–60 s for both.
→ [06 X1d](06-cross-pool.md), [04 D8](04-pool-d-fake-instrumental.md), [03 C8](03-pool-c-real-instrumental.md)

**2 · 🔴 The first 20 ms is a second shortcut, as strong as duration and independent of it.**
`onset_level_deficit_db` — how far below its own median level a clip opens — separates pool C from
pool D at **AUC 0.869**, against the same 0.60 gate. Pool C opens **43.35 dB** below its own level,
pool D **0.90 dB**; the interquartile ranges do not overlap. ⚠️ **The direction is the opposite of
what C5 predicted**: the *real* pool carries the editing artefact (a de-click ramp at the excerpt
cut) and the *generated* pool carries none, because a generation simply begins. Ruled out before
publishing: mp3 decoder padding (`musan-music`, a **wav** source, shows the largest deficit in the
corpus at 77.04 dB) and loudness compression (|r| ≤ 0.33 against `crest_factor_db_chain`).
→ [03 C5b](03-pool-c-real-instrumental.md)

**3 · The vocoder artefact survives 16 kHz at 82–115%, and lives below 72 Hz.**
500 LJSpeech utterances against all seven WaveFake vocoders — the only unconfounded real/fake
comparison in the corpus. For five of seven the dominant term is sub-speech rumble the vocoder
invents (+5.58 dB for `parallel_wavegan`). 🔴 **Consequence: do not high-pass.** A render-time DC or
rumble filter would delete the largest discriminative feature pool B has. And the vocoder **names
are anti-correlated with the artefact families** — `melgan` ↔ `melgan_large` correlate at 0.895,
`melgan` ↔ `multi_band_melgan` at **−0.186**.
→ [02 B1b](02-pool-b-fake-voice.md)

**4 · Metadata is worse than useless: AUC 1.000 in CV, 0.0000008 under an archive holdout.**
`music_fake` is separable from metadata *alone* at 1.000 — `fakemusiccaps` is the corpus's only
`pcm_f32le` source, so one categorical value identifies the pool. Under a source holdout the same
features score 0.209 and 0.0000008: the relationship **inverts** on an unseen publisher. **Decision
recorded: neutralise. Per-file metadata is not a submission feature.**
→ [06 X2b](06-cross-pool.md), [01 A1b](01-pool-a-real-voice.md)

**5 · Duration is also a *generator* id.** All five FakeMusicCaps generators have a **single
duration each** across all 27,605 files — `mustango` 10.242 s, `MusicGen_medium` 10.180 s, the other
three 10.000 s — so `mustango` separates from its pool-mates at **AUC 1.000** on duration alone. A
generator-disjoint fold in pool D is recoverable from the clock.
→ [02 B6b](02-pool-b-fake-voice.md), [04 D2b](04-pool-d-fake-instrumental.md)

**6 · 🔴 Leading silence is a synthesis artefact, and it survives pairing.**
`lead_silence_s` separates real from fake voice at **AUC 0.612** corpus-wide, **0.717** inside a
single publisher that ships both halves (CFAD), and **0.639** between LJSpeech and its own vocoded
copies — same utterance, same speaker, one vocoder apart. The shortcut is **stronger where the
controls are tighter**: mixing publishers *hides* it, because `ljspeech`'s median lead is 0.00 s,
exactly like the TTS sources. The chain does not touch it (0.6116 native → 0.6087 chain).
**Decision: turn on the `A-A11` silence-edit augmentation** — `silence_lead_s` / `silence_tail_s`
are 0.0 today, so no RNG draw happens at all. ⚠️ symmetric across pools A and B, or it becomes the
cue it inoculates against.
→ [01 A6b](01-pool-a-real-voice.md)

**7 · The voice heads are clean *of publisher confounds*.** Under the same holdout they collapse to
0.586 and 0.401. Whatever separates real from fake *voice* in this corpus is not a publisher
artefact. ⚠️ That is a **multivariate** result under an **archive** holdout and does not cover
finding 6, which is a single column under a same-utterance control.
→ [06 X1d](06-cross-pool.md)

**8 · Above 8 kHz is unavailable.** Ten of fourteen sources are natively ≤ 16 kHz, so the chain
removes nothing from them (`bandwidth_lost_hz` 0.000, `hf_ratio_8k_native` ~1e-09 to 1e-12). Only
`fma` (−10,389 Hz), `wavefake` (−3,220), `ljspeech` (−2,853) and `mlaad` (−2,530) lose anything, and
those four sit in three different pools.
→ [00 §4d](00-harness.md#4d---r1-on-the-whole-corpus-ten-of-fourteen-sources-lose-nothing)

**9 · Level is a publisher fingerprint.** Median RMS spans **22.6 dB** across sources on the chain
plane. All three MUSAN partitions sit at peak −0.000265 dBFS with 86–88% of files clipping:
peak-normalised to full scale.
→ [01 A9](01-pool-a-real-voice.md)

**10 · "Instrumental" is not instrumental.** Pool C contradicts its own `voice_present = 0` on
**336 of 2,660 files, 12.6%** — FMA at 13.7%, p90 speech ratio 0.418. Pool D at 3.6%. **Both are
lower bounds** (§5.1).
→ [03 C2b](03-pool-c-real-instrumental.md), [04 D9](04-pool-d-fake-instrumental.md)

**11 · 41% of pool B is shorter than the test set's minimum**, and only 10.3% of it holds a 4 s
non-silent span — the least usable material in the corpus.
→ [02 B0c](02-pool-b-fake-voice.md)

**12 · Two structural leaks were found and fixed.** WaveFake ships its Common-Voice half twice
(16,283 byte-identical files); CompSpoof shares **292 parent recordings** across its two splits
(1,060 files), which the path-derived key did not catch.
→ [02 B0b](02-pool-b-fake-voice.md), [05 E1b](05-pool-e-noise.md)

---

## 5 — 🔴 Traps: readings that are wrong

Each of these was actually walked into during the EDA and corrected. They are the highest-value part
of this page.

### 5.1 · A speech VAD cannot see singing

`signal_content.parquet` reports `sonics` with **median speech ratio 0.030** and 1,439 of 2,000 rows
below the evidence threshold, against an asserted `voice_present = 1`.

**That is not evidence of mislabelling.** SONICS' own `fake_songs.csv` reports `no_vocal = False`
for **all 49,074 rows** — every file has vocals. Silero detects *speech*; SONICS **sings**. Read
naively this says *72% of cell 8 has no voice*, and acting on it would relabel 49k AI songs as
voiceless — the exact defect that made SONICS a `whole_file` cell-8 source rather than a pool-D
component.

Those rows are counted in the **`unevidenceable`** column, not `contradicted`. And the consequence
travels: **every contradiction count on a music source is a lower bound**, because the sung
remainder is invisible to the detector.
→ [00 §4e](00-harness.md#4e---what-the-vad-can-and-cannot-evidence)

### 5.2 · `group_key` is the *weaker* holdout, not the stronger one

`shortcut_audit` and `signal_audit` accept two holdout columns and they answer different questions:

| holdout | asks | `music_present` |
|---|---|--:|
| `source_name` | does this survive an **unseen publisher**? | **0.852** |
| `group_key` | does this survive an unseen clip/speaker from a publisher **already in training**? | **0.902** |

Holding out one FakeMusicCaps clip leaves 5,520 others in training, so an archive-level confound is
fully available — which is why it scores *higher*. `group_key` is what makes `music_fake` measurable
at all (5,764 groups against 4 sources, giving 0.892), and **that number must never be quoted as an
archive holdout**.

### 5.3 · `na` is not `pass`

Gate verdicts are tri-state and ordered `fail > na > pass`. `na` means *the input does not exist
yet*, never *we looked and found nothing*. **Three** gates are `na` and none of them is a clean bill.

The four failures are not equivalent either: `G-EDA1/allowlist/fma` is a licence partition,
`G-EDA2` is structural and **cannot** go green here (X1 reads the corpus on disk; every fix is a
render-time transform), `G-EDA3`'s remaining shortfalls are facts about the sources, and `G-EDA6` is
a work list of 2,442 rows to reassign.

### 5.4 · `content_duplicates.parquet` is a shortlist, not a verdict

Its 868 pairs are LTAS cosine ≥ 0.999. The median file's best match among 58,883 others is **already
0.965**, and decoded top hits differ by up to **0.61 in amplitude** — two different recordings of a
bus agree spectrally because buses sound like buses. It **shortlists candidates**; it does not
confirm duplicates. The real leak it led to was confirmed by filenames, not by the metric.

### 5.5 · `nyquist_hz` is metadata wearing an acoustic name

`nyquist_hz_chain` is 8000.0 on every row; `nyquist_hz_native` is `orig_sr / 2`. Including it in a
"what does the audio leak?" model re-measures the metadata confound and reports it as acoustic
evidence. `eda.analyze.signal.CHAIN_EXCLUDED` drops it; do the same in any new design matrix.

---

### 5.6 · C5's hypothesis is stated backwards in the spec

[03 C5](03-pool-c-real-instrumental.md) predicts that **real** music begins with a hard cut and
**fake** music begins from silence. The spec text is still there and is still worth reading for the
mechanism — but the measurement
([C5b](03-pool-c-real-instrumental.md#c5b---answered-2026-09-16-the-boundary-shortcut-is-real-and-it-runs-backwards))
found the **opposite**: pool D opens at its own level (73.1% `hard_cut`) and pool C opens 43 dB
below it (71.3% `natural`).

Anyone who reads the hypothesis and implements a mitigation for the predicted direction will reason
about the crop policy the wrong way round. The shortcut is real and strong (AUC 0.869); only its
sign was assumed rather than measured.

⚠️ It was also nearly closed **without** measuring, on the argument that pool C's fixed 30 s
excerpts are hard cuts by construction. That argument is about pool C alone, and C5's claim is a
*comparison* — the inference covered one arm of a two-sided claim and reported the claim as settled.

### 5.7 · `group_key` is not the generator

`group_key` is the **fold-grouping** atom — the speaker, the parent clip. The **generator** is the
stratification axis, and in pool D they are incommensurable: `group_key` there is the FakeMusicCaps
parent clip, giving **5,521 groups averaging 5 rows**. A per-generator analysis keyed on `group_key`
returns *nothing* in pool D, and an empty result reads like a broken query rather than a wrong axis.

Use `eda.analyze.screens.generator_key`. It reproduces the published counts independently — CFAD
**11**, FakeMusicCaps **5**, WaveFake **10**, MLAAD **133** of 175 present in the draw.

⚠️ Its `GENERATOR_DEPTH` table was wrong in **three of its first four entries**, and every error was
silent: one component too shallow returns the CFAD split (`fake_clean`), the MLAAD **language**
(`bg` — 54 of them, which looks exactly like a plausible generator count) or WaveFake's archive
directory. Each produced a confident, plausible, wrong table. The function now refuses a depth that
resolves to one constant per source.

### 5.8 · `std == 0` does not find a constant column

`mustango` and `MusicGen_medium` are each a **single repeated duration** (10.242 s and 10.180 s)
across 5,521 files, and `pandas.Series.std(ddof=0)` returns **`1.776357e-15`** for them, not `0.0` —
float error accumulated over thousands of identical addends.

A `std == 0` test therefore finds **3** of FakeMusicCaps' **5** constant-duration generators. The
first draft of [04 D2b](04-pool-d-fake-instrumental.md) said *"3 of 231"*; the answer is 5 of 231.
`census_generator_census.parquet` carries an exact `duration_constant` column computed as
`min == max`. Use it rather than re-deriving from `duration_s_std`.

### 5.9 · A threshold written relative to its own constant is not tested

Not a reading of the data — a warning about the **tests**, and it is the one that most affects how
much you should trust a number here. Several tests in this repo were written as
`assert classify(HARD_CUT_DB + 1, ...) == "fade"`. Scaling the constant scales the assertion with
it, so the mutant is invisible **by construction**: widening `HARD_CUT_DB` from 6.0 to 60.0 left
every test in `tests/test_eda_envelope.py` green.

Every threshold in `eda/extract/envelope.py` now has a literal-pinned test. If you add a threshold
anywhere, pin it with a literal or `scripts/mutate_eda.py` will report a kill it did not earn.

---

## 6 — Provenance: which numbers are live

| | |
|---|---|
| **Authoritative** | the parquet files in §3. Regenerate with §7 |
| **Dated snapshots** | [06 X1b](06-cross-pool.md) and [06 X1c](06-cross-pool.md) state their own population — X1c is **264,085 rows / 13 sources, 2026-09-12, before WaveFake landed**. Its numbers do not match today's `shortcut_audit.parquet` (382,068 rows / 14 sources) and are not meant to |
| **Current** | [06 X1d](06-cross-pool.md) and everything in §4 above are the post-WaveFake corpus |

⚠️ If you re-run `eda analyze`, `shortcut_audit.parquet` and `gates.parquet` are **overwritten**.
They describe whatever census was on disk at the time; `files.parquet`'s row count is the check.

---

## 7 — Regenerating, and the convenience loaders

```bash
V=/data/project/private/dacon-venvs/dacon311/bin/python
$V -m eda.cli sources | keys | report | analyze | gates    # read-only, no decode
$V -m eda.cli screens | roles | b1 | envelope              # read-only, no decode
$V -m eda.cli probe | consolidate | sample | signal        # rebuild the tiers (~8 h for signal)
$V -m eda.cli envelope --run                               # C5's pass, ~5 min over the corpus
```

```python
from eda.config import load_eda_config
from eda.driver import load_files, load_signal, load_vectors
cfg = load_eda_config("configs/eda.yaml")
files  = load_files(cfg)            # M tier, 382,068 rows, all partitions
signal = load_signal(cfg)           # S + C tiers, 58,885 rows, joined to M on file_id
vec    = load_vectors(cfg, "D")     # V tier, one partition at a time
```

⚠️ `load_signal` is an **inner join** to the M tier: an S-tier row with no census row cannot be
labelled, and a census row outside the draw has no S-tier statistics.

Labels are not columns — derive them, because two label sources exist and `row_kind` picks between
them (a component's labels come from its **pool**, a whole-file row's from its **cell**):

```python
from eda.analyze.shortcut import HEADS, head_labels
y = head_labels(files, "music_fake")     # NaN where the metric ignores the row
```

---

## 8 — What is *not* measured

Stated so you do not go looking:

* **No PANNs / audio tagging.** Decided, not skipped: a 32 kHz model against a 16 kHz plane, with
  the same singing blind spot as the VAD from the other side ([09 §7](09-next-steps.md)).
* **Sung voice is not evidenced at all** (§5.1).
* **`FILE_FAKE`, 0.45 of the metric**, is not auditable here — it is X5's question over the spec
  stream.
* **The `mixed` partition is unprobed** — `compspoof-v2` is declared, on disk, never enumerated.
* **Four sources are not on disk**: `common-voice-en`, `common-voice-ko`, `libritts-r`,
  `mtg-jamendo`. Six are blocked, each with a recorded reason.
* **Gates `na`**: G-EDA1/mtg-jamendo (unfetched), G-EDA4 (needs a fold table), G-EDA7 (needs a
  filter; Phase 0 applies none by design).
* **The looping/babble detector** B4 and D6 ask for. It needs the mel envelope **over time**, and
  the V tier stores per-band statistics over time, not the series. It needs a decode
  ([02 B4b](02-pool-b-fake-voice.md)).
* **Music presence is never evidenced**, only speech. Every cell-9 count in
  [05 E5b](05-pool-e-noise.md) is an upper bound for that reason (§5.1).

---

## 9 — If you are looking for features

The conclusions the EDA hands forward:

1. **Two shortcuts, not one, and they are independent.** Duration (AUC 0.852) is the file's
   *length*; onset deficit (AUC **0.869**) is its first *20 ms*. A crop that fixes one does not fix
   the other — **only a random offset inside the file removes both**, and R2 makes it symmetric by
   construction only if pools C and D both go through it.
2. **Nothing in `files.parquet` is a feature.** All 30 M-tier columns are a leakage risk, a split
   key or the label ([`roles.md`](06-cross-pool.md), §3). Metadata inverts under an archive holdout.
3. **Above 8 kHz is not available** — ten of fourteen sources have nothing there to begin with.
4. **Do not high-pass.** Five of seven vocoder signatures live below 72 Hz
   ([02 B1b](02-pool-b-fake-voice.md)). ⚠️ But `cfad/pwg` is identifiable at AUC 0.998 from
   `dc_offset` alone ([02 B6b](02-pool-b-fake-voice.md)) — DC removal and a high-pass pull in
   opposite directions and have to be decided together.
5. **`artifact_family` is measured, not named** — `melgan` and `multi_band_melgan` share a name and
   have opposite artefacts (r = −0.186).
6. **Level is a publisher fingerprint** and is *not* neutralised by the render chain — 22.6 dB of
   median-RMS spread.

And the largest unexploited asset: **45.3 M of the 56.0 M measurements are the `[128]`-wide LTAS and
mel-moment vectors**, and no analysis has read them beyond the duplicate sweep. If you rank
candidate features, rank them **under a `source_name` holdout** (§5.2) — that is the test that killed
every metadata feature and the one duration survived.
