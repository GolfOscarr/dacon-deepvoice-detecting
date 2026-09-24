# 03 — Data processing and feature strategy: the implementation spec

**Revised 2026-09-20.** This is the specification the data-processing pipeline is implemented
from. It is organised as the pipeline runs — one numbered stage per block, every block in the same
shape — with the evidence kept in [02](02-analysis-report.md) and referenced by section number.
Read §1 (the decisions) and §2 (the map) first; §3 is the reference you implement against; §4–§5
are the data and config contracts; §6 is the ordered change list with its tests.

**Status vocabulary**, used on every item:

| mark | meaning |
|---|---|
| **EXISTS** | the code does this today, unchanged |
| **CHANGE** | the code does something here today; a value or rule changes |
| **NEW** | nothing does this today; it must be written |
| **FIXED** | the value is decided by a measurement in [02](02-analysis-report.md) |
| **OPEN** | a placeholder; §8 says who decides and how |

Rules every stage obeys (details in [pipelines/03](../pipelines/03-transforms.md) and
[data/10](../data/10-preprocessing-and-filtering.md)): a step is exactly one of **draw** (sampler;
labels fixed by the cell), **augment** (`(wav, rng) → wav`, cannot see labels, cannot move time),
**preprocess** (`(wav, sr, lengths) → wav`, shipped, identical at test time, batch-invariant) or
**filter** (offline sidecar, never audio); anything that moves audio on the timeline is a draw;
every rate is per sample, never per pool or label (R2).

---

## 1 — Decision register

The decisions, in the order the pipeline meets them. `D-` ids are used throughout.

| id | decision | value | status | evidence |
|---|---|---|---|---|
| D-1 | composition policy `f8` | **1.0** (strict — every mixed sample composed) | FIXED (sweep 2026-09-24: `1.0` 18/18 · `0.5` 15/18 · `0.0` 13/18 at n = 20,000 over the built manifest; presence heads read composedness at 0.635 / 0.672, the fake heads at 0.636 under `f8 = 0`; I2c fails at both lower values) | [02 §4.2](02-analysis-report.md#42-why--derived-not-fitted): only `f8 = 1` balances composedness on the component heads, for any mix |
| D-2 | cell mix | reference `{.060 .130 .060 .135 .155 .125 .125 .095 .115}` | FIXED (owner, 2026-09-24: reference mix kept; revisit only from per-head VAL EER) | [02 §4.3](02-analysis-report.md#43-the-cell-mix-has-the-same-shape-of-gap): 0.60 presence-pattern residual on the component heads; minimax alternative reaches 0.51 |
| D-3 | component take | `U(3, 8)` s, **strictly inside** the file (`edge_margin 0.5` s), random offset, **every role incl. whole-file** — **revised by D-21: `U(2, 3)` s, one take per sample** | FIXED (range fixed with D-5 by the owner, 2026-09-24) | [02 §3](02-analysis-report.md#3--crop-policies): music-only draw AUC 0.993 → 0.497 |
| D-4 | timeline fill | **tile** the take to its span with sigmoid joins, every role | FIXED | [02 §2](02-analysis-report.md#2--baseline-what-rundefaultyaml-draws-today): 52.8 % vs 13.9 % silence was the largest cue |
| D-5 | component duration floor | **2.0 s** (timeline floor stays 4.0 s) — **revised by D-21: 4.0 s**, bound to the take (`floor − 2·margin ≥ take_hi`) | FIXED (owner, 2026-09-24: 4.0 s / `U(2, 3)` kept) | [02 §11](02-analysis-report.md#11--manifest-actions-in-hours): the 4 s floor drops 22.5 % of pool B's hours; 2 s recovers 63.5 h — reachable only with a 1 s take |
| D-6 | silence lead / tail | lead `U(0, 3.0)` s, tail `U(0, 1.0)` s, **every sample** | FIXED (owner, 2026-09-24: tail `U(0, 1.0)` kept, unmeasured) | [02 §5](02-analysis-report.md#5--silence): 0.595 → 0.553; tail unmeasured |
| D-7 | silence trimming | **off** | FIXED | A5b asymmetry (A 7.78 pp vs B 4.12 pp) |
| D-8 | gain jitter | `U(−12, 12)` dB, `p = 1` | FIXED | [02 §6](02-analysis-report.md#6--level): music 0.683 → 0.599, voice 0.586 → 0.579 |
| D-9 | loudness normalisation | **off** | FIXED | [02 §6](02-analysis-report.md#6--level): normalising raises the voice residue to 0.786 |
| D-10 | DC removal | **on**, shipped | FIXED | [02 §7](02-analysis-report.md#7--dc-and-the-high-pass): a generator id (0.994) that inverts across archives (0.392) |
| D-11 | high-pass | **none** (`band_hz[0] = 0`) | FIXED | [02 §7](02-analysis-report.md#7--dc-and-the-high-pass): 40 Hz removes 25–75 % of four vocoders' pair difference |
| D-12 | upper band edge | `band_hz = [0, 7200]` | FIXED (owner, 2026-09-24: `[0, 7200]` kept; `null` stays a training-time ablation) | [02 §8](02-analysis-report.md#8--bandwidth-the-resampler-shelf-and-codec): resampler shelf + SONICS 7.3 kHz rolloff |
| D-13 | test-chain menu | mp3 64–192 / wav / flac; mono / stereo; 8 kHz + μ/A-law at 0.2 | FIXED (menu) | [data/06 A-S3/A-S4](../data/06-augmentation-spec.md); one menu for every cell |
| D-14 | G-EDA6 actions | C → cell 5 (336), D → cell 8 (1,006), E restricted (1,032), 23 degenerate dropped | FIXED | [02 §11](02-analysis-report.md#11--manifest-actions-in-hours) |
| D-15 | `artifact_family` | WaveFake → **3** families by measured correlation; others by generator name | FIXED | [02 §10](02-analysis-report.md#10--families-and-domains) |
| D-16 | `domain_cap` | 500, **also on whole-file rows** | FIXED | [02 §10](02-analysis-report.md#10--families-and-domains): uniform draw confirmed |
| D-17 | `aug_strength` | 1.0 everywhere | FIXED | [02 §9](02-analysis-report.md#9--the-vectors): every source identifiable ≥ 0.98 |
| D-18 | model input | waveform → SSL frontend; **no handcrafted feature reaches the model**; scalars are sidecars | FIXED | [02 §7–§9](02-analysis-report.md#7--dc-and-the-high-pass): every scalar is a fingerprint |
| D-19 | segmentation | `whole_file` first; `tiling 5 s / 2.5 s` fallback | FIXED (order) | [architecture/04 §6.1](../architecture/04-heads-and-pooling.md#61--cross-window-aggregation-and-the-duration-trap) |
| D-20 | Tier-2 input channels | none in v1; ablate one at a time after the first scored model | FIXED (order) | §3 FEAT-2 |
| D-22 | fold count | **4** folds, `probe_share 0.10`, PROBE row budget 2× | FIXED (measured in step 10) | the music head has 5 composable fake families (FakeMusicCaps) + SONICS' 2 whole-file-only; PROBE seals one and every VAL side needs a composable one, so 5 folds is infeasible with a music PROBE. Built: PROBE 20 % of rows, VAL folds 102k / 46k / 40k / 36k rows, music 1–2 families per fold (variance caveat), VG1 A1–A7 pass. the 5-fold alternative was declined by the owner on 2026-09-24 |
| D-21 | one take per sample, never capped by a file | `take` drawn once and shared by every role and row kind; `take_hi ≤ component_floor_s − 2·edge_margin_s` asserted | FIXED (measured in steps 3 and 6) | two cues the harness could not see: a take capped by a short file means more tiles, and pool B is short (39 % of fake-voice DOSS mass under 3 s of interior vs 15 % real) — `voice_fake` I1b 0.652; and a take drawn per role makes the larger of two join counts read as "two components" — `voice_present` 0.68–0.81. With both halves: presence 0.59 (the D-2 residual), `voice_fake` 0.50, `music_fake` 0.53; the capped control 0.755. Replaces the duration-matched weights tried first |

---

## 2 — Pipeline map

```
 OFFLINE (once per corpus version)            owner
 ┌ OFF-1  corpus → manifest.parquet            scripts/build_corpus_manifest.py (NEW)
 │ OFF-2  label-evidence actions (D-14)        same, reads reassignment_worklist.parquet
 │ OFF-3  duplicates → dup_group               same
 │ OFF-4  filters → verdict sidecar            training/registries.py FILTER
 │ OFF-5  folds.parquet                        training/folds.py
 └ OFF-6  decode cache (int16 @16 kHz)         NEW
 TRAINING, per sample, rng = hash(sample_id, epoch, seed)     training/sampler.py
 ┌ DRAW-1 timeline U(4, 60)
 │ DRAW-2 cell ← cell_mix; composed ← f (D-1, D-2)
 │ DRAW-3 components: file (DOSS D-16), take/offset/tiles (D-3, D-4, D-5)
 │ DRAW-4 placement: lead/tail (D-6), sequential, gain ratio
 │ DRAW-5 noise layer (restriction + additive)                NEW
 │ DRAW-6 augment draw → spec.transforms (D-8, …)             NEW
 └ DRAW-7 normalize draw → spec.normalize (D-13)              NEW
 RENDER, per spec                                             training/render.py
 ┌ REN-1  decode slice + resample_poly (cache hit)
 │ REN-2  tile + compose on the canvas (D-4)                  CHANGE
 │ REN-3  augment_chain(spec.transforms)                      EXISTS
 └ REN-4  _normalize(spec.normalize)                          EXISTS
 SHIPPED (submit.zip) — train and test identical              models/, training/registries.py
 ┌ SHIP-1 robust decode  SHIP-2 resample  SHIP-3 downmix
 │ SHIP-4 dc_offset (D-10)  SHIP-5 band_hz (D-11, D-12)  SHIP-6 no loudness stage (D-9)
 └ MODEL-1 frontend  MODEL-2 segmentation + SED heads + pooling (D-19)  → submission.csv
 FEATURES
 ┌ FEAT-1 the input representation (= MODEL-1/2)
 │ FEAT-2 additive channels, ablation-gated (D-20)
 └ FEAT-3 sidecars: VAD, energy, level, fingerprints — never model input (D-18)
 VERIFY  run_audit (I1–I21) + scripts/strategy/stream_harness.py + VG1–VG6
```

Labels are fixed at DRAW-2/3 and nowhere else. `spec.transforms` and `spec.normalize` are applied
by REN-3/REN-4 today but **nothing draws them** — DRAW-6/7 are the missing producers.

---

## 3 — Stage specifications

Every block: **Purpose · In → Out · Rule · Parameters · Targets · Code · Consumed by · Verify ·
Status.**

### OFF-1 · Corpus → manifest

**Purpose.** One validated `manifest.parquet` over the real corpus (14 runnable sources,
382,068 files), with every split key and label filled from the EDA.
**In → Out.** `eda/out/*/files.parquet` + `_shared/*.parquet` → `manifest.parquet`
(`training.manifest.REQUIRED_COLUMNS`, validated).
**Rule.** Per source, §4.1. Labels come from `POOL_LABELS` (component rows) or `CELL_TABLE`
(whole-file rows) and are never typed by hand.
**Parameters.** none.
**Targets.** all sources.
**Code.** `scripts/build_corpus_manifest.py` — **NEW**, successor of `scripts/build_test_corpus.py`
(which builds the 2,177-row smoke corpus with a placeholder pool D). **As built (step 8):** the
logic is `processing/corpus.py` (`assign_keys`, `apply_worklist`, `dup_groups`,
`licence_verdicts`, `verdicts`, `check_rules`); the script is its CLI. Output
`/data/project/private/dacon-corpus/manifests/strategy-v1/{manifest,verdict}.parquet` +
`build_report.json`; paths carry the stage directory (`interim/…`, `raw/…` for `mlaad`) so one
render root — the corpus root — resolves every source. Extra columns beyond `REQUIRED_COLUMNS`:
`noise_has_speech`, `licence_verdict`, `stage`, `reassigned_from`. Build: 30 s.
**Consumed by.** OFF-2…OFF-5, the sampler.
**Verify.** `validate_manifest`; every §4.1 rule as an assertion (a real row with a family, a fake
row without a `domain_key`, a whole-file row with a pool — each must raise); pool coverage
asserted (all five pools non-empty — `validate_manifest` does **not** check this).

### OFF-2 · Label-evidence actions (D-14)

**Purpose.** Rows whose audio disputes their label get the action the EDA decomposed.
**In → Out.** `reassignment_worklist.parquet` (3,206 rows, 5 actions, at 99.97 % pool-C coverage
after step 7) → manifest edits.
**Rule.**

| action | rows | manifest change |
|---|--:|---|
| `reassign_cell` C → 5 | 1,100 (336 before step 7; `fma` 1,038 = 13.0 % of its 8,000, `musan-music` 62) | `row_kind = whole_file`, `cell = 5`, `pool = null`, labels `(1,1,0,0)`, `artifact_family = null` |
| `reassign_cell` D → 8 | 1,006 | `whole_file`, `cell = 8`, labels `(1,1,1,1)`, family kept |
| `restrict_noise` | 1,032 | **NEW column** `noise_has_speech = True`; row stays in pool E (see DRAW-5) |
| `degenerate` | 23 | drop (or `label_confidence = low` if per-tier loss is adopted) |
| `unevidenceable` · `sparse_real` | 45 | keep, `label_confidence = reported` |

**Parameters.** `VOICE_EVIDENCE_RATIO = 0.20` at VAD threshold 0.5; `MIN_JUDGEABLE_S = 4.0`.
**Targets.** pools C, D, E; the C rows are `fma` and `musan-music`.
**Code.** `eda.analyze.reassign` produces the list; the builder applies it.
**Consumed by.** the sampler (cells 5/8 whole-file rows exist only for the `f8` sweep under D-1).
**Verify.** ✅ step 7 done: `eda.cli vad --partition C --source fma` (the content tier over the
6,000 files the draw left out, 27 min, `eda/out/C/vad_extra.parquet`; 3 unreadable files are the
F-S2 finding) → pool-C VAD coverage 8,657 / 8,660; `reassign` and `analyze` re-run. The sampled and
unsampled halves agree (13.7 % vs 12.7 % of `fma` carries speech at ratio ≥ 0.20). Still to do: G7
review pack (10 examples per source); G4 loss table (C now −13.0 % of `fma` rows, D −3.7 % h,
E −5.3 % h).
**Status.** NEW (builder); the list EXISTS at full coverage.

### OFF-3 · Duplicates → `dup_group`

**Purpose.** No recording on both sides of a fold under two ids.
**Rule.** Union of: byte-identical groups (`duplicates.parquet`, 39 groups), WaveFake's duplicated
Common-Voice half (16,283 rows), CompSpoof's 292 shared parent recordings (1,060 files),
RIRS-pointsource ↔ MUSAN if fetched (88 pairs). `content_duplicates.parquet` (868 LTAS pairs) is a
**shortlist only** — confirm by filename or waveform cross-correlation before adding.
**Code.** builder; `folds.grouping_atoms` unions on `dup_group`.
**Verify.** VG1 A5; G-EDA5.
**Status.** NEW in the builder.

### OFF-4 · Filters (verdict sidecar)

| filter | registry | parameter | applied |
|---|---|---|---|
| `corruption` | `FILTER "corruption"` | — | yes (3 unreadable pool-C files) |
| `usable_duration` | `FILTER "usable_duration"` | `min_seconds = 2.0` (**CHANGE** from 4.0; the timeline floor stays 4.0 in `duration_range`) | yes — D-5 |
| `label_evidence` | `FILTER "label_evidence"` | `min_component_snr_db`, `threshold_version` | **no** — needs a component-SNR proxy and a G3 pack; nothing measured asks for it |
| noise / bitrate / clipping | — | — | **never** (P1: the target domain) |
| licence | `_licences/fma_allow.csv` | — | 23 DENY rows dropped; ND kept; regenerate only when pool-C volume binds |

**Verify.** rate per label of every applied filter into the G-EDA7 ledger (the 2 s floor still
drops more of pool B than A — recorded, not hidden).

### OFF-5 · Folds

**Purpose.** `folds.parquet` once, `scheme_version` bumped.
**Rule.** grouped k-fold over `artifact_family` with PROBE sealed; `source_name`, `speaker_ref_id`,
`pair_id`, `dup_group` disjoint.
**Parameters.** `n_folds 5`, `probe_share 0.10`, `caveat_families_per_val_fold 2`,
`require_component_coverage true`.
**Targets / consequences.** Voice: CFAD 11 + WaveFake 3 + MLAAD ≥ 100 families → as designed.
Music under D-1: FakeMusicCaps' **5** families < 8 floor → **leave-one-family-out, no music PROBE**,
variance caveat on every music row. Real music has two publishers → record `fma`'s `source_name`
at **artist** granularity (156 atoms) so every VAL side has real music.
**SHADOW.** S-a re-renders of VAL: telephone chain, mp3 64/96, stereo, component SNR −15…−3 dB,
4 s and 60 s. S-b slices: Korean (`zeroth` vs MLAAD `ko`), sung (SONICS whole files — their only
scoring under D-1).
**Code.** `training.folds.build_folds`, `configs/run_v1.yaml` `folds:` — **as built:
`processing/splits.py` + `scripts/build_folds.py`, the `folds:` section of `configs/processing_v1.yaml`,
outputs beside the manifest (`folds.parquet`, `folds.caveats.txt`, `folds.vg1.txt`,
`folds_report.json`).** Three fixes to `training.folds._seal_probe` were needed on the real corpus
(step 10): composable groups are sealed before whole-file-only ones (SONICS advanced the music
target while giving PROBE nothing it could draw from); a seal never leaves a head below `n_folds`
*composable* rotating families (the D→8 rows make every FakeMusicCaps family a voice family too, and
all seven music families were sealed for the voice target); a family-advancing seal has a row
budget (`PROBE_ROW_BUDGET = 2` × `probe_share` of the rows — the LJSpeech pair atom, 32 % of the
corpus, was sealed forever by family count). `source_name` is recorded at the publisher's atom
(family / sub-corpus / speaker / artist) — at corpus granularity the manifest was 13 atoms.
S-b (Korean, sung) is **OPEN**: a shadow row is invisible outside its fold, and holding all Korean
out of TRAIN is not what the target domain wants; SONICS rows rotate as ordinary cell-8 rows (never
drawn under `f8 = 1`).
**Verify.** VG1 A1–A7 ✅ (4 folds), G-EDA3, G-EDA4.
**Status.** built (D-22).

### OFF-6 · Decode cache

**Purpose.** Decode once; the crop reads a slice.
**Rule.** `resample_poly` → 16 kHz, native channel count, int16 `.pt` per file, on
`/data/project/private`. The resampler object is the one SHIP-2 uses.
**Size.** 2,676 h = 308 GB mono; SONICS is 1,971 h of it and is not drawn under D-1 → **≈ 81 GB**
for the five component pools (stereo-native sources larger). Cache SONICS only for the `f8` sweep.
**As built (step 11).** `processing/cache.py` + `scripts/build_cache.py`; `render.cache_root` in
`configs/processing_v1.yaml`. `.npy` int16 `(C, n)` per file (not `.pt`: `numpy.load(mmap_mode="r")`
reads the slice without touching the rest), native channels, the resampler is `RenderConfig`'s. The
contract, stated exactly: a cached file is `quantise(resample(full decode))`, and a slice read back is
**bit-identical to that slice of that array** — not to `load_audio` of the slice (int16 < 1 LSB, and
`load_audio` resamples the *slice* while the cache slices the *resample*: they differ at each tile's
edges by the resampler's transient, ~0.02 peak, < 2 % RMS). So a renderer with `cache_root` set reads
the cache and nothing else — a missing file is an error, not a fallback — and one run uses one
regime. Build: 228,801 files (cell 8 skipped), 621 audio-h, ~6 min on 32 threads, ~80 GB; resumable.
**Status.** built.

---

### DRAW-1 · Timeline

`duration_s ~ U(4.0, 60.0)` drawn first (`SamplerConfig.duration_range`). **EXISTS.** ⚠️ The
whole-file branch today sets `duration_s = min(timeline, file)` — **CHANGE**: the timeline is the
timeline for every branch (D-3 applies to whole-file rows too).

### DRAW-2 · Cell and composition policy (D-1, D-2)

`cell ← cell_mix`; `composed ← rng < f[cell]` with `f` from `composed_fractions(cell_mix, f8, …)`.
Under `f8 = 1`: `f5 = f8 = 1`, so cells 5–8 are always composed and whole-file rows are never
drawn. **EXISTS**; values in §5. Verify: I2, I2b, I2c, I8.

### DRAW-3 · Components: file, take, offset, tiles (D-3, D-4, D-5, D-16)

**Purpose.** The music-head fix, applied to every role.
**In → Out.** (role, fake, span) → `ComponentDraw`s.
**Rule.**

```
file    ← weighted draw over the role's rows, w = min(count(domain), cap) / count(domain)
          (DOSS; applies to component AND whole-file rows -- CHANGE)
take    ~ U(take_lo, take_hi)
take     = min(take, file_duration − edge_margin_s, span)
          if file_duration − edge_margin_s < take_lo: take = file_duration − edge_margin_s
          (rows under component_floor_s never reach here: OFF-4)
n_tiles  = ceil(span / take)
for each tile: offset_i ~ U(0, file_duration − take)   # independent per tile
               ComponentDraw(file, role, offset_i, take (last tile: span − (n−1)·take),
                             target_start_s = slot_start + i·take, gain_db)
```

**Parameters.** `take_lo 3.0`, `take_hi 8.0`, `edge_margin_s 0.5`, `component_floor_s 2.0`
(all **NEW** `SamplerConfig` fields); `domain_cap 500`.
⚠️ **Measured in steps 3 and 6 (D-21):** the cap `take ≤ file − 2·margin` makes the tile count
follow the file's length, and a take drawn per role makes the join count follow the presence
pattern — the harness saw neither, because it audited joins only on the component heads and only
on the training sampler's draws. As built, the take is drawn **once per sample**, shared by every
role, and `take_hi ≤ floor − 2·margin` so no file caps it: `take_range_s [2, 3]`,
`component_floor_s 4.0` (OPEN together — 3.0 / [1.5, 2] keeps 92 % of pool B's hours, 4.0 / [2, 3]
80 %, 6.0 / [3, 5] 63 %; every pair passes the audit identically).
**Why these values.** 3–8 s is below pool D's 10 s (so the offset range is never empty) and inside
the segment grid ★ `[BirdCLEF playbook]` D4 recommends; the audit is flat across 2–8 s (P5/P7/P8
within 0.03). `edge_margin 0.5` took onset exposure from 20 % → 0.8 % (voice) and 78 % → 1.9 %
(noise; CompSpoof clips are exactly 4.00 s). Independent offsets per tile keep the join count a
function of `(span, take)` only.
**Targets.** every role, every row kind.
**Code.** `Sampler.sample_spec` (both branches); `SampleSpec` gains nothing — tiles are ordinary
`ComponentDraw`s, so `frame_intervals` stay exact and I13 holds.
**Consumed by.** REN-2.
**Verify.** `P(offset < 10 ms)` per pool equal and < 1 %; music-only draw AUC < 0.60; joins AUC
at chance on every head; mutation: `edge_margin_s → 0` must make the pool-D test fail.
**Status.** CHANGE (rule) + NEW (fields).

### DRAW-4 · Placement: lead/tail, structure, gain ratio (D-6)

| item | rule | parameters | status |
|---|---|---|---|
| lead / tail silence | `lead ~ U(0, silence_lead_s)`, `tail ~ U(0, silence_tail_s)`, capped at half the timeline; components occupy `[lead, duration − tail]`; **drawn for every sample including the whole-file branch** | `silence_lead_s 3.0`, `silence_tail_s 1.0` (OPEN) | CHANGE (values; branch) |
| structure | `sequential` with `p = sequential_prob`, slot = span / n_components; overlap otherwise; sigmoid tapers of `crossfade_ms ~ U(10, 200)` at joints | `sequential_prob 0.25`, `crossfade_ms_range (10, 200)` | EXISTS |
| gain ratio (A-A3) | voice `gain_db ~ clip(N(mean, σ), range)` when both components present | `gain_db_range (−15, 15)`, `gain_db_mean −3.6`, `gain_db_sigma 4.0` | EXISTS |
| silence trimming (P-A2) | **not applied** — the inside-crop (DRAW-3) already removes the file's own lead from > 99 % of samples without an asymmetric filter (D-7) | — | EXISTS (absent) |

**Verify.** effective lead AUC on `voice_fake` < 0.60 under the three A6b controls (harness);
`any_lead_silence_s` at chance on the presence heads (it is drawn before the cell decides roles).

### DRAW-5 · Noise layer — **NEW**

**Purpose.** Pool E as an additive layer under any cell, and the `restrict_noise` rule.
**Rule.** With `p_noise_layer`, draw one pool-E row and add it under the composite at
`snr_db ~ U(10, 30)` (a `ComponentDraw` with `role = "noise"`, its own take/offset/tiles by DRAW-3,
`gain_db` set from the SNR against the composite's RMS at render). If the cell has
`voice_present = 0`, rows with `noise_has_speech = True` are excluded from the draw. Never draw
`musan-music` as noise (it is pool C).
**Parameters.** `p_noise_layer 0.5`, `noise_snr_db_range (10, 30)`.
**Why.** ★ `[BC2026]` SNR 10–30 at p 0.5; the EDA's 1,032 speech-carrying noise files would
mislabel any music-only composite they are mixed into.
**Verify.** I1b on `noise_take`, `noise_snr`; no flagged row under a `voice_present = 0` spec.
**As built (step 9).** The layer is a `ComponentDraw(role="noise", snr_db=…)` — `snr_db` is a
new optional field of the shared spec type — tiled by DRAW-3 with the sample's take; the decision
and the SNR are drawn before the cell (R2), the row after it (the `noise_has_speech` exclusion needs
`voice_present`); under **every** cell including 9 (a cell-9 sample never layers its own file) and
under whole-file rows. The renderer builds the composite first and scales each layer once so its
RMS over its span sits `snr_db` below the composite's; a silent composite skips the layer. A layer
carries no frame target. Measured on the S-tier stream at n = 20 000: every gate passes with the
layer on and off; `voice_present` 0.587 vs 0.590, `voice_fake` 0.532 vs 0.530.

### DRAW-6 · Augment draw → `spec.transforms` — **NEW**

**Purpose.** Produce the label-blind transform list REN-3 already applies.
**Rule.** For each entry of `SamplerConfig.augments` (name, `p`, parameter ranges): with
probability `p`, append `(name, {params drawn from ranges})`. Drawn **before** the cell (so the
draw cannot see it) from the sample's RNG.
**Menu (v1).**

| name | registry | params | `p` | status |
|---|---|---|---|---|
| `gain_jitter` | exists | `db ~ U(−12, 12)` (D-8) | 1.0 | value CHANGE |
| `rawboost_ssi` | exists | `snr_db_range (10, 40)`, `tilt_db_range (−12, 12)` | 0.5 | EXISTS |
| `gaussian_noise` | exists | `snr_db_range (10, 30)` | 0.3 | EXISTS |
| `pink_noise` | **built (step 12)** | `snr_db_range (10, 30)`, 1/f spectrum | 0.3 | built |
| `stereo_imbalance` | exists | `db_range (−4, 4)` | 0.3 (stereo samples) | EXISTS |
| `rir` | **built (step 12)** | RIRS' **218 real responses** (not the 92 isotropic noises — those are pool E), dry/wet `U(0.3, 1.0)`, direct path aligned to 0 so **no delay is declared** (an augment may not declare one) | 0.2 | built |
| SpecAugment / structured dropout | inside the frontend (spectrogram domain) | 2 masks, 15 % F / 20 % T | 0.5 | NEW (later) |
| pitch shift, time stretch, loudness normalisation | — | — | — | **not in v1** (D-9; resynthesis manufactures label noise) |

**Verify.** I1 (names) and I1b (every parameter) over the stream; the registry probe refuses a
length change or an undeclared shift.

### DRAW-7 · Normalize draw → `spec.normalize` (D-13) — **NEW**

**Purpose.** Model what the organizers did to the test set, drawn per sample from **one menu for
every cell**.
**Rule.** keys ⊆ `render.NORMALIZE_KEYS = {container, bitrate, channels, telephone_hz, companding}`.

| key | menu |
|---|---|
| `container` / `bitrate` | `wav` 0.30 · `flac` 0.15 · `mp3` 64 0.15 · 96 0.15 · 128 0.15 · 192 0.10 |
| `channels` | `mono` 0.5 · `stereo` 0.5 |
| `telephone_hz` + `companding` | none 0.80 · `8000` + `ulaw` 0.10 · `8000` + `alaw` 0.05 · `8000` + `null` 0.05 |

**Why.** the test set is MP3/WAV/FLAC, mono and stereo, with 전화채널; MP3 64 cost MusicDET +37 EER
pts; SONICS' native 36.8 kbps says low-rate MP3 is in the target family.
**Caveat.** Stage S3 (`training/stages.py`) expands each spec uniformly over `CODEC_VARIANTS`
(4-way) — keep that for S3; DRAW-7 serves S1/S2. AAC/OPUS/AMR are not wired (encoder delay
unverified) — do not add a container without `_codec_roundtrip`'s length assertion.
**Verify.** I1b on the `normalize` keys; `_codec_roundtrip` length assertion.

---

### REN-1 · Decode and resample — EXISTS

`load_audio(path, offset_s, duration_s)` from the cache (OFF-6) or ffmpeg; `resample_poly` →
16 kHz; raises `DecodeError` on a short or non-finite result (a corpus row that lies about its
duration is a defect, not padding).

### REN-2 · Tile and compose — CHANGE

Each `ComponentDraw` is placed at `target_start_s`; consecutive tiles of one component get
complementary sigmoid tapers over `crossfade_ms` at each join, exactly as sequential segments do
today (`_place` + the taper code in `render.py`). Overlap components sum; the gain ratio is applied
to the voice component. No length change, no time shift.
**Verify.** `render(spec) == render(spec)` (I10); the canvas has no digital-silence run longer
than `max(lead, tail)`; `frame_intervals` union = the placed spans (I13).

### REN-3 · Augment apply — EXISTS (`augment_chain(spec.transforms)`)
### REN-4 · Normalize apply — EXISTS (`_normalize(spec.normalize)`)

---

### SHIP-1…6 · The shipped chain (identical for train and test)

| # | step | code | parameters | decision | verify |
|---|---|---|---|---|---|
| SHIP-1 | robust decode, any container | `render.load_audio` + ffmpeg fallback; per-file try/except at inference with a 0.5 fallback row | `n_fallback / n < 0.01` asserted | P-S1 | P0-a contract probe |
| SHIP-2 | resample | `resample_poly_to` → 16 000 | one kernel | A-S1 | I13 |
| SHIP-3 | channel policy | `AudioConfig.channels` | `downmix` | — | I14 |
| SHIP-4 | DC removal | `PREPROCESS "dc_offset"` (exists; **not in any config today**) | per row over `lengths` | **D-10** | I14; harness `voice_dc_offset` → chance |
| SHIP-5 | band limit | `AudioConfig.band_hz` (rFFT brick wall over `lengths`) | `[0, 7200]` | **D-11, D-12** | harness `music_near_nyquist`, `min_bandwidth_hz` |
| SHIP-6 | loudness | — | **none** | **D-9** | — |
| — | `pre_emphasis` | exists | **off** | nothing measured asks for it | — |

All registered steps have `group_delay = 0`; the same `preprocess_chain` object is vendored into
`script.py`; startup assertions per
[competition/02](../competition/02-submission.md#-guard-against-a-silently-broken-submission).

⚠️ **As built (step 5):** the pipeline is independent of the training loop, so the chain lives in
`processing/ship.py` — `ShipConfig` is the `ship:` section of `configs/processing_v1.yaml`
(`channels`, `preprocess: [dc_offset]`, `band_hz: [0, 7200]`) and `ship()` is the one function
both the loop and `script.py` will call. `render.audio.band_hz` stays `null`: the renderer does not
apply the band, the shipped chain does.

### MODEL-1 · Frontend (= FEAT-1a)

| knob | v1 | why |
|---|---|---|
| frontend | **BEATs**, vendored, MIT | the general-audio model we can ship; candidate A is stage one of B |
| mel (inside the frontend) | 128 bins, 0–8 kHz, patch stride 16 → `fps 6.25` (asserted by the frontend) | fixed by BEATs |
| `layers` | 9 (**OPEN**: measure 6 / 9 / 12) | truncation depth unmeasured; 4 probed XLS-R layers matched the full model at 1.34 M trainable params |
| `adapter` | LoRA `rank 16, alpha 32` on `q_proj, v_proj`; `freeze true` | training compute is free, inference is scarce |
| `freq_pool` | GeM `p_init 3.0`, learnable | learned mean↔max interpolation |
| specialists (candidate B) | voice XLS-R/WavLM-class (licence C1–C3 open); music BEATs/EAT | after A scores |

### MODEL-2 · Segmentation and pooling (= FEAT-1b, D-19)

`segmentation.mode whole_file` first (deletes the cross-window duration bias: measured spread
`max` 1.63 · `topk_mean` 1.18 · `quantile` 1.08 · `mean` 0.004 over 1–12 windows); fallback
`tiling` with `window_seconds 5.0`, `hop_seconds 2.5`, `aggregation topk_mean k 3` or
`confidence_gated`. SED attention head per branch, `clip_weight 1.0`. Output float64 `softsign`,
no rounding, no rank normalisation (rule 2.4, VG5). Low-band ensemble member `band_hz [0, 4000]`
for the telephone slice, combined at training time only.

---

### FEAT-2 · Additive input channels — ablation-gated, none in v1 (D-20)

Admission rule for each: computed per file over `lengths` (rule 2.4); in the shipped chain
(symmetric); adopted only if VAL EER improves under [validation/03](../validation/03-decision-protocol.md)
**and** the harness shows no new acoustic residue ≥ 0.60 under the `source_name` holdout.

| id | channel | target | parameters | consumed as | risk |
|---|---|---|---|---|---|
| F2-1 | delta / delta-delta of the log-mel | all; music first | width 2, on the 128-mel at 100 fps before patching | 3-channel patch conv: BEATs' `nn.Conv2d(1, embed, …)` (`models/frontends.py:328`) re-initialised as `Conv2d(3, …)` with the pretrained kernel in channel 0, or a learned 3→1 projection | none beyond compute |
| F2-2 | PCEN view | mixed cells with quiet voice (ratio to −15 dB) | `s 0.025, α 0.98, δ 2, r 0.5`, per band, per file | second channel, or the voice specialist's sole channel | adaptive gain must never see batch statistics |
| F2-3 | real-class average-PSD whitening | fake heads | PSD over TRAIN-fold pools A + C, chain plane, 1024-pt, Tukey α 0.5; stored as a 513-vector in `model/` | a preprocess step in the rFFT domain (like `bandpass`) | the PSD averages four *publishers* — re-run the harness and VG3 after |
| F2-4 | denoise-residual channel | telephone / noisy slice | deterministic spectral gating (no neural denoiser); `[raw, denoised, raw − denoised]` | 3-channel input | 6-vCPU decode budget |
| F2-5 | band energy / band-SNR, 8 × 1 kHz | mixed cells | 50 ms hop, log energy relative to the file median | 8-d **auxiliary** input to the SED heads only | `band_energy_0` is at chance for real/fake — expect presence-head gains only |
| F2-6 | music structure (beat regularity, repetition) | C, D, cell 8 | beat tracker, onset-envelope autocorrelation | report first; feature only if it survives a `source_name` holdout | Tier C |

### FEAT-3 · Sidecar features — consumed by the pipeline, never by the model (D-18)

| id | feature | tool / parameters | consumed by | status |
|---|---|---|---|---|
| F3-1 | speech evidence | Silero VAD (`models/vendor/silero_vad`, `.jit`, offline), 512-sample chunks at 16 kHz, thresholds **0.5 and 0.4** stored, min span 4 chunks (128 ms), ratio ≥ 0.20 = evidence | OFF-2 reassignment; presence-head sanity on VAL; G-EDA6 | EXISTS; **coverage extension** for `fma` (6,000 files) is the job |
| F3-2 | validity masks (V-A1/V-A2) | VAD probability → boolean at 0.5 → close 100 ms / open 250 ms → min run 2.0 s | region-limited crops (`validity_mask_ref`) | **deferred** — under DRAW-3 a 3–8 s inside-crop of a file with silence ratio 0.15–0.17 rarely lands on silence; build only if error analysis shows silent crops |
| F3-3 | level / timing scalars | `eda/extract/level.py`, `timing.py` (peak, RMS, crest, DC, clipping, silence ratio, lead/tail, longest valid span; `DB_FLOOR −120`, 50 ms hop) | OFF-4 filters, degenerate screen (`−60 dBFS`, `0.01`, `0.99`), G-EDA7 ledger | EXISTS |
| F3-4 | duplicate fingerprints | sha256 (done) → LTAS shortlist (done, weak) → waveform cross-correlation over the 868 shortlisted pairs | OFF-3 | NEW (the xcorr pass) |
| F3-5 | metadata | `files.parquet`'s 30 columns; `column_roles.parquet` assigns each as split key or leakage risk | OFF-1, OFF-5 only. **`script.py` must not read container / bitrate / rate into anything the heads see** | EXISTS |
| F3-6 | the harness's effective features | take, coverage, onset/end exposure, joins, effective lead, level after gain, bandwidth, DC | I1b's feature frame (VERIFY) | NEW in `training/audit.py` |

---

## 4 — Data contracts

### 4.1 Manifest — per-source rules for the split keys (D-15)

| source | pool / cell | `artifact_family` | `speaker_ref_id` | `domain_key` | `pair_id` | `label_confidence` |
|---|---|---|---|---|---|---|
| `cfad-real` | A | null | `group_key` (`<split>/real_clean/<subcorpus>`, 14) | null | `cfad:<SSB id>` where the id exists in `cfad-fake` (7,900) | reported |
| `zeroth-korean` | A | null | `group_key` (115 speakers) | null | null | reported |
| `ljspeech` | A | null | `ljspeech_LJ` (one atom → one fold or PROBE) | null | `lj:<LJ id>` | reported |
| `musan-speech` | A | null | `group_key` | null | null | reported |
| `cfad-fake` | B | `cfad/<vocoder>` (11) | `SSB` speaker prefix | `cfad-fake\|cfad/<vocoder>` | `cfad:<SSB id>` | exact |
| `wavefake` | B | `wf_melgan` = {melgan, melgan_large} · `wf_gan` = {full_band_melgan, hifiGAN, parallel_wavegan, waveglow} · `wf_mb_melgan` = {multi_band_melgan}; JSUT / CV-prompt subsets: `wf_jsut_<vocoder>`, `wf_cv_fastspeech2_pwg` | `ljspeech_LJ` / `jsut` / `common_voice` | `wavefake\|<subset>` | `lj:<LJ id>` for the `ljspeech_*` subsets, only where the LJSpeech twin survives the filters | exact |
| `mlaad` | B | `mlaad/<generator>` (**205** directories measured, not 175); codec > vocoder > backbone merges applied by hand to the 12 `ko` generators first (not done) | `<language>/<generator>` | `mlaad\|<generator>` | null | exact |
| `fma` | C | null | **artist** — the id from `tracks.csv` (**2,309** atoms; the EDA's 156 were `fma_small/NNN/` numbering buckets, not artists) | null | null | reported; `licence_verdict` allow 5,084 / derivatives_barred 2,867 / deny 23 dropped |
| `musan-music` | C | null | `group_key` (82) | null | null | reported |
| `fakemusiccaps` | D | `fakemusiccaps/<generator>` (5) | parent clip (5,521) | `fakemusiccaps\|<generator>` | null | exact |
| `sonics` | cell 8 | `suno_chirp` = {chirp-v2-xxl-alpha, chirp-v3, chirp-v3.5} · `udio` = {udio-30s, udio-120s} | generator | `sonics\|<generator>` | null | exact |
| `compspoof-env-bonafide` | E | null | parent recording (10,710; 292 shared across splits → `dup_group`) | null | null | reported |
| `musan-noise` · `rirs-isotropic-noise` | E | null | `group_key` | null | null | reported |

Columns added to `REQUIRED_COLUMNS`: **`noise_has_speech`** (bool, pool E; NEW). `aug_strength`
= 1.0 everywhere (D-17). `validity_mask_ref` null in v1 (F3-2 deferred).

### 4.2 Sidecars

| file | producer | consumer |
|---|---|---|
| `verdict.parquet` (file_id, filter, verdict, reason, threshold_version) | OFF-4 | OFF-1 (drops) |
| `folds.parquet` | OFF-5 | the sampler |
| `cache/<file_id>.pt` int16 | OFF-6 | REN-1 |
| `eda/out/_shared/reassignment_worklist.parquet` | `eda.cli reassign` | OFF-2 |
| `eda/out/_strategy/*` | the harness | VERIFY |

### 4.3 `SampleSpec` — what changes

Nothing structural. Tiles are extra `ComponentDraw`s; `spec.transforms` and `spec.normalize` are
filled by DRAW-6/7 (today always empty). `SamplerConfig` gains `take_range_s`, `edge_margin_s`,
`component_floor_s`, `p_noise_layer`, `noise_snr_db_range`, `augments`, `normalize_menu` (§5).

---

## 5 — Configuration draft — `configs/run_v1.yaml`

Every field written out, as `run_default.yaml` does. Values marked `# OPEN` are §8's.

```yaml
sampler:
  cell_mix:
    p: {1: 0.060, 2: 0.130, 3: 0.060, 4: 0.135, 5: 0.155,
        6: 0.125, 7: 0.125, 8: 0.095, 9: 0.115}        # D-2, OPEN: sweep vs minimax
  f8: 1.0                                               # D-1  (was 0.0)
  single_composed_rate: 0.0
  noise_composed_rate: 0.0
  balance_marginal_composedness: true
  domain_cap: 500                                       # D-16, now also on whole-file rows
  duration_range: [4.0, 60.0]
  # -- DRAW-3 (NEW fields) --
  take_range_s: [3.0, 8.0]                              # D-3
  edge_margin_s: 0.5                                    # D-3
  component_floor_s: 2.0                                # D-5
  # -- DRAW-4 --
  gain_db_range: [-15.0, 15.0]
  gain_db_mean: -3.6
  gain_db_sigma: 4.0
  sequential_prob: 0.25
  crossfade_ms_range: [10.0, 200.0]
  silence_lead_s: 3.0                                   # D-6  (was 0.0)
  silence_tail_s: 1.0                                   # D-6, OPEN: unmeasured
  # -- DRAW-5 (NEW) --
  p_noise_layer: 0.5
  noise_snr_db_range: [10.0, 30.0]
  # -- DRAW-6 (NEW): name, p, parameter ranges; drawn before the cell --
  augments:
    - {name: gain_jitter,      p: 1.0, db_range: [-12.0, 12.0]}        # D-8
    - {name: rawboost_ssi,     p: 0.5, snr_db_range: [10.0, 40.0], tilt_db_range: [-12.0, 12.0]}
    - {name: gaussian_noise,   p: 0.3, snr_db_range: [10.0, 30.0]}
    - {name: pink_noise,       p: 0.3, snr_db_range: [10.0, 30.0]}     # NEW registry entry
    - {name: stereo_imbalance, p: 0.3, db_range: [-4.0, 4.0]}
  # -- DRAW-7 (NEW): one menu for every cell --
  normalize_menu:
    container: {wav: 0.30, flac: 0.15, mp3_64: 0.15, mp3_96: 0.15, mp3_128: 0.15, mp3_192: 0.10}
    channels:  {mono: 0.5, stereo: 0.5}
    telephone: {none: 0.80, ulaw: 0.10, alaw: 0.05, plain: 0.05}   # all at 8000 Hz
  scheme_version: strategy-v1
  allow_unsound_mix: false

render:
  root: .
  cache_root: /data/project/private/dacon-corpus/cache16k          # OFF-6 (NEW)
  audio:
    sample_rate: 16000
    min_seconds: 4.0
    max_seconds: 60.0
    channels: downmix                                   # SHIP-3
    band_hz: [0, 7200]                                  # D-11 / D-12 (OPEN: ablate vs null)
  preprocess: [dc_offset]                               # SHIP-4, D-10 (NEW section)
  resampler: training.render.resample_poly_to
  crossfade_shape: sigmoid
  check_duration: true

folds:
  n_folds: 5
  probe_share: 0.10
  scheme_version: null
  caveat_families_per_val_fold: 2
  require_component_coverage: true
  allow_no_probe: false                                 # music head: LOFO, caveat in the ledger
  assigned_at: null

loop: {out_dir: runs/v1, n_buckets: 4, ema_decay: 0.999, grad_clip: 5.0,
       device: cuda, checkpoint_every: 0, max_steps: null}
```

Model config deltas (`configs/a_shared_trunk.yaml`): `segmentation: {mode: whole_file}`,
`audio: {band_hz: [0, 7200]}`; low-band member: `band_hz: [0, 4000]`. Everything else unchanged.

---

## 6 — Change list, in build order, each with the test that must be seen to fail

| # | stage | file | change | test (mutation-style) | closes |
|---|---|---|---|---|---|
| 1 | DRAW-3 | `training/sampler.py` | take/offset/tile rule; `SamplerConfig` fields; DOSS on whole-file rows | a 10 s row never gets `offset < 10 ms`; `edge_margin_s → 0` fails it; per-generator whole-file counts ∝ `min(count, cap)` | D-3, D-5, D-16 |
| 2 | REN-2 | `training/render.py` | tapers at every tile joint | `render(spec) == render(spec)`; no silence run > `max(lead, tail)`; I13 | D-4 |
| 3 | DRAW-1/4 | `training/sampler.py` whole-file branch | timeline not capped by the file; lead/tail drawn; same take rule | a whole-file spec has `target_start_s = lead` and `duration_s` from DRAW-1 | D-6 |
| 4 | DRAW-6/7 | `training/sampler.py`, `training/config.py` | augment and normalize draws from `augments` / `normalize_menu`; drawn before the cell | I1 sees the names; I1b sees every parameter and the normalize keys; a label-conditioned draw injected in a test must fail I1b | D-8, D-13 |
| 5 | SHIP-4/5 | `training/config.py`, `configs/*.yaml` | `render.preprocess` section; `band_hz` value | `test_no_config_field_is_silently_ignored`; I14 on the chain | D-10, D-11, D-12 |
| 6 | VERIFY | `training/audit.py` — **as built: `processing/audit.py`** (the training invariants over tile-collapsed specs, plus I1c per head over the harness's draw features and I1d edge exposure per pool) | harness features in `_feature_frame` | H1, H2 and the raw lead reproduced as **failing** I1b on the old sampler; passing on the new | — |
| 7 | OFF-2 | `eda.cli signal` (VAD only) — **as built: `eda.cli vad --partition C --source fma`**, `eda.driver.vad_coverage` (resumable, draw-fingerprinted), `load_signal(with_extra=True)` for `reassign` / G-EDA6 only | `fma` VAD to 100 % | coverage 1.000 — **measured 0.9997** (3 unreadable files) | D-14 |
| 8 | OFF-1…4 | `scripts/build_corpus_manifest.py` (NEW) — **as built: `processing/corpus.py`** | §4.1 rules, D-14 actions, `dup_group`, `noise_has_speech`, verdict sidecar | every §4.1 rule as an assertion (`check_rules`; a missing pool, a real row with a family, a one-sided pair each refuse the build); G-EDA gates re-run | D-14, D-15, D-17 |
| 9 | DRAW-5 | `training/sampler.py`, `training/render.py` — **as built: `processing/sampler.py`, `processing/render.py`, `ComponentDraw.snr_db`** | noise layer; SNR-to-gain at render | no `noise_has_speech` row under a `voice_present = 0` spec (mutation: unflagged rows do go under); the rate per sample equal across labels and cells; the rendered layer at its SNR ± 0.05 dB; I1c on the layer's features | — |
| 10 | OFF-5 | `training/folds.py` config — **as built: `processing/splits.py`, `scripts/build_folds.py`, three `_seal_probe` fixes** | folds, SHADOW re-renders (S-a needs re-rendered files: OPEN) | VG1 A1–A7 ✅; the seal starvation, order and budget each tested | D-22 |
| 11 | OFF-6 | new `scripts/build_cache.py` — **as built: `processing/cache.py`, `.npy`** | int16 cache | a cached slice equals `load_audio` bit-for-bit after resample — **as built: equals the quantised full-file resample's slice bit-for-bit; `load_audio` of the slice differs at the edges by design (see OFF-6)** | — |
| 12 | DRAW-6 | `training/registries.py` | `pink_noise`, later `rir` — **built: both**. `pink_noise` = white noise shaped `1/√f` (−3 dB/octave) at a drawn SNR; `rir` = convolution with one of RIRS' 218 real responses (the `_rir_` files beside its isotropic noises), the direct path aligned to sample 0 (the raw responses put it ~2,100 samples in, and an augment may not move audio), dry/wet `U(0.3, 1.0)`, output rescaled to the input RMS; a synthetic response when no bank is configured, so the registry probe needs no corpus. In the v1 menu at p 0.3 / 0.2 | registry probe (length, `group_delay`) ✅ lag 0 at head and tail; the pink slope 8–10 dB over 3 octaves; SNR ± 0.1 dB; `wet = 0` is the identity; a burst's onset does not move through a real response | — |
| 13 | FEAT-2 | `models/frontends.py` | one channel at a time, after the first scored model | I14; harness residues; promotion protocol | D-20 |

Steps 1–6 are one day and unblock training on the S-tier manifest; 7–11 are the corpus build
(the `fma` decode runs in the background from day one); 12–13 follow the first scored model.

---

## 7 — Acceptance: the strategy is implemented when

**Measured 2026-09-24 on the built artifacts: [04-verification-report.md](04-verification-report.md)** — items 1, 2, 5, 6 pass; 3 passes its first half (the DC and near-Nyquist residues are gone) and fails its second on level; 4 is written and G-EDA7 fires on the duration floor; 7 is measurable only on the metric half until `script.py` exists. The four open issues are in 04 §9.

1. `training.audit.run_audit(n = 20_000)` on `configs/run_v1.yaml` over the built manifest:
   **13 of 13** pass (I2 = I2c = 0.000; I8 inside `[0.2, 0.8]`; I21 ≥ 3 families per fake role).
2. `scripts/strategy/stream_harness.py` on the same stream: every **draw** feature < 0.60 ungrouped
   on every head and stratum, except the documented D-2 residual (≤ 0.61 pooled on the component
   heads); `P(offset < 10 ms)` < 1 % and equal across pools; joins at chance.
3. Acoustic residues after SHIP-4/5: `voice_dc_offset` and `music_near_nyquist` no longer top
   features; every acoustic family < 0.60 under the `source_name` holdout where measurable.
4. G-EDA6 re-run at 100 % pool-C coverage; G-EDA7 ledger written (rate per label of every filter);
   G4 table; G7 pack reviewed.
5. VG1 A1–A7 on `folds.parquet`; G-EDA3/4 measured.
6. `render(spec) == render(spec)` across processes (I10); I13; I14 on the shipped chain.
7. The P0-a contract probe returns exactly 0.5000 with the shipped chain in `script.py`.

---

## 8 — Open decisions (do not resolve silently)

On 2026-09-24 the owner fixed D-2, D-3/D-5, D-6, D-12 and D-22 at the values as built (rows above)
and asked for the D-1 sweep; the sweep (D-1 row) leaves `f8 = 1.0` as the only value that passes the audit. What remains open:

| id | decision | how it gets decided |
|---|---|---|
| OFF-2 | `restrict_noise` as flag vs drop | deferred by the owner; the flag exists (DRAW-5 built), no row dropped |
| S-a / S-b | shadow slices (re-rendered files; Korean, sung) | deferred by the owner |
| MODEL-1 | `layers` 6 / 9 / 12; specialist frontends (licence) | deferred by the owner; ablation after the first scored model |
| FEAT-2 | which channel first | deferred by the owner; F2-1 (delta) is the cheapest; F2-3 (whitening) the most informative |
| — | the test chain itself | X4 blocked at n = 3; every symmetric choice is robustness, not matching |

---

## Appendix A — Evidence index

| number used above | value | where measured |
|---|---|---|
| baseline music-head draw AUC | 0.995 | [02 §2](02-analysis-report.md#2--baseline-what-rundefaultyaml-draws-today) |
| pool-D silence / onset exposure under the current draw | 52.8 % / 87.2 % | [02 §2](02-analysis-report.md#2--baseline-what-rundefaultyaml-draws-today) |
| music-only draw AUC after the inside-crop | 0.497 | [02 §3](02-analysis-report.md#3--crop-policies) |
| onset exposure with / without the 0.5 s margin | 20 % → 0.8 % voice; 78 % → 1.9 % noise | [02 §3](02-analysis-report.md#3--crop-policies) |
| composedness on the component heads at `f8 = 0` / `1` | 0.734 / 0.509 (mixed `music_fake`) | [02 §4](02-analysis-report.md#4---composedness-the-constraint-the-design-has-is-not-the-one-the-component-heads-need) |
| cell-mix presence-pattern residual | 0.60 (reference) / 0.51 (minimax) | [02 §4.3](02-analysis-report.md#43-the-cell-mix-has-the-same-shape-of-gap) |
| leading silence, raw → `U(0, 3)` at `p = 1` | 0.595 → 0.553 | [02 §5](02-analysis-report.md#5--silence) |
| level: none / normalise / jitter ±12 (grouped) | music 0.683 / 0.453 / 0.599 · voice 0.586 / 0.786 / 0.579 | [02 §6](02-analysis-report.md#6--level) |
| DC: ungrouped / grouped / `cfad/gl` | 0.806 / 0.392 / 0.994 | [02 §7](02-analysis-report.md#7--dc-and-the-high-pass) |
| near-Nyquist: resampled vs native vs SONICS | 0.12–0.29 / ≤ 0.09 / 0.000 | [02 §8](02-analysis-report.md#8--bandwidth-the-resampler-shelf-and-codec) |
| source identifiability from LTAS | ≥ 0.980 all; 8 of 12 ≥ 0.995 | [02 §9](02-analysis-report.md#9--the-vectors) |
| WaveFake families | 3 clusters, both planes | [02 §10](02-analysis-report.md#10--families-and-domains) |
| 4 s floor cost / 2 s recovery | B 68.3 h (22.5 %) / 63.5 h | [02 §11](02-analysis-report.md#11--manifest-actions-in-hours) |
| trimming asymmetry | A 7.78 pp vs B 4.12 pp | [EDA/01 A5b](../EDA/01-pool-a-real-voice.md#a5b--what-silence-trimming-would-cost-per-pool) |
| pooling duration spread | max 1.63 · topk 1.18 · quantile 1.08 · mean 0.004 | [architecture/04 §6.1](../architecture/04-heads-and-pooling.md#61--cross-window-aggregation-and-the-duration-trap) |
| handcrafted vs waveform | MFCC 26.67 % vs raw 13.75 % EER | [survey/03](../survey/03-sota-singing-mixed.md) |

*Registry, config and constant names are those in `training/registries.py`, `training/sampler.py`,
`training/render.py`, `training/config.py` and `models/config.py` on 2026-09-20; `NEW` fields are
proposals and their names are the implementer's to keep or rename consistently.*
