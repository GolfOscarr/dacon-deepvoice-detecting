# 04 — Verification report: the strategy-v2 pipeline against §7 of the spec

*Re-measured 2026-09-24 on the built `strategy-v2` artifacts (manifest 351,306 rows, 4 folds,
the 16 kHz decode cache at 301,226 files), branch `feat/eda`, after the remediation of
[05](05-review-findings.md) under [06](06-remediation-plan.md). Every number below was produced
by a script in this repository that can be re-run; the command is given with each item. The
first measurement of this report (strategy-v1, before the review) is in the git history at
`ee23f2f`; where it matters the earlier number is quoted for the difference.*

| # | item | status | where |
|---|---|---|---|
| 1 | processing audit at n = 20,000 over the fold-0 train view | ✅ 19 / 19 | §1 |
| 2 | draw features < 0.60 except the D-2 residual; edge exposure < 1 %; joins at chance | ✅ | §1 |
| 3 | acoustic residues after SHIP: `dc_offset` and `near_nyquist` gone; every residue family < 0.60 under the source holdout | ✅ every head at chance (0.48–0.52) | §3 |
| 4 | G-EDA6 at full pool-C coverage; G-EDA7 ledger; G4 table; G7 pack | ✅ written · G-EDA7 **passes** at the 2.5 s floor (voice gap 0.050) | §4 |
| 5 | VG1 A1–A7 on `folds.parquet`; G-EDA3 / G-EDA4 | ✅ VG1, ✅ G-EDA4 (18,392 pair sets, CFAD bound) · G-EDA3 fails on music/fake (5 families, the D-22 caveat) | §5 |
| 6 | I10 across processes, I13, I14 on real samples | ✅ 64/64 · 64/64 · gap 0.0, and bitwise across torch thread counts | §6 |
| 7 | P0-a contract probe returns exactly 0.5000 with the shipped chain in `script.py` | ✅ `script.py --probe` → 0.5000 through `processing.infer` | §7 |

---

## 1 · The audit and the draw features (items 1, 2)

`processing.audit.run_audit` over `apply_folds(manifest, folds, 0)`, slice `train`, fold 0,
n = 20,000, seed 0 (the raw manifest carries no fold and the sampler refuses it, 05 A7):

| check | value | gate |
|---|---|---|
| I1 transform-name independence | worst gap 0.008 (`rir`) | 0.02 |
| I1b metadata AUC | 0.540 music-only, 0.507 pooled | 0.60 |
| I2 / I2c composedness | 0.000 / 0.000 | 0.02 |
| I2b mixedness | at chance | 0.032 |
| I8 positive rates | file .61 · voice .51 · music .51 · presence .69 / .70 | [0.2, 0.8] |
| I21 families | voice 222 · music 5 | ≥ 3 |
| I1c draw AUC, fake heads (17 features incl. `unique_fraction`, `n_files`, `n_buckets`) | file 0.518 · voice 0.550 · music 0.522 | 0.60 / 0.61 |
| I1c draw AUC, presence heads (cell 9 included) | voice 0.577 · music 0.583 | 0.60 |
| I1d edge exposure at 50 ms | 0.000 in every pool | 0.01 |
| I1e edge proximity (reported, not gated) | `music_fake/relpos` 0.744, `voice_fake/relpos` 0.675 | — |

The presence heads sit under their gate by 0.02, driven by `sequential` alone (the D-2
presence-pattern residual the owner kept). I1e says where a tile sits inside its file still
reads the pools' length distributions (05 B8); it is not audible in itself and is reported so
the question stays visible. The audit now also probes each presence label against the transform
names, their parameters and the normalize draw (I1p / I1bp, 05 B10), keeps cell 9 for the
presence heads (05 B9), reports sub-gate cues beyond 4 SE as warnings and reads the strongest
single feature raw and folded (05 B11); the fifteen mutations of the review are tests
(`tests/test_processing_audit.py`), every one caught.

## 2 · The `f8` sweep (D-1)

Unchanged from the first measurement ([f8_sweep_2026-09-24.log](f8_sweep_2026-09-24.log)):
`1.0` 18/18, `0.5` 15/18, `0.0` 13/18. `f8 = 1.0` stays.

## 3 · Acoustic residues on the shipped audio (item 3)

`scripts/strategy/shipped_residues.py --n 6000 --workers 16 --fold 0` — 6,000 specs from the
fold-0 train view, rendered through the cache and the full menu, then `processing.ship`
(downmix → `dc_offset` → band `[0, 7200]`), then the EDA's level and spectral extractors over the
**sample**, and the harness's CV AUC per head under the `source_name` holdout. Two families,
judged apart: **residues** are what the pipeline can manufacture and a test file cannot carry
(the level, a DC offset, the resampler's near-Nyquist shelf); **content** is the spectral shape
and the dynamics — a vocoder's high band, a generator's flatness, music's density — which is
the signal the heads are asked for and is reported, not gated.

**Residues after SHIP, grouped AUC** (gate 0.60): `file_fake` 0.496 · `voice_fake` 0.480 ·
`music_fake` 0.511 (music-only 0.494, mixed 0.522) · `voice_present` 0.507 · `music_present`
0.510. `|dc_offset| ≤ 6.4e-9`, `near_nyquist_ratio ≤ 1.7e-7` on every sample.

What changed since the first measurement, in order, each re-measured: the level target (06 D7)
took the mixed-sums-louder presence cue from 0.67 / 0.78 to chance; bucket tiling then stitched
tracks of different loudness into one real-music slot and the crest factor read `music_fake` at
0.78, which per-tile equalisation (`tile_rms_dbfs`, [03 REN-2](03-processing-and-feature-strategy.md))
took to 0.60; with the level equalised the peak *is* the crest and belongs with the content.
The clip (06 D6) removed the over-range cue (a third of samples exceeded 1.0, 05 A4).

**Content, reported**: `voice_fake` 0.66 pooled and 0.79 in voice-only (`hf_ratio_8k`,
flatness, centroid — CFAD's Griffin-Lim and WaveNet vocoders, which the 2.5 s floor admits many
more of); `music_fake` 0.62 (flatness, centroid, `band_energy_0`); presence 0.79 / 0.75
(crest and centroid: music is denser and brighter than speech). These transfer across sources
because they are real.

## 4 · G-EDA6, G-EDA7, G4, G7 (item 4)

`python -m eda.cli gates` at the 99.97 % pool-C VAD coverage: G-EDA6 reports the 3,206 rows OFF-2
acted on (2,100 reassigned, 1,028 restricted, 23 dropped, 45 kept and recorded).
`scripts/verify_gates.py --manifest-dir …/strategy-v2` writes `gates_report.json`,
`g_eda7_ledger.csv`, `g4_table.csv`, `g7_pack.csv`, `g7_examples.csv`.

**G-EDA7, per role** (05 B13; tolerance 0.10, stated in `processing/gates.py`): the 2.5 s floor
drops 12.2 % of fake voice against 7.1 % of real (gap **0.050**, was 0.126 pooled / 0.187 in
voice at 4.0 s); music 0.000 vs 0.000; one-sided filters (`label_evidence` on 23 degenerate
fake rows, the licence gate on real music) are reported as one-sided, not as a gap. **Pass.**

**G4**: pool B drops 12.2 % of rows (15 h of 303), pool A 7.1 % (3 h of 195) — under the 4.0 s
floor it was 41 % and 23 %. **G7**: fma C → 5 1,032 of 8,000 (12.9 %), musan-music 62 of 660,
fakemusiccaps D → 8 1,006 of 27,605; the ten examples per move are listed for the owner (D15).

## 5 · Folds (item 5)

`scripts/build_folds.py` under 06 D1 / D3 / D4: VG1 A1–A7 pass (`folds.vg1.txt`).

* **G-EDA4 ✅** — 0 of 18,392 multi-row `pair_id` sets and 0 of 39 `dup_group` sets straddle a
  fold or PROBE. The CFAD twins are bound (5,809 pair ids over 39,354 rows; 05 A1) and the LJ
  voice is one atom (12,583 + 101,283 + 291 rows; 05 B2); fma and MUSAN share 15 artist atoms.
* **PROBE's real side** (05 A2): real voice 9.0 h, real music 5.2 h, noise 5.9 h, five atoms
  each; 29.4 h of fake voice and 15.6 h of fake music. The budget counts drawable rows only.
* **VAL hours per fold**, real music 34 / 26 / 16 / 16, noise 5.9 / 4.5 / 2.7 / 2.7, fake
  music 14 / 14 / 15 / 15, fake voice **182 / 62 / 8 / 8** — the LJ atom (LJ real, seven
  WaveFake families, the LJ-voice MLAAD models) is 182 h and indivisible; the caveat names it.
* **G-EDA3** — music/fake 5 atoms (5 families): the D-22 caveat; the fold count is 4 for it.

## 6 · Reproducibility on real samples (item 6)

`scripts/strategy/verify_reproducibility.py --n 64 --fold 0` → `eda/out/_strategy/reproducibility.json`:

* **I10** 64 / 64 identical across two fresh interpreters (38 of 64 with a layer), and the
  augment chain is pinned to one torch thread so the same holds across thread counts
  (`tests/test_processing_render.py`, 05 B14).
* **I13** 64 / 64: a role branch is that role's placement with the tiles of a slot merged
  across their crossfades; the file branch lists every slot's span with its own label.
* **I14** max |solo − in-batch| = 0.0 over the valid prefix, 64 real samples, mono and stereo in
  one batch, lengths 66,666 to 955,816 samples.

## 7 · P0-a (item 7)

`script.py --probe --test-dir …` decodes every file, runs it through `processing.ship` and
writes `output/submission.csv` with every column 0.5; `tests/test_processing_infer.py` scores it
with `metrics.dacon.dacon_score` against random ground truth: **0.5000 exactly**. The same
`script.py` with `model/scored.pt` and `model/processing.json` beside it scores a trained model
behind the chain it was trained behind (`tests/test_train_integration.py` trains a fold through
`scripts/train.py` and scores it this way). A decode failure is a counted fallback row and more
than 1 % of them raises; a constant real prediction is refused (02 §guards).

## 8 · Defects fixed on the way

Every item of [05](05-review-findings.md) §A is fixed; §B and §C are fixed except where the
owner deferred them (D10, D15, the S slices, FEAT-2, MODEL-1). Two more surfaced during the
remediation and are in the code's own comments: bucket tiling *with replacement* let the number
of distinct files per slot read the bucket's size, and bucket sizes read the label (0.997 on a
synthetic corpus); leaving an exhausted bucket for the pool let the number of speakers per slot
read it (0.81); a fixed per-slot file budget let the repetition read the files' lengths. The
rule that holds — without replacement, an exhausted bucket reused round-robin, the pool only
when the bucket holds no eligible file, one bucket per side for music and noise — is measured in
§1.

## 9 · Open for the owner

* **D15** — the G7 examples (`g7_examples.csv`) are listed, not listened to.
* **Korean fake voice** — real voice is domain-capped now (06 D2) but there is still no Korean
  fake voice to learn from; the owner is sourcing it.
* **The LJ atom** — 182 of 288 fake-voice hours are one indivisible fold atom; fold 0's VAL is
  that atom, folds 2–3 validate fake voice on 8 h. A per-fold voice number carries that caveat.
* **Content residues** (§3) are the signal; whether a strong model overfits the CFAD vocoders'
  high band in voice-only is a training question, not a pipeline one.

---

*Scripts: `scripts/verify_gates.py`, `scripts/strategy/shipped_residues.py`,
`scripts/strategy/verify_reproducibility.py`, `scripts/verify_metric.py`, `script.py`. Gates on
this commit: the full `pytest` suite in two halves; `pyflakes` clean; no line over 100 columns.*
