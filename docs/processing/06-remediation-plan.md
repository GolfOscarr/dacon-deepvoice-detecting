# 06 — Remediation plan: from the review to a trainable pipeline

*Written 2026-09-24 after the owner's decisions on [05](05-review-findings.md) and the D5–D15
questions. This is the build order, the code structure each change lands in, and the gate each
phase must pass before the next starts. Each phase is one or more commits on `feat/eda`.*

## 0 · The decisions this plan implements

| id | decision (owner, 2026-09-24) |
|---|---|
| D1 | PROBE seal: undrawable whole-file rows leave the budget; the real side gets a floor per role (hours and atoms) |
| D2 | Real voice gets the same DOSS domain cap as fake (source atom, 500); the owner sources Korean fake voice separately |
| D3 | LJ-voice fakes join the `ljspeech` atom; fma and MUSAN artists share one artist atom |
| D4 | Folds balance on role-hours, not rows; the imbalance caveat reads hours |
| D5 | **Bucket tiling**: each tile of a component may come from a different file of the same bucket; a file is eligible for a tile iff it can hold it (`duration ≥ tile + xfade + 2·margin`); take `U(1.5, 2.5)`; margin 0.5; the manifest floor becomes `take_lo + 2·margin = 2.5 s` |
| D6 | Clip to ±1 before the normalize step, as a wav on disk would |
| D7 | Per-sample RMS normalisation of the composite (after the layer, before augments); re-measure the voice cue at sample level |
| D8 | True crossfade: tiles overlap by `crossfade_ms` with equal-power complementary ramps |
| D9 | Repetition: bucket tiling first; a unique-content audit feature; the fixed window only if the feature still reads the label |
| D10 | Channel layout: accept under downmix; record the re-measure condition on FEAT-2 |
| D11 | Pin torch to one thread inside the augment chain; fall back to "same thread count" if it costs too much |
| D12 | Delete `single_composed_rate`, `noise_composed_rate`, `balance_marginal_composedness` |
| D13 | Integrate with `training/` on this branch: renderer and ship at every call site, fold views mandatory, eval mode, frozen val specs, `train.py` on the processing config, val-view audit mode |
| D14 | A minimal `script.py` (decode → ship → constant 0.5) closes P0-a; train and test share one `ship` |
| D15 | The owner listens to the G7 examples |
| G-EDA7 | Per role, tolerance stated in the doc, recorded as the standing exception once the floor is 2.5 s |

## 1 · Contract change: a component is a *slot*, its tiles may be many files

`training/spec.py::ComponentDraw` gains `slot: int = 0`. Every tile of one logical component
carries the same `slot` and `role`; under bucket tiling its `file_id` may differ per tile. Every
consumer that today groups tiles by `(file_id, role)` groups by `(slot, role)` instead:
`processing.render.frame_intervals_for`, `_joints`, `processing.audit.collapse_tiles`,
`draw_features`, `scripts/strategy/verify_reproducibility.check_i13`. The whole-file invariant in
`SampleSpec.__post_init__` becomes "one slot, one role" (a whole-file row is still one file).
`SampleSpec.from_dict` coerces list-valued transform params back to tuples so a JSON round trip is
equal (review C5). Default `slot = 0` keeps every existing spec and test valid.

With D8 a tile's `duration_s` is `tile + xfade` (the last tile of a slot: `tile`), and consecutive
tiles start `tile` apart, so they overlap by `xfade`. `frame_intervals_for` merges overlapping
same-slot spans. The renderer's placement invariants change from "abutting" to "overlapping by
exactly `xfade` samples"; the sample-exact placement test and the joint tests are rewritten, not
loosened.

## 2 · Phases, in order

### P1 — corpus builder (`processing/corpus.py`, `processing/gates.py`) → rebuild `manifest.parquet`

* Fix the cfad-real pair regex to `/(SSB\d+)_aishell3\.` (A1) and assert ≥ 7,000 pairs in `check_rules`.
* CFAD speaker ids from the file name (`SSB\d{4}` → aishell3 speaker; `BAC009(S\d{4})` → aishell1
  speaker) into `speaker_ref_id` on both sides; the pair then binds twins and A3 sees speakers.
* D3 atoms: `speaker_ref_id = "ljspeech"` for `ljspeech`, every WaveFake LJ family,
  `wf_cv_fastspeech2_pwg` (LJ-trained) and the ten `mlaad/tts_models_en_ljspeech_*` families;
  `speaker_ref_id = "artist:<normalised name>"` for fma and MUSAN-music rows so shared artists unite.
* Floor: `BuildInputs.component_floor_s` becomes `take_lo + 2·margin` read from the draw config
  (2.5 s); `usable_duration` verdicts re-judged.
* G-EDA7 per role with `G_EDA7_TOL` documented; a side with no rows reports `n = 0`, not rate 0;
  G7's denominator is the post-drop count; `_both_sides` and module docstrings corrected.
* Gate: `tests/test_processing_corpus.py` (+ regex, speaker, atom tests); rebuild; `check_rules`;
  `scripts/verify_gates.py` shows G-EDA4 pass with ≥ 7,000 cfad pairs bound.

### P2 — folds (`training/folds.py`, `processing/splits.py`) → rebuild `folds.parquet`

* `FoldConfig` gains `probe_budget_excludes: ("whole_file",)` (rows the sampler never draws
  under `f8 = 1` leave the row budget), `probe_min_real_hours: 5.0` and `probe_min_real_atoms: 5`
  per role (the seal keeps taking real groups until both hold), and `balance_on: "hours"`.
* Fold assignment target and the imbalance caveat read role-hours; a fake-voice or fake-music
  VAL side under `caveat_min_role_hours` gets a caveat naming the fold.
* Gate: `tests/test_folds.py` (+ budget, real floor, hours balance); rebuild; `folds.vg1.txt`
  A1–A7; `verify_gates.py` G-EDA3/4; the fake-voice hours per VAL fold printed in `folds_report.json`.

### P3 — cache → extend `cache16k` with the rows the 2.5 s floor admits (resumable build).

### P4 — draw (`processing/config.py`, `processing/sampler.py`, `processing/audit.py`)

* `DrawConfig`: `take_range_s = (1.5, 2.5)`, `component_floor_s = 2.5`, the D-21 assertion becomes
  `component_floor_s ≥ take_lo + 2·edge_margin_s`; delete the three dead knobs (D12); validate
  ordered ranges, non-negative sigma, finite SNR, `duration_range` inside the render limits;
  `DrawConfig.for_eval()` returns the config with `augments = ()` (normalize menu kept: it models
  the test chain).
* Buckets: voice → `speaker_ref_id` (fallback `source_name`); real music → pool C; fake music →
  `artifact_family`; noise → pool E (under a voice-absent cell, minus `noise_has_speech`, for the
  **primary** as well as the layer — A3). The anchor file is drawn by DOSS weights as today; each
  tile then draws uniformly among the bucket's files eligible for that tile; an empty eligible set
  falls back to the (pool, domain) eligible set and is counted.
* Real rows get a `domain_key` = source atom and the same cap (D2). Rows sorted by `file_id` in
  `__init__` (B16). RawBoost `snr_db` / `tilt_db`, `stereo_imbalance.db` and the RIR file index
  are drawn at spec time as scalars (B12; the augments accept the scalar).
* Fail closed: `Sampler` raises unless the frame's `slice` is one of `train / val / probe` **and**
  `fold` is set on every row — the raw manifest (slice `train_val`, fold NA) is refused; callers go
  through `training.folds.apply_folds` (A7). `stream_harness.py`, `shipped_residues.py` and
  `verify_reproducibility.py` take `--fold`.
* Audit: group by slot; new draw features per role — `unique_fraction` (union of source windows /
  span), `n_files`, `edge_1s_exposed`, `relpos` (tile offset / file duration); `I1d` reports 50 ms
  and 1 s.
* Gate: `tests/test_processing_sampler.py` rewritten for buckets (every tile eligible, take shared
  and uncapped, joins `= ceil(span/tile) − 1`, cell-9 primary never flagged, real cap applied,
  order-invariance, eval mode); `run_audit` at n = 20,000 on the fold-0 train view: 18/18 plus the
  new features under the gate, `unique_fraction` AUC reported per head.

### P5 — render (`processing/render.py`, `training/registries.py`, `training/spec.py`)

* D8 overlap crossfade with equal-power ramps (`sqrt` of the sigmoid pair so `a² + b² = 1`);
  `_joints` by slot. Layer RMS measured after tapering (B5). D7: composite RMS-normalised to
  `target_rms_dbfs` (config, −23 dBFS) after the layer, before augments; D6: `clip(−1, 1)` before
  `_normalize`. `spec_rng(..., domain="render")` for the augment chain (B6). Threads pinned to 1
  around the augment chain and restored (D11), with the cost measured. `render()` takes a
  `ManifestIndex` and documents that a DataFrame is coerced once per call (C7).
* `training.render.render` gains a guard: a spec with `snr_db` or `slot > 0` raises "use
  processing.render" (C1 tripwire).
* Gate: `tests/test_processing_render.py` rewritten for overlap (ramps sum to unit power at every
  joint, no zero run at a joint, placement exact, frame intervals merged); `shipped_residues.py`
  on the fold-0 train view: clip rate per label, level AUC re-measured (the D7 decision point:
  voice-only `voice_fake` under 0.60 grouped, presence heads reported), DC and near-Nyquist still
  gone; `verify_reproducibility.py` 64/64 at 1 and 16 threads.

### P6 — audit gates (`processing/audit.py`, `training/audit.py`)

* Cell 9 included in the presence-head probes (B9); presence heads probed against transform names
  and parameters and normalize keys (B10; I1 and I1b per presence label). A `warn` tier when
  `AUC − 0.5 > 4·SE` under the 0.60 fail gate, printed in the detail (B11); a univariate
  `|x − median|` AUC beside the linear probe (bimodal cues). `run_audit(..., view="val")`: I21 floor
  from the fold caveat, I1 tolerance `max(0.02, 4·SE)` (C6).
* Gate: the mutation set from the review (15 mutations) as tests: every one caught.

### P7 — acceptance re-run → `04` updated (audit on fold views, residues, reproducibility, gates).

### P8 — integration (D13, D14)

Structure: `training/` keeps owning the loop, collate, stages, checkpoints and validation;
`processing/` owns spec drawing, rendering and the shipped chain. The loop reaches processing
through three seams only, each with a tripwire:

1. **Render**: `training/dataset.py`, `training/loop.py`, `training/validate.py` import `render`,
   `ManifestIndex`, `RenderConfig` from `processing.render` (the old renderer raises on a
   processing spec, P5). `SpecDataset.audit` uses `processing.audit.audit_specs` (collapsed tiles).
2. **Ship**: `processing.ship.ship(batch["wav"], cfg.ship, batch["lengths"])` replaces
   `prepare_waveform` in the loop and in `evaluate`; `ShipConfig` is stored on the model checkpoint
   so inference reuses it; a load-time check asserts `render.audio.sample_rate == ship.sample_rate
   == model.cfg.audio.sample_rate` and `model.cfg.audio.band_hz is None`.
3. **Config and views**: `scripts/train.py` takes `--processing configs/processing_v1.yaml`
   (draw / render / ship / folds) and `--run` (loop / training only; `load_run_config` rejects
   draw and ship sections, so the run file loses `sampler` and `render`), `--manifest-dir` and
   `--corpus-root` separately; every Sampler is built on `apply_folds(manifest, folds, k)`;
   val specs come from `Sampler(view, cfg.draw.for_eval(), slice_="val", fold=k)` and are written
   to `fold<k>/val_specs.json` (round-trip-equal after the `from_dict` fix).
4. **Test time** (D14): `processing/infer.py` — `load_audio` → `ship` → model, batch by length,
   0.5 fallback row on a decode error with the `n_fallback / n < 0.01` assertion; `script.py` at
   the repo root calls it with a constant model to return the P0-a probe.
* Gate: `tests/test_train_integration.py` — a fold trained for a few steps on the synthetic corpus
  through `train.py` with the processing config, the val report produced, ship applied exactly
  once (mutation: a second ship changes nothing, `prepare_waveform` removed), the old render path
  unreachable; `script.py` on a directory of synthetic files scores 0.5000 through
  `scripts/verify_metric.py`.

### P9 — PR to `main`.

## 3 · What is deliberately not in this plan

Korean fake voice (the owner sources it), the S-a / S-b shadow slices, FEAT-2 channels (D10's
re-measure condition recorded there), the training renderer's own canvas rule, and the G7
listening (D15).
