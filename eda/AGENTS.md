# `eda/` — measuring what is actually in the corpus

The plan is [`docs/EDA/`](../docs/EDA/README.md). This package is **Phase 0** of it --
everything answerable without decoding a single sample -- plus **Phase 1**, the S tier, which
decodes.

```bash
V=/data/project/private/dacon-venvs/dacon311/bin/python

$V -m eda sources                    # what the config declares, and whether it is on disk
$V -m eda probe --partition A        # M tier: ffprobe + sha256 over 100%, resumable
$V -m eda consolidate                # parts -> <partition>/files.parquet
$V -m eda sample --partition E       # the seeded draw -> <partition>/sample.json
$V -m eda signal --partition E       # S tier: decode the draw on both planes, resumable
$V -m eda analyze                    # X1 + E1 + grouping, then the gates; exit 1 on any fail
$V -m eda gates                      # re-read the saved analyses and re-run the gates alone
```

🔴 `sample` is its own verb, before `signal`, because **the draw is part of the corpus
definition** (docs/EDA/00 §3). It decides every S-tier number anyone will quote, so it is
recorded to `sample.json` before anything decodes and `signal` refuses to improvise one.
Redrawing takes `--redraw`: it makes every S-tier number published from the old draw
unreproducible.

Config is [`configs/eda.yaml`](../configs/eda.yaml). Unknown keys are an error — the rule and its
implementation are `models.config`'s, imported rather than restated, exactly as `training/config.py`
does.

---

## What Phase 0 answers

| Task | Module | Doc |
|---|---|---|
| **A1/B3/C6/D2** native-format census | `extract/metadata.py` | [EDA/01](../docs/EDA/01-pool-a-real-voice.md) |
| **E1 pass 1** byte-identical duplicates | `analyze/duplicates.py` | [EDA/05](../docs/EDA/05-pool-e-noise.md) |
| **X1** the shortcut audit | `analyze/shortcut.py` | [EDA/06](../docs/EDA/06-cross-pool.md) |
| **X6/A3/B2/C3/E4** grouping atoms | `analyze/grouping.py` | [EDA/06](../docs/EDA/06-cross-pool.md) |
| **G-EDA1..7** | `gates.py` | [EDA/07](../docs/EDA/07-order-and-gates.md) |

### 🔴 Two label sources, and `row_kind` picks between them

`validate_manifest`'s invariant, mirrored in `SourceSpec.__post_init__`: a **component** source is
drawn and composed, so it carries a `pool` and no `cell`; a **whole_file** source is used as-is, so
it carries a `cell` and *"pool = null -- a whole file is used as-is, not drawn from a component
pool"*. `analyze/shortcut.head_labels` reads `POOL_LABELS` for the first and `CELL_TABLE` for the
second. Because a whole-file source has no pool to partition by, the on-disk directory is a
**partition**: `A`-`E`, `mixed`, or `cell8`. (Deliberately not "group" — `folds.grouping_atoms`
owns that word.)

SONICS is why this exists. It was acquired as `pool: D`; its own `fake_songs.csv` reports
`no_vocal = False` for **all 49,074** rows, and `POOL_LABELS["D"]` asserts `voice_present = 0`. As a
pool-D component it would have told the model there is no voice in 49k AI songs that all sing, and
routed sung vocals into the 0.27-weight music-fake head. It is `whole_file`, `cell: 8`.

`ids.py` holds the three things more than one module needs — `file_id` construction, its
decomposition, and path-component matching. All three were written twice before it existed, and one
of them was wrong both times.

Not here, and deliberately: the S tier (`planes.py`, level/timing/spectral/content extractors), the
manifest projection, and every analysis that needs a decode. `gates.py` reports **`na`** for the
gates those would answer, with the reason.

---

## The four contracts

**1. An extractor declares the columns it emits, and the declaration is enforced on every call.**
Not in a test — in `Extractor.__call__`. A column computed and not declared cannot reach the table;
a column declared and not computed raises. This is aimed at one defect class this repo keeps paying
for: 19 silently-ignored config fields, a guard satisfied by a textual mention, and `dup_group`
computed and hardcoded to `None` — the last of which put 88 byte-identical recordings on both sides
of a fold.

**2. A signal extractor cannot see which plane it is on.** The bound callable takes exactly
`(wav, sample_rate)` and registration refuses a function that could take a third argument, the way
`training/registries.py` refuses an augment that could receive labels. R1 — the native/chain split —
is a property of the driver, so no extractor has to remember it.

🔷 Worth knowing when the S tier lands: the native plane needs **no bypass**.
`training.render.resample_poly_to` returns its input unchanged when the rates match
(`training/render.py:139`), so `load_audio(path, sample_rate=orig_sr)` is the undisturbed file
through the *same* decode path — same ffmpeg fallback, same `DecodeError`, same everything.

**3. A failure is a row, not an exception.** An unreadable file gets `probe_ok = False` and the
error text. A file that vanishes from a census makes the census wrong in the direction that hides
problems, and corruption is a finding (F-S2), not an error. Nothing in this package drops a row —
R2: a filter is a distribution-shift decision, and Phase 0 makes none.

**4. Gates are a tri-state, and `fail > na > pass`.** `training.validate`'s `quotable` ordering,
restated. A gate whose input was never computed reports `na`, never `pass`. VG1 A10 reported SKIP on
every run because `scheme_version` was null, and it read as fine; `quotable` was earnable entirely by
SKIPs until the tri-state landed. `na` does not fail the run, but it never reads as a pass either.

⚠️ And `na` has to mean *not run*, not *found nothing*. Three separate paths could confuse the
two, and all three are now explicit: a clean duplicate sweep (the summary is written
unconditionally), a probe that skipped `identity` (`NoHashes`, since `ProbeConfig.extractors` can
narrow the M tier), and a pool that was declared but never consolidated
(`load_files(...).attrs["missing_pools"]`). The duplicate sweep wrote its detail table
only when it found something, so a clean corpus left no artifact and a later `eda gates` reported
`na` for what was a pass. The summary is now written unconditionally — see `DUPLICATE_SUMMARY` in
`cli.py`.

---

## Two structural decisions

**Three tables, not one** ([EDA/00 §1](../docs/EDA/00-harness.md)). `files.parquet` is M tier, 100%,
scalars, and the manifest projects from it. `signal.parquet` + `vectors.npz` are S tier and sampled.
One table would force a choice between 1.7 M rows of mostly-null vector columns and silently
sampling the thing X1 needs the *population* of.

**MUSAN and RIRS are declared as one source per partition**, not one per archive —
`musan-speech` → A, `musan-music` → C, `musan-noise` → E. `scripts/sources.yaml` marks
"never feed MUSAN's music partition in as noise" as Critical; splitting it in the config makes that
structural instead of a rule somebody has to remember ([EDA/03 C7](../docs/EDA/03-pool-c-real-instrumental.md)).
`rirs-pointsource` and `rirs-isotropic` were split for a different reason: the first is MUSAN's
`free-sound` redistributed and the duplicate sweep should be able to *name* the pair. It named it —
843 of 843 files — and **both are now `blocked:`**, the first as a duplicate and the second as
impulse responses rather than noise ([EDA/05 E0](../docs/EDA/05-pool-e-noise.md)). Splitting them
is what made the reasons sayable in one line each; a combined `rirs-noises` would have had to
block or keep both together.

---

## Verification

```bash
$V -m pytest tests/test_eda.py -o addopts=""         # 62 tests, no decode
$V -m pytest tests/test_eda_signal.py -o addopts="" # 31 tests, the S tier
```

Every invariant is paired with a mutation that was **observed** to fail — **46 run, 46 killed** (`scripts/mutate_eda.py`):
the declared-columns check, the arity check, the `fail > na > pass` ordering, null labels coerced to
0, the `DONE`-marker and `FIXED_COLUMNS` checks in `consolidate`, the exclusion in
`enumerate_source` and its component-wise matcher, the `file_id` type guard, duplicate detection,
the `NoHashes` guard, the duplicate-summary completeness check and its unconditional write,
missing-pool reporting, `HEADS` derived from the manifest's label order, `top_k` reaching the audit,
the CV clamp, the empty-sources refusal, the registration side-effect imports `eda.driver` depends
on, and the six that guard the component/whole-file split — both `pool`/`cell` exclusivity rules,
the cells-6-and-7 refusal, `head_labels` reading `CELL_TABLE`, partitioning by cell, and the shipped
config registering SONICS as cell 8 rather than pool D.

⚠️ One of those was a **false kill** at first: `test_the_cli_records_a_clean_sweep` failed with the
mutation *and* without it, because it asserted on the verb's exit code while the fixture's flat
`src-c` directory trips G-EDA3 at any floor. A mutant that dies against an already-red test is not
evidence of anything. Check that a test passes clean before counting its kill.

⚠️ **A green run is not a clean corpus.** On a fixture with a planted format confound and a planted
cross-source duplicate, `analyze` returned `G-EDA2 fail` (AUC 1.000 on `voice_fake`, carried by
`bit_rate` / `file_bytes` / `orig_sr`) and `G-EDA5 fail`, and exited 1. That is the tool working.
Nothing here has yet run on the real corpus — **nothing in S3 is extracted**, and
`scripts/fetch_from_s3.py --extract` is the prerequisite.
