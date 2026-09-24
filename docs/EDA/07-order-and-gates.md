# 07 — Execution Order, Gates, and What Each Finding Changes

---

## 1 — Phase order

Ordered by **cost per answer**, not by pool. Everything in Phase 0 runs without decoding a single
sample, and three of its four items are blocking.

### Phase 0 — no decode (hours, not days)

| Task | Pool | Why first |
|---|---|---|
| **X4** dummy-file forensics | — | 3 files. Unblocks `normalize`, which is a no-op today |
| **D1** archive inventory + real-half exclusion | D | A **licence** gate. Must precede ingestion, not follow it |
| **C1** licence-allowlist intersection | C | Decides whether pool C is even large enough |
| **A1 / B3 / C6 / D2** metadata census (`ffprobe`, 100%) | all | Feeds X1; the population statement |
| **E1** duplicate sweep (sha256) | all | Has already cost one corpus rebuild |
| **A3 / B2 / C3 / E4** grouping-atom inventories | all | Irreversible if recorded late |
| **X6** metadata role table | — | Freezes the split keys before any fold exists |
| **X1** shortcut audit, metadata-only | — | The gate. Runs on Phase 0's output alone |

🔴 **Phase 0 answers the two questions that can invalidate the corpus** — can we legally use it,
and is the label predictable from the archive it came from — before a single GPU-hour is spent.

### Phase 1 — decode, stratified ([00 §3](00-harness.md))

`A2 A5 A6 A7 A8` · `B1 B4 B5` · `C2 C5` · `D3 D6` (100%) · `E2 E3 E5` · `E1` fingerprint pass.

**B1 first.** The WaveFake↔LJSpeech join is the only unconfounded real/fake comparison in the
corpus, it needs no model, and it produces the artifact gallery the 2nd-stage report wants anyway.

### Phase 2 — derived and comparative

`C4 ↔ D5` genre matching · `B6` separability screen · `D4` fold-count shortfall ·
`X2` metadata-leak read · `X3` adversarial validation · `X5` composition audit ·
`X7` the data memo.

### Phase 3 — needs a trained model (out of scope here)

E-A1 proper, E-A9 proper, E-B6 component-SNR-vs-accuracy, E-C4 error analysis. Listed only so the
boundary is explicit: **everything in Phases 0–2 is answerable with `ffprobe`, an STFT, a VAD and a
logistic regression.**

---

## 2 — Gates

Pre-committed, so a result cannot be re-read favourably after the fact.

| Gate | Criterion | On failure |
|---|---|---|
| **G-EDA1** | **Licence.** Every pool-C row's track id is in the allow list; no pool-D row resolves under a MusicCaps real-audio directory | Stop. Remove the rows. This is not a quality issue |
| **G-EDA2** | **Shortcut.** Metadata-only AUC **< 0.60** per head *and* per cell (X1) | Neutralize with a symmetric transform, re-measure. Do not train first |
| **G-EDA3** | **Groups.** ≥6 independent grouping atoms per role (speaker / artist / generator-family / noise-source) | Fold count drops, and the caveat travels into the ledger row |
| **G-EDA4** | **Pairs.** No `pair_id` or `dup_group` set straddles a fold boundary | Rebuild folds. Verify on the built table, not on the intent |
| **G-EDA5** | **Duplicates.** Zero shared sha256 *and* zero near-duplicate pairs across TRAIN/VAL in any fold | Link with `dup_group` and rebuild |
| **G-EDA6** | **Evidence.** Every row's asserted components are evidenced (VAD/PANNs/energy), or carry a validity mask, or are dropped with a recorded reason | Reassign (F-A1) before dropping |
| **G-EDA7** | **Symmetry.** Every filter's *rate* is within a stated tolerance across labels (R2) | The filter is manufacturing a cue. Widen or drop it |

⚠️ **G-EDA2 and G-EDA7 are the two that will actually fire**, and both are tempting to argue past.
Write the numbers down before looking at them.

---

## 3 — What each finding changes

The point of this table: no item in this plan terminates in a plot.

| Finding | Knob it sets | Where |
|---|---|---|
| A1/B3/C6/D2 format confound | mandatory codec round-trip, with a bitrate distribution drawn from the census | `training/render.py` S3 `codec_aware` |
| A2 resampler shelf | a **resampler-choice transform**, drawn per sample, applied to the test path too | `training/registries.py` |
| A3/B2/C3/E4 grouping atoms | `speaker_ref_id`, `artifact_family`, `source_name` granularity | manifest; `training/folds.py` |
| A4 CV-en subset | `domain_cap`; the corpus row set, seeded and recorded | `configs/run_default.yaml`, `cv_en_subset.json` |
| A5/E3 duration vs the 4 s floor | whether V-B2 concatenation is required to reach 60 s | sampler `duration_range` |
| A6 silence asymmetry | `silence_lead_s` / `silence_tail_s` (both **0.0** today, so no RNG draw happens) | `configs/run_default.yaml` |
| A7 loudness gap | P-A1 on/off and its parameterization | render `normalize` |
| A8/C2/E5 evidence checks | reassignment to a mixed cell; `validity_mask_ref` | sidecar `verdict.parquet` |
| B1 pairs | `pair_id`; the artifact gallery; the voice-side 16 kHz survivability number | manifest; report |
| B4/D6 degenerate outputs | drops, and the `label_confidence` tier that per-tier loss consumes | manifest |
| C4↔D5 genre divergence | **the ACE-Step prompt distribution** for pool-D generation | Phase-B generation plan |
| C5/D2 duration + onset morphology | one crop policy over both music pools | sampler |
| D3 16 kHz survivability screen | whether E-A1 runs immediately or on schedule | experiment order |
| D4 family shortfall | the GPU-hours budget for new music families | Phase-B generation plan |
| E1 duplicates | `dup_group`; possibly a source dropped | manifest |
| E2 impulse responses | routed into a **reverb transform**, not into pool E | `training/registries.py` |
| X1 shortcut AUC | the blocking gate on training at all | — |
| X5 composition audit | `cell_mix`, `f8`, `balance_marginal_composedness` | `configs/run_*.yaml` |

---

## 4 — ⚠️ Known limits of this plan

Stated so they are not discovered as surprises.

- **Three dummy files.** Everything about the test chain (X4) is `n = 3`. Adversarial validation
  against DACON's data is impossible; X3 runs against our own splits only, which detects
  corpus-identity confounds and **not** train→test shift.
- **A spectral screen is not an EER.** D3 and B6 measure whether *summary statistics* separate
  families. An SSL frontend may find structure they miss. A collapse is strong evidence; survival
  is weak evidence and retires nothing.
- **X1 cannot prove the absence of a shortcut**, only of the shortcuts in its feature set. A
  confound carried by something not measured — a per-generator phase artifact, a mastering chain —
  passes it cleanly.
- **Pool D is one source.** Five families, no redundancy, 0.27 weight. Every conclusion about fake
  music in this plan is a conclusion about FakeMusicCaps.
- 🔴 **The stale-doc risk is live.** [PROGRESS](../../PROGRESS.md) and
  [data/12](../data/12-acquisition-status.md) both still say pool D has no acquisition path;
  `fakemusiccaps` landed on 2026-09-10 and is **not in `scripts/sources.yaml`**, which means the
  fetcher's licence gate — the G2 gate as data — never saw it. Its provenance record is complete
  and was written by hand. Reconcile the registry before the next fetch run, or the one source
  carrying 0.27 of the metric is the one source outside the mechanism that enforces the rules.
- **Nothing here has run.** Every number quoted in Phases 0–2 as an expectation is marked 🔷 and is
  a hypothesis to be measured, not a result. The repo's own standing lesson applies:
  *a green suite is not evidence that a check can fail*, and a plausible plan is not evidence that
  a confound exists.

---

## G-EDA6b — ✅ Answered 2026-09-16: the 2,442 are **three problems**, not one work list

`eda/out/_shared/reassignment_worklist.parquet`, written by `python -m eda.cli reassign`. The gate
names a count; this names the rows and the action.

🔴 **Reading the 2,442 as a single list is the error this decomposition prevents — they need
opposite actions.**

| action | rows | what it means |
|---|--:|---|
| **`reassign_cell`** | **1,342** | F-A1: asserts `voice_present = 0`, carries speech, **becomes a mixed-cell whole_file sample** |
| **`restrict_noise`** | **1,032** | ⚠️ same contradiction, but pool E — **not** a reassignment, see below |
| `unevidenceable` | 42 | under the 4 s floor. `na`, not a contradiction |
| `degenerate` | 23 | F-S4: normal length, no speech, and the pool *generates* voice |
| `sparse_real` | 3 | the same shape from pool A, which generates nothing |

### `reassign_cell` — 1,342 rows, and the target is derived, not typed

| source | pool | n | → cell | median speech ratio |
|---|---|--:|--:|--:|
| `fakemusiccaps` | D | 1,006 | **8** `(1,1,1,1)` | 0.559 |
| `fma` | C | 274 | **5** `(1,1,0,0)` | 0.666 |
| `musan-music` | C | 62 | **5** | 0.582 |

Pool C is real music, so real music + the real voice the VAD heard is **cell 5**. Pool D is
text-to-music, so any vocal in its output was generated with the rest of the clip — **cell 8**.

⚠️ **The median speech ratio is 0.56–0.67**, not something marginal sitting on the threshold. These
files carry speech through *more than half* their duration. `fma` at 13.7% contradicted is the
corpus's worst rate, and "instrumental" is the label it was carrying.

🔴 `TARGET_CELL` is checked against `CELL_TABLE` **at import**: a target that does not assert voice,
or that changes the pool's music label, raises. It is the one defect this table could introduce, and
it would move 1,006 files into a cell asserting the opposite of what was intended.

### `restrict_noise` — 1,032 rows, and it is **not** a reassignment

⚠️ **Pool E is an additive layer, not a standalone sample.** Noise is mixed *under* a composite whose
labels come from the other components. A noise clip carrying speech, mixed under a music-only
composite, makes that composite carry voice while asserting `voice_present = 0` — so **one
contaminated noise file mislabels every sample it is ever mixed into**. That is a larger failure
than a single mislabelled row, and it is invisible in any per-file audit of the composite.

And the obvious reassignment is no better: a field recording with distant speech is a poor cell-1
*voice* sample. The action is to **restrict where it may be used** — never as a layer under a
`PRESENT = 0` composite — which is neither "reassign" nor "drop".

⚠️ This overlaps [05 E5b](05-pool-e-noise.md) and the two numbers differ **because the thresholds
do**: E5 asks whether *any* speech evidence exists (`ratio > 0`, giving 2,113 pool-E rows) because
[#417333 A3](../competition/05-talkboard-qa.md) makes `PRESENT = 1` true at any audibility. G-EDA6
asks whether the contradiction is *material* (`ratio >= 0.20`, giving 1,032). Both are right for
their own question; quoting one for the other is not.

### `degenerate` — 23 rows, and they are the population B4 could not find

`mlaad` 14, `cfad-fake` 9. Normal length for their source — `mlaad`'s median is **11.28 s** against
the source's 7.99 — asserting fake voice, and carrying **no speech**.

🔴 [02 B4b](02-pool-b-fake-voice.md) screened the whole corpus for degenerate output and found no
population worth dropping. This is that population, and B4 could not see it: the right detector for
*"the TTS produced no speech"* is a **VAD**, not a clipping or level threshold. A babbling or silent
generation has ordinary peak, RMS and clipping statistics.

### `sparse_real` — 3 rows, and why they are not the previous category

`cfad-real`, ratio 0.192, just under the threshold. **F-S4 is *generation*-failure detection and
pool A generates nothing**, so filing a real recording's quiet passage as a broken generation is
wrong by construction. Measured: the first version of this analysis did exactly that, and the split
on `POOL_LABELS[pool][voice_fake]` is what prevents it.

### What this does and does not close

`G-EDA6` still **fails**, and correctly: the rows are identified and nothing has been reassigned.
Acting on the list is a **manifest** change, not an EDA one — `row_kind` goes `component` →
`whole_file`, `pool` → null, `cell` → 5 or 8 — and the EDA does not write the manifest.

⚠️ **Do not act on `reassign_cell` by dropping.** F-A1 is explicit: *"Reassign, don't drop — it
becomes a mixed-cell sample"*. 1,342 files of genuine music-with-vocals is material the corpus is
short of, and cells 5–8 are exactly where the composed majority of the test set lives.
