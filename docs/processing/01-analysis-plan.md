# 01 — From EDA results to a processing strategy: the analysis plan

**Written 2026-09-19.** The EDA is closed ([EDA/10 §1](../EDA/10-final-plan.md): 34 answered · 7
blocked · 2 Phase 2) and its exit document is [EDA/data_memo](../EDA/data_memo.md). This page is
the plan for the next step — turning those results into **a processing strategy that is a set of
config values, manifest changes and a verification harness**, not prose.

The standard it is written to is ★ `[BirdCLEF playbook 2026]` E8
([kaggle/05](../kaggle/05-transferable-playbook.md)): *"1) data split → 2) sampler/loss →
3) architecture → 4) optimizer. Common mistake: over-searching schedules before solving data shift
and imbalance."* Everything here is steps 1 and 2.

> **The one-line version.** Every EDA finding becomes one *decision* that lands in exactly one of
> four places — the **draw** (`training/sampler.py`), **preprocess** (symmetric, shipped),
> **augment** (train-only, label-blind) or the **manifest** (sidecar filter / reassignment) — and
> every decision is paired with the measurement that proves it closed the shortcut it targets,
> **measured on the rendered stream the model sees, not on the raw corpus.**

---

## 0 — What this plan produces, and the rules it obeys

### Deliverables

| # | artifact | what it is |
|---|---|---|
| D1 | `docs/processing/02-decision-table.md` | one row per EDA finding: finding → registry → knob → chosen value → the before/after number → gate that reviewed it |
| D2 | `configs/run_v1.yaml` | the shipped run config with every changed knob commented with its evidence, in the style of `configs/run_default.yaml` |
| D3 | manifest change spec | the G-EDA6 actions, `artifact_family`, `dup_group`, `domain_key`, `label_confidence`, `validity_mask_ref` — what `scripts/build_test_corpus.py`'s successor must write |
| D4 | `eda/out/_strategy/` | the harness outputs: the drawn stream's effective-feature table and its shortcut audit, per candidate policy |

### The rules that every decision below is checked against

| rule | statement | where it is stated |
|---|---|---|
| **R1** | measure on two planes; the *difference* is the analysis | [EDA/README](../EDA/README.md#two-rules-every-task-below-obeys) |
| **R2** | a **filter** is a distribution-shift decision (train-only); a **transform** is a symmetry obligation. An asymmetric rate *is* a shortcut | [data/10 P2](../data/10-preprocessing-and-filtering.md#-p2--transformations-must-be-traintest-symmetric-filters-are-train-only) |
| **P1** | filter for *label-evidence sufficiency*, never for cleanliness — 전화채널 noise is the target domain | [data/10 P1](../data/10-preprocessing-and-filtering.md#-p1--filter-for-label-evidence-sufficiency-not-for-cleanliness) |
| **the governing rule** | `P(T \| L) = P(T)` for every transform `T` | [data/06](../data/06-augmentation-spec.md#-the-governing-rule) |
| **time invariance** | every time-moving decision lives in the **draw**, never in a transform | [pipelines/03 §4](../pipelines/03-transforms.md#4--time-invariance--the-second-structural-rule) |
| **the holdout** | rank anything under a **`source_name` holdout** — the test that killed every metadata feature and that duration survived | [EDA/RESULTS §5.2](../EDA/RESULTS_FOR_ANALYSIS.md#52--groupkey-is-the-weaker-holdout-not-the-stronger-one) |
| **the gate** | shortcut AUC **< 0.60**, per head, worst stratum taken | [validation/04 VG2](../validation/04-audit-gates.md#vg2--shortcut-audit), [pipelines/05 I1b](../pipelines/05-invariants.md) |

🔴 **Priority is by metric weight.** Music fake carries **0.27** and feeds most of the **0.45** file
head; both music-head shortcuts (L1 duration 0.852, L2 first-20 ms 0.869) are on it. The crop thread
(§3.1) is therefore first, and everything else is scheduled behind it.

---

## 1 — Inputs

Everything is already on disk. Nothing in §2–§3 decodes audio except where a cost is stated.

| input | rows | what it carries | join key |
|---|--:|---|---|
| `eda/out/{A,B,C,D,E,cell8}/files.parquet` | 382,068 | M tier: `duration_s`, `orig_sr`, `container`, `group_key`, `pool` / `cell` | `file_id` |
| `eda/out/<partition>/signal.parquet` | 58,885 | S tier (both planes) + C tier: `rms_dbfs_chain`, `lead_silence_s_chain`, `longest_valid_span_s_chain`, `band_energy_{0..7}_chain`, `dc_offset_chain`, `vad_speech_ratio_50` … | `file_id` |
| `eda/out/<partition>/vectors.npz` | 58,885 | V tier: `ltas_*`, `mel_band_skew_*`, `mel_band_kurt_*`, `(n,128)` | **position** — never re-sort |
| `eda/out/_shared/envelope.parquet` | 58,885 | C5: `onset_level_deficit_db`, `onset_rise_s`, `onset_lead_silence_s`, `offset_*` | `file_id` |
| `eda/out/_shared/reassignment_worklist.parquet` | 2,442 | G-EDA6: `action` ∈ {reassign_cell, restrict_noise, degenerate, unevidenceable, sparse_real}, `target_cell` | `file_id` |
| `eda/out/_shared/b1/` | 8 files | the LJSpeech ↔ WaveFake paired experiment: per-band difference images, family correlation, surviving fraction | `vocoder` |
| `eda/out/_shared/census_generator_census.parquet` | 231 | per generator: `n`, `hours`, `duration_constant` (`min == max`, not `std == 0`) | `generator` |
| `eda/out/_shared/grouping_report.parquet` · `duplicates.parquet` · `column_roles.parquet` | 14 · 129 · 118 | grouping atoms, byte duplicates, every column's role | — |

Loaders: [EDA/RESULTS §7](../EDA/RESULTS_FOR_ANALYSIS.md#7--regenerating-and-the-convenience-loaders).
Labels are **derived**, never read: `eda.analyze.shortcut.head_labels(files, head)`.

⚠️ **Coverage differs by partition and it matters for §2.** The S/C/E tiers are **complete for D
(27,605) and E (14,194)** and a 2,000-per-source draw for A, B, C and cell 8. Any analysis over
"every file" of pool C or the voice pools is an analysis over the draw, and a manifest change that
needs a per-file verdict (§3.5) needs the measurement extended to 100% first.

⚠️ The nine traps in [EDA/RESULTS §5](../EDA/RESULTS_FOR_ANALYSIS.md#5---traps-readings-that-are-wrong)
apply to every number below. The three that bite hardest here: a speech VAD cannot see singing
(§5.1), `group_key` is the *weaker* holdout (§5.2), and C5's hypothesis runs backwards (§5.6).

---

## 2 — Analysis 0: the stream harness — measure what the model actually sees

🔴 **This is the piece the EDA could not do and everything else in this plan is scored against.**
Every AUC in the memo is over *raw files*. The model never sees a raw file; it sees a
`SampleSpec` rendered onto a timeline of drawn length. The shortcut that matters is the one that
survives *the draw*, and the draw can both remove a shortcut (a random offset) and **manufacture
one** (a sampler rule whose effect differs by pool).

### 2.1 Why it is needed: three things the current draw does, read from the code

These are readings of `training/sampler.py` and `training/render.py`, marked 🔷 until the harness
measures them. Each is a concrete hypothesis with its arithmetic:

| # | mechanism | reading | arithmetic |
|---|---|---|---|
| **H1** | `take = min(span, row.duration_s)` then `source_offset_s = U(0, duration_s − take)` (`sampler.py:407`, `:466`) | a pool-D file (10.000 s) gets **offset 0 exactly** whenever its span is ≥ 10 s — the random offset that is supposed to remove **L2** never fires for the pool that carries it | for an overlap or music-only cell the span is the timeline: `P(U(4,60) ≥ 10) = 50/56 = 0.893` for D; `P(U(4,60) ≥ 30) = 30/56 = 0.536` for C. The onset is exposed at **different rates by label** — the L2 cue is present *and* asymmetric |
| **H2** | the canvas is `np.zeros(total)` and a component occupies `[target_start_s, target_start_s + take]` (`render.py:348`) | a 10 s pool-D component on a 45 s timeline leaves **35 s of digital silence**; a 30 s pool-C component leaves 15 s. The *sample's* silence ratio is then a music-fake cue the raw corpus never had | expected uncovered fraction of a music-only timeline over `U(4,60)`: D **0.573**, C **0.164** (§2.4 states the integral) — a manufactured `silence_ratio` shortcut on the 0.27 head |
| **H3** | `usable = df[df.duration_s >= duration_range[0]]` (`sampler.py:319`) | the 4 s floor is a **filter**, and its rate differs by label: pool A loses **17.4%**, pool B **41.4%** (`signal_duration.parquet`) | not a shortcut in the rendered audio (both sides are ≥ 4 s afterwards) but a **distribution shift** that G-EDA7 must record, and a volume problem for pool B |

⚠️ H2 is the most consequential and the least visible. `check_duration` (I12) asserts the
*sample* is in `[4, 60]` s; nothing asserts how much of it is audio. `signal_duration.parquet`'s
`silence_ratio` is measured on the source file, where pool D sits at 0.0049 — the raw corpus says
"no silence shortcut" and the rendered stream may say the opposite.

### 2.2 Method

1. **Manifest.** Build a manifest restricted to the S-tier rows (so every drawn component has
   measured features): `REQUIRED_COLUMNS` from `training/manifest.py`, labels from `POOL_LABELS` /
   `CELL_TABLE`, `slice = "train"`, `fold = null`, `artifact_family` = `group_key` for fake rows
   (a placeholder — §3.6 assigns the real one). No fold table: G-EDA4 stays `na`.
2. **Draw.** `Sampler(manifest, SamplerConfig from configs/run_default.yaml).sample_spec(i)` for
   `i < 20,000` at `seed = 0`. No audio is opened.
3. **Derive the effective features per sample**, by joining each `ComponentDraw` to
   `signal.parquet` / `envelope.parquet` on `file_id`:

   | effective feature | derivation | the shortcut it carries |
   |---|---|---|
   | `sample_duration_s` | `spec.duration_s` | none by construction — same draw for every cell (the control) |
   | `music_take_s`, `voice_take_s` | the component's `duration_s` | **L1** as it survives the draw |
   | `coverage` | `Σ take / spec.duration_s` | **H2** — the manufactured silence |
   | `music_onset_exposed` | `source_offset_s < HOP_S (0.010)` | **H1** — whether the file's first frame is in the sample |
   | `music_onset_deficit_db` | `onset_level_deficit_db · exposed`, beside `exposed` itself — when the offset lands inside the file the sample opens on a mid-file cut that has no measured deficit, and the pair of columns says so rather than imputing one | **L2** |
   | `voice_lead_silence_s` | `lead_silence_s_chain` if exposed, else 0, plus the drawn `silence_lead_s` | **L4** |
   | `component_rms_dbfs` | `rms_dbfs_chain + gain_db` | level (§3.2) |
   | `bandwidth_hz` | `effective_bandwidth_hz_chain` of each component | what §3.3 must not create |

4. **Audit.** Logistic regression on the effective features, per head, **two holdouts** exactly as
   `eda.analyze.shortcut.shortcut_audit` runs them: ungrouped 5-fold (the gate) and
   `StratifiedGroupKFold` on the drawn component's `source_name` (does it survive an unseen
   publisher?). Per stratum (voice-only / music-only / mixed), **worst taken** — the I1b rule,
   because a duration shift of opposite sign in mixed vs non-mixed files scored 0.499 marginally and
   0.897 with the stratum interaction ([pipelines/05](../pipelines/05-invariants.md)).
5. **Also run the repo's own audit** on the same stream: `training.audit.run_audit(sampler,
   n=20_000, manifest=manifest)` — I1, I1b, I2, I2b, I3, I8 over the specs. It reads spec fields only
   (duration, composedness, offsets, gains), so it will pass where the effective features fail; the
   gap between the two is the finding.

### 2.3 Output and gate

`eda/out/_strategy/stream_<policy>.parquet` (one row per drawn sample, the effective features and
the labels) and `stream_audit_<policy>.parquet` (one row per head × holdout × stratum). Policy
`baseline` is `configs/run_default.yaml` as shipped.

**Gate: every head < 0.60 under both holdouts, worst stratum.** 🔷 Expected on `baseline`: fails on
`music_fake` and `music_present` through `coverage` and `music_take_s` (H2, L1) and through onset
exposure (H1, L2), and on `voice_fake`
through `voice_lead_silence_s` (L4, since `silence_lead_s = 0.0`). If the baseline **passes**, H1
and H2 are wrong and §3.1 shrinks to L2 alone — a cheaper plan, which is why this runs first.

### 2.4 The one integral worth writing down

For a component of fixed length `c` on a timeline `T ~ U(4, 60)`, the uncovered fraction is
`E[max(0, 1 − c/T)] = (1/56) ∫_{c}^{60} (1 − c/T) dT` for `c ≥ 4`:
`c = 10` → `(50 − 10·ln 6)/56 = 0.573`; `c = 30` → `(30 − 30·ln 2)/56 = 0.164`.
So under the current draw a **fake-music** sample is on average **57% silence** and a **real-music**
one **16%** — if H2 holds, that is the largest shortcut in the whole project, and it is one the
render chain created. The harness measures it rather than trusting the integral.

**Cost:** minutes. No decode. ~200 lines against existing APIs.

---

## 3 — The eight analysis threads

Each thread has the same shape: *question → inputs → method → statistic and gate → decision it
sets → where it lands → references → cost*. They are independent except where stated, and each is
scored by re-running §2 with the candidate policy.

### 3.1 Crop and duration — L1 + L2 + H1 + H2, one policy for both music pools 🔴 first

**Question.** What crop rule makes a pool-C component and a pool-D component indistinguishable in
*length*, *coverage of the timeline* and *first 20 ms*, applied by the same code to both?

**Inputs.** `files.parquet` durations (C median 30.003 s, D 10.000 s constant per generator, cell 8
median 120 s); `envelope.parquet`; `signal_duration.parquet`; §2 harness.

**Candidate policies** (each is a `SamplerConfig` change, measured in the harness):

| policy | rule | what it removes | what it costs |
|---|---|---|---|
| **P0** baseline | `take = min(span, file)`, offset `U(0, file − take)` | nothing for pool D (H1) | — |
| **P1** bounded take | draw `take ~ U(t_lo, t_hi)` with `t_hi < 10 s` (strictly — at `take = 10.000` a pool-D offset is 0 again) **independently of pool**, then offset `U(0, file − take)` — offset is > 0 for **both** pools with probability 1 | L1 (both pools same take distribution), L2 (onset never exposed), H1 | timeline coverage: a 45 s cell-4 sample now holds < 10 s of music → H2 is *worse* unless P2/P3/P4 is added |
| **P2** take-then-timeline | draw the component take first, set the **timeline** to cover it: `spec.duration_s = take + lead + tail` with lead/tail drawn | H2 for single-component cells | breaks "duration drawn first" ([data/06 pipeline order](../data/06-augmentation-spec.md#pipeline-order-per-training-sample)) and needs a spec-level change; for mixed cells the voice component sets the timeline anyway |
| **P3** fill by repetition | tile a music component to the timeline with A-A4 sigmoid joins, **at the same rate for C and D** | H2 | a splice cue — must be measured under I1b as a transform parameter; ★ PartialSpoof reports EER *worst* at zero boundaries, so joins are not free ([data/02](../data/02-label-taxonomy.md#-the-mechanism-one-composed-fraction-shared-across-cells)) |
| **P4** duration-bucketed timeline | draw `spec.duration_s` from a distribution conditional on nothing but keep components ≥ 80% coverage by rejecting draws that cannot be filled | H2, without tiling | reduces long-timeline mass for music-only cells; measured as a shift of the sample-duration distribution *by cell* — which must stay label-independent (I1b) |

**Statistic.** In the harness, per policy: the grouped AUC of `music_take_s`, `coverage`,
`(music_onset_exposed, deficit)` on `music_fake` and `music_present`; **and** `P(offset = 0)` per
pool, which must be equal (R2). Report the usable-hours cost per pool per policy.

**Gate.** `music_fake` and `music_present` < 0.60 under the `source_name` holdout, worst stratum.

**Decision it sets.** A `SamplerConfig` take rule and possibly a timeline rule; the fixed-length
element of the test set (4–60 s) is *not* something we can match, so the rule must make length
carry **no label information**, not match a distribution.

**Where it lands.** `training/sampler.py` (the draw — [pipelines/03 §4](../pipelines/03-transforms.md#4--time-invariance--the-second-structural-rule) forbids doing it in a transform), `configs/run_v1.yaml`.

**References.** ☆ `[FoR dataset]` ships a `for-2sec` variant purely to kill duration-vs-label bias
([kaggle/05 A6](../kaggle/05-transferable-playbook.md)). ⚠️ ★ `[BirdCLEF 2024, 3rd]` crops 5 s
"from the first 6 s" ([kaggle/02](../kaggle/02-audio-classification.md)) — a **fixed-window** crop
that would leave L2 in place here; do not copy it. ★ `[BirdCLEF playbook]` D4 segment grid
(5 s default, 8–10 s for context). [EDA/03 C5b](../EDA/03-pool-c-real-instrumental.md#what-it-changes):
*"the crop policy moves from optional to mandatory."*

**Cost.** Harness re-runs, minutes each. P2/P4 need a spec change; P3 needs a render change.

### 3.2 Level — is loudness a cue, a fingerprint, or both?

**Question.** Median RMS spans **22.6 dB** across sources and 35.9% of its variance is between
sources ([EDA/01 A7b](../EDA/01-pool-a-real-voice.md#a7b--the-publisher-sets-the-level-not-the-content)).
Is that (a) a publisher fingerprint the chain must neutralise, (b) a genuine synthesis cue P-A1 would
erase, or (c) both — and what does each candidate normalisation leave behind?

**Inputs.** `signal_level.parquet`; per-file `rms_dbfs_chain`, `peak_dbfs_chain`,
`crest_factor_db_chain`; `b1/b1_signal.parquet` (the same utterance real and vocoded).

**Method.** Three cheap measurements on scalars, no decode:

1. **Label AUC of level** per head, both holdouts, corpus-wide and *within* the two publishers that
   ship both halves — CFAD (real/fake voice) and, for music, `fma` vs `fakemusiccaps` (which is the
   only C-vs-D comparison there is: −15.0 vs −19.9 dB median RMS).
2. **Pair test** on B1: for 500 LJSpeech utterances × 7 vocoders, `rms_fake − rms_real` per pair.
   A non-zero median with tight spread is a **synthesis** cue (the vocoder changes level); zero says
   level is a corpus artefact only.
3. **Candidate normalisations simulated on the scalars.** Peak-normalise ⇒ the residual level is
   `rms − peak = −crest_factor`, so the post-normalisation AUC is `AUC(crest_factor_db_chain)`;
   RMS-normalise ⇒ level is constant, the residual is `peak` (= crest) and `clipping_ratio`; no
   normalisation + A-A7 gain jitter ±6 dB ⇒ convolve the level distribution with `U(−6, 6)` and
   re-measure. Each is one column and one AUC.

**Statistic and gate.** AUC < 0.60 of the *residual* level feature under a `source_name` holdout,
on `voice_fake` and `music_fake`; and the pair-test median with its IQR.

**Decision it sets.** P-A1 on/off and its form — a **preprocess** (symmetric, shipped, G6). ⚠️ The
organizers may have normalised the test set and X4 cannot tell us (n = 3, blocked). So the
decision is *robustness*, not matching: the option whose residual AUC is lowest **and** that does
not erase a pair-test cue.

⚠️ K-weighted LUFS needs `pyloudnorm`, which is **not** in the eval server's preinstalled list
([competition/02](../competition/02-submission.md)); a shipped normaliser is RMS or peak, or an
in-house K-weighting. State this in the decision.

**Where it lands.** `training/registries.py` preprocess step + `render.normalize`; also
`SampleSpec.normalize` for the test-chain stage.

**References.** [data/10 P-A1](../data/10-preprocessing-and-filtering.md#2-preprocessing-catalog-p--symmetric-train-and-test);
★ `[BC2026]` gain ±6 dB ([kaggle/06 §5](../kaggle/06-notebook-code.md)); ★ `[DFDC 2020, 1st]`
"no single processing path becomes the signal" ([kaggle/05 B6](../kaggle/05-transferable-playbook.md)).

**Cost.** Minutes.

### 3.3 DC removal and the high-pass — one decision, two opposite pulls

**Question.** Five of seven vocoder signatures live below 72 Hz (B1b), so a rumble high-pass would
delete pool B's largest discriminative feature; but `cfad/pwg` is identifiable at **AUC 0.998** from
`dc_offset_chain` alone (B6b). Is the sub-72 Hz energy a *real/fake* cue across the corpus or only a
*generator id* inside pool B — and which of {nothing, DC-only, HP 20 Hz, HP 40 Hz, HP 72 Hz}
keeps the first while giving up the least?

**Inputs.** `b1/b1_difference_chain.parquet` (per-band real−fake LTAS difference, 128 bands),
`b1/b1_surviving_fraction.parquet`, `signal.parquet` `band_energy_0_chain` and `dc_offset_chain`,
`screen_separability_generator.parquet`.

**Method.**
1. **Across the corpus, not only in the pair:** AUC of `band_energy_0_chain` and `dc_offset_chain`
   on `voice_fake` under a `source_name` holdout, and within CFAD. B1b is a same-utterance pair on
   one real source; this asks whether the same band separates A from B when the publisher changes.
2. **Per candidate cutoff**, from the B1 difference image: the fraction of each vocoder's difference
   energy that lies above the cutoff (the analogue of `b1_surviving_fraction`, over frequency rather
   than over the plane).
3. **DC as a cue vs an id:** `dc_offset_chain` AUC real-vs-fake (a cue) vs its AUC generator-vs-rest
   inside pool B (an id, already 0.998 for `pwg`). A generator id is only worth keeping if the real
   side lacks it; if `cfad-real` has the same DC distribution as every CFAD generator but `pwg`,
   removing DC costs one generator's id and no real/fake cue.

**Statistic and gate.** Surviving difference-energy fraction per vocoder per cutoff (report), and
the real/fake AUC of the removed band (the loss). No pass/fail — this is a trade recorded in D1.

**Decision it sets.** `AudioConfig.band_hz` (exists) and whether a DC-removal preprocess step is
registered. ⚠️ Whatever is chosen is **shipped**: it runs on the test files too (P2).

**Where it lands.** `models/audio` `AudioConfig.band_hz`; preprocess registry.

**References.** [EDA/02 B1b](../EDA/02-pool-b-fake-voice.md#b1b---answered-2026-09-16-the-artifact-does-not-die-at-16-khz);
[data/10 P-S4, P-B2](../data/10-preprocessing-and-filtering.md#2-preprocessing-catalog-p--symmetric-train-and-test);
★ `[G2Net 2021, 3rd]` real-class average-PSD whitening as the *opposite* strategy — remove the
common channel, leave the anomaly ([kaggle/03](../kaggle/03-weak-signal-anomaly.md)) — noted as a
later front-end experiment, not a preprocess decision.

**Cost.** Minutes; the B1 vectors are `b1_vectors.npz`.

### 3.4 Silence — L4, and the parameters of the augmentation that is switched off

**Question.** `lead_silence_s` separates real from fake voice at 0.612 corpus-wide, **0.717** inside
CFAD, **0.639** on the same utterance one vocoder apart ([EDA/01 A6b](../EDA/01-pool-a-real-voice.md#a6b---answered-2026-09-17-the-silence-shortcut-is-real-and-it-survives-pairing)).
`silence_lead_s` / `silence_tail_s` are `0.0`. What draw makes the *effective* leading silence of a
rendered voice sample label-independent, at what probability, and what does it cost the 4 s floor?

**Inputs.** `lead_silence_s_chain`, `tail_silence_s_chain` per source (A6b's table); the harness.

**Method.** In the harness, for each candidate `(p, range)` — e.g. `p ∈ {0.2, 0.5, 1.0}` ×
`lead ~ U(0, {0.5, 1.0, 2.0})` s — the effective lead is `file_lead·exposed + drawn_lead`. Measure
its AUC on `voice_fake` under three controls, exactly as A6b did: corpus-wide, within CFAD, on the
LJSpeech ↔ WaveFake pair. ⚠️ At `p = 0.2` (the A-A11 default) 80% of samples keep the raw cue, so
the corpus-wide AUC may not fall under 0.60 — the analysis chooses `p`, it does not assume it.

**Statistic and gate.** Effective-lead AUC < 0.60 under **all three** controls, and the fraction of
voice components whose span after `lead + tail` still holds a 4 s non-silent window (pool B holds a
4 s span in only 10.3% of files today — `has_4s_span` in `signal_duration.parquet`).

**Decision it sets.** `silence_lead_s`, `silence_tail_s`, and their `p`. **Symmetric across pools A
and B** or it becomes the cue (R2). ⚠️ Changes the drawn stream — an I1b re-run is part of the
verification, not optional. **Do not trim** — settled by A5b: trimming costs A 7.78 pp vs B 4.12 pp
of the 4–60 s window, an asymmetric transform ([EDA/data_memo §8.9](../EDA/data_memo.md#8--what-to-carry-into-processing-strategy)).

**Where it lands.** `configs/run_v1.yaml` `sampler.silence_lead_s` / `silence_tail_s`; the draw.

**References.** [data/06 A-A11](../data/06-augmentation-spec.md); the ASVspoof silence-statistics
shortcut ([survey/01](../survey/01-sota-speech.md)); ★ `[BC2026]` silence insertion as a
segment-level aug ([kaggle/06](../kaggle/06-notebook-code.md)).

**Cost.** Minutes per candidate.

### 3.5 Component evidence — the G-EDA6 actions, validity masks, and the 4 s floor

**Question.** 2,442 measured rows dispute their labels, decomposed into five actions
([EDA/07 G-EDA6b](../EDA/07-order-and-gates.md#g-eda6b---answered-2026-09-16-the-2442-are-three-problems-not-one-work-list)).
What does the manifest do with each, what does each do to the pool balances, and which need a
measurement extended before they can be applied?

**Inputs.** `reassignment_worklist.parquet`; `signal_content.parquet`; `screen_cell9.parquet`;
`screen_degenerate.parquet`; `vad_speech_ratio_50` (⚠️ `vad_spans_50` is a **count**, not a span
list — masks need a re-decode, see below).

**Actions and what each needs:**

| action | rows (draw) | manifest change | balance effect | needs first |
|---|--:|---|---|---|
| `reassign_cell` C → **5** | 274 `fma` + 62 `musan-music` = 336 of 2,660 measured (**12.6%**) | `row_kind = whole_file`, `cell = 5`, `pool = null`: real voice **+** real music. 🔴 These are the corpus's **first natural cell-5 rows** — the material `f8 = 0` assumes exists (27.5% of cell 5 natural, [pipelines/02 §3](../pipelines/02-sampler.md#3-the-composed-fraction-is-a-constraint-not-a-knob)) and that no source supplies today | pool C shrinks ~13% (≈ 1,100 files at census scale); cell 5 gains its only whole-file rows | 🔴 **VAD over all 8,660 pool-C files** — the draw covers 2,660. Cost ≤ 3 h at the measured 0.61 ah/min; VAD-only is faster |
| `reassign_cell` D → **8** | 1,006 of 27,605 (**3.6%**, complete) | `whole_file`, `cell = 8` | pool D −3.6%; cell 8 gains 1,006 short (10 s) rows beside SONICS' 49,074 long ones — the **duration** cue inside cell 8 needs §3.1's crop | nothing — D is 100% measured |
| `restrict_noise` | 1,026 `compspoof-env-bonafide` + 6 `musan-noise` = 1,032 of 14,194 (**7.3%**, complete) | pool E is a *layer*; a speech-carrying noise file may only be layered under `voice_present = 1` composites. Two implementations: (i) a manifest flag the sampler reads when drawing the noise role; (ii) drop them from pool E | (ii) drops 7.3% of pool E → **G4 fires** (> 5%). (i) needs a sampler feature and no loss | decide (i) vs (ii) at G4/G7 |
| `degenerate` | 9 `cfad-fake` + 14 `mlaad` = 23 | drop, **or** `label_confidence = low` and per-tier loss (F1/F2) | negligible | nothing |
| `unevidenceable` / `sparse_real` | 42 + 3 | keep; `label_confidence = asserted` | none | nothing |

**Validity masks (V-A1 → V-A3).** The C tier stored the speech *ratio* and span *count*, not the
spans. Building `validity_mask_ref` (50 ms resolution, [data/10 §4](../data/10-preprocessing-and-filtering.md#4-salvage-v--how-we-rescue-partially-bad-files))
needs one more C-tier pass that writes the Silero timestamps — over the **voice pools** (A 74,846 /
B 207,689 files, 497 h; ~14 h at 0.61 ah/min, or run VAD alone). Before paying that: the harness
answers whether masks are needed at all — measure the `silence_ratio_chain` of drawn voice components
and how many samples would be > 40% silence. 🔷 With A's median silence ratio at 0.17 and B's at
0.15, masks may be a Phase-2 nicety rather than a strategy item.

**The 4 s floor (H3).** The sampler's `duration_s >= 4.0` filter removes **41.4% of pool B and
17.4% of pool A** — an asymmetric *rate* that G-EDA7 exists to record. Options to measure: (a) accept
and record; (b) **V-B2 concatenation** of same-`group_key` utterances to reach ≥ 4 s, applied to
pools A and B **at the same rate** with A-A4 sigmoid joins and tagged `spliced = True`; (c) lower
the floor for *components* while keeping it for *timelines* — a component may be 2 s inside a 4 s
sample. (c) is free and is what the composition already allows for sequential structure; measure
its effect on the voice pools' usable hours.

**Statistic and gate.** Per action: files and hours moved per pool; per-cell salvage / drop rate
parity (**G4**: no pool loses > 5% without review; salvage rates within 10 pp across cells);
**G-EDA6 re-run at 100% coverage** → the contradicted count after reassignment; **G7** for the
reassignment itself (10 listenable examples per source).

**Decision it sets.** The manifest change spec (D3).

**References.** [data/10 F-A1, F-S4, V-A1..V-B2, G4, G7](../data/10-preprocessing-and-filtering.md);
★ `[BC25 separation notebook]` Silero VAD, threshold 0.4 vs 0.5 ([kaggle/06 §9](../kaggle/06-notebook-code.md));
★ `[Freesound 2019, 1st]` per-tier loss and ☆ `[HMS 2024]` two-stage clean→noisy
([kaggle/05 F1, F2](../kaggle/05-transferable-playbook.md)); ★ `[BirdCLEF playbook]` A7 —
automate hash/near-dup/label-map/corrupt/fold-leak checks, **manually review** suspicious
top-scoring clips ([kaggle/05](../kaggle/05-transferable-playbook.md)).

**Cost.** Manifest edits: hours. Pool-C VAD: ≤ 3 h decode. Voice-pool masks: ~14 h, deferred until
the harness says they are needed.

### 3.6 Split keys and domain balance — what the fold builder will be given

**Question.** Which columns does `folds.parquet` get, from what evidence, and can the music head be
validated at all?

**Inputs.** `grouping_report.parquet`; `census_generator_census.parquet`; `b1/b1_family_correlation_chain.parquet`;
`census_mlaad_inventory.parquet`; `duplicates.parquet`; [EDA/05 E1b](../EDA/05-pool-e-noise.md#-e1b--e1-pass-2-the-method-fell-short-and-following-it-found-a-real-leak) (CompSpoof's 292 shared parents);
`files.parquet` `group_key`.

**Products, each with its method:**

| column | method | evidence |
|---|---|---|
| **`artifact_family`** | [validation/01 §1](../validation/01-split-scheme.md#-artifactfamily-not-generatormodel)'s priority (codec > vocoder > backbone > model) **by name** for CFAD 11 / MLAAD 133 / FakeMusicCaps 5 / SONICS 5; for the **7 WaveFake vocoders, by measured correlation**: cluster `b1_family_correlation_chain` (chain plane: melgan ↔ melgan_large **0.82**, melgan ↔ multi_band_melgan **−0.23**; the memo's 0.895 / −0.186 are the native plane — the names are wrong on both) and reconcile with the name-based table where they disagree. Frozen into `configs/artifact_families.yaml`, reviewed once | [EDA/02 B1b-iii](../EDA/02-pool-b-fake-voice.md#b1b-iii--artifact-families-the-names-are-wrong) |
| **`dup_group`** | union of `duplicates.parquet` (39 groups) and the CompSpoof parent-recording map | E1, E1b |
| **`speaker_ref_id`** | `group_key` where `group_key_kind ∈ {path, publisher}`; `ljspeech` is one atom and cannot be split | `grouping_report` |
| **`domain_key`** | `source_name × generator` (`eda.analyze.screens.generator_key`, not `group_key` — trap §5.7) | — |
| **`pair_id`** | `eda.analyze.pairs.pair_table` for LJSpeech ↔ WaveFake (13,100 utterances × 7); CFAD is a second pair set — **7,900 `SSB…` utterance ids** appear in both `cfad-real` and `cfad-fake` (verified on `files.parquet`), one real recording against up to 11 vocoders | B1; [validation/04 VG4](../validation/04-audit-gates.md#vg4--corpus-identity-leakage) consumes it |

**Music-head fold feasibility.** Count music *families* after `artifact_family` is assigned: pool D
has 5 (all 10 s, `mustango` at AUC 1.000 on duration alone); cell 8 adds SONICS' 5
(`chirp-v2-xxl-alpha`, `chirp-v3`, `chirp-v3.5`, `udio-120s`, `udio-30s`) — whether Chirp versions
are one family or three is exactly the judgement the priority order settles. **≥ 8 is the floor**
for the designed 5-fold + PROBE ([validation/01 §3](../validation/01-split-scheme.md#-the-music-head-cannot-support-the-planned-split));
under it, leave-one-family-out with the variance caveat. This count is a strategy output.

**Domain balance.** Hours per `domain_key`: SONICS is **1,970.6 of 2,676 h** and `chirp-v3.5`
alone is 1,057 h. `domain_cap = 500` is a *file* weight over component rows; ⚠️ `_whole_by_cell`
draws whole-file rows **uniformly** (`sampler.py:396`) — check whether the cap reaches whole-file
rows at all, and if not, that is a sampler change. Also count Korean: MLAAD `ko` is 12 generators ×
30 files (0.92 h) against `zeroth-korean` 52.9 h real — a real/fake **language** imbalance to
record, since language is not a split key and cannot be balanced by folds.

**Statistic and gate.** VG1 A1–A7 on the built table; G-EDA3 (≥ 6 atoms per role — 3 sources are
genuinely short and stay so); G-EDA4 goes from `na` to measured; I21 realised diversity ≥ 3 families
per fake role.

**Decision it sets.** D3's split columns and `configs/artifact_families.yaml`.

**References.** ★ `[BirdCLEF playbook]` E2 hybrid grouped + stratified, E4 folds generated once
([kaggle/05](../kaggle/05-transferable-playbook.md)); ★ `[BirdCLEF 2024, 3rd]` per-class cap 500
(C7); DOSS 0.2k h balanced > 6.4k h naive ([papers/05](../papers/05-generalization.md)).

**Cost.** Hours of table work; the family judgement is a review gate.

### 3.7 Format and codec — what the normalise stage must draw

**Question.** Metadata separates the pools perfectly (`pcm_f32le` is FakeMusicCaps alone; three
header fields identify cell 8 — [EDA/06 X1c](../EDA/06-cross-pool.md#the-sharpest-case-three-header-fields-identify-cell-8-exactly))
and the render chain neutralises the *headers*. Does anything the codec did to the **audio** survive
decode — and what bitrate / container menu should the A-S3 round-trip draw so the test chain is
represented and no source's native codec is label-correlated?

**Inputs.** `census_format_census.parquet`, `census_codec_provenance.parquet`
([EDA/01 C6](../EDA/01-pool-a-real-voice.md#-c6s-answer-the-mp3-question-is-about-bit-rate-not-about-mp3):
the mp3 question is about bit rate), `effective_bandwidth_hz_chain`, `near_nyquist_ratio_chain`.

**Method.** In the harness, `bandwidth_hz` and `near_nyquist_ratio_chain` of the drawn components
per head under both holdouts (already in §2.2's feature table). Then the menu: the census's native
`bit_rate` distribution per pool, and a round-trip menu `{mp3 64/96/128/192, wav, flac}` with
probabilities that are **the same for every cell** — the draw is `SampleSpec.normalize`, read by
I1b, so a label-correlated menu is caught there.

**Statistic and gate.** Residual bandwidth AUC < 0.60 under `source_name`; I1b on the `normalize`
draw.

**Decision it sets.** The A-S3 menu and `p`; whether A-S4's telephone leg runs at `p ≈ 0.2`.
Lands in `SampleSpec.normalize` / `render._normalize` and `configs/run_v1.yaml`.

**References.** [data/06 A-S1, A-S3, A-S4](../data/06-augmentation-spec.md); ★ `[DFDC 2020, 1st]`
compression with a quality *range* ([kaggle/05 B8](../kaggle/05-transferable-playbook.md)); RADAR /
SAFE — codec-aware preprocessing is the top robustness intervention ([kaggle/01](../kaggle/01-ai-content-detection.md)).

**Cost.** Minutes; the menu is config.

### 3.8 (Exploratory) The 45 M unread measurements — LTAS and mel moments

Not a decision thread; an evidence thread for two decisions above (§3.3, §3.6's `aug_strength`) and one later front-end experiment.

**Question.** Under a `source_name` holdout, what in the `[128]` LTAS / mel-skew / mel-kurt vectors
separates each head? Three uses: (i) a per-source *identifiability* score → `aug_strength`
(★ `[Bengali.AI 2023]` B7: heavy augmentation on clean sources, light on degraded — the number is
how easily the source is recognised); (ii) class-conditional LTAS per cell and generator, 0–8 kHz
(E-A3) — the figure the 2nd-stage report wants; (iii) whether real-class average-PSD whitening
(★ `[G2Net 2021, 3rd]` D2) is worth an experiment.

**Method.** Logistic regression on `ltas_chain` (+ the two moment vectors), per head, both holdouts,
and per-source one-vs-rest; then **adversarial validation** (E-A2 / X3, ★ `[G2Net 2021, 3rd]`)
between sources *within* a label — AUC ≈ 0.5 says two real sources are acoustically alike, 0.99
says the model can name the archive. ⚠️ `music_fake` grouped AUC is unmeasurable by `source_name`
(4 sources — one held out leaves a single class); use `group_key` **and label it as the weaker
holdout** (trap §5.2).

**Output.** `eda/out/_strategy/vector_audit.parquet`; `aug_strength` per source in D3.

**Cost.** Minutes to an hour; vectors are on disk.

---

## 4 — Dividing the data: the slice-specific strategies

The threads above are corpus-wide levers. This section is what each slice of the corpus needs from
them, so the strategy document (D1) can be written per slice and the sampler configured per role.

| slice | what the EDA says | the decisions that bind it | open |
|---|---|---|---|
| **A real voice** (4 sources, 194.8 h) | level is a publisher fingerprint (MUSAN peak-normalised to −0.000265 dBFS, 87% clipping); `ljspeech` lead silence 0.00 like TTS; 17.4% under 4 s | §3.2 level; §3.4 silence on, symmetric; §3.3 no high-pass; the 4 s floor rate recorded (§3.5) | Korean is one real source (`zeroth-korean`) vs 12 fake generators |
| **B fake voice** (3 sources, 302.8 h) | 41% under 4 s, 10.3% hold a 4 s span; vocoder artefact below 72 Hz survives 16 kHz at 82–115%; generators identifiable from one scalar (median best-AUC 0.904); WaveFake ships its CV half twice | §3.4; §3.3 (keep sub-72 Hz); §3.6 families by correlation; `dup_group`; H3 salvage; `domain_cap` | `artifact_family` for MLAAD's 133 drawn generators is name-based only |
| **C real instrumental** (2 sources, 109.2 h) | 30 s excerpts opening 43 dB below their own level (71% `natural`); 12.6% carry speech; `fma` loses 10.4 kHz to the chain (the only source it cuts); 23 DENY / 2,884 ND licence rows | §3.1 crop from inside; §3.5 reassign to cell 5 after a 100% VAD pass; the fma allowlist regeneration when volume binds (~10 min, [EDA/03 C1b](../EDA/03-pool-c-real-instrumental.md#c1b---answered-2026-09-16-the-allowlist-is-stale-not-restrictive--and-the-split-is-23--2884)) | `mtg-jamendo` is in the store, unfetched — the third publisher pool C lacks |
| **D fake instrumental** (1 source, 5 generators, 77.3 h) | every file 10.000 / 10.180 / 10.242 s; opens at its own level (73% `hard_cut`); chain is a no-op (native 16 kHz); 3.6% carry speech; `pcm_f32le` identifies it in metadata | §3.1 — this pool is the reason for the thread; §3.5 → cell 8; §3.6 families; H1/H2 | one publisher for 0.27 of the metric — every conclusion is a conclusion about FakeMusicCaps |
| **cell 8 AI songs** (SONICS, 1,970.6 h — 74% of the corpus) | median 120 s, 80% over 60 s; sung — 72% *unevidenceable* by a speech VAD, not mislabelled; 5 generators | §3.1 crop from inside (offset already random for whole-file rows, `sampler.py:407`); `domain_cap` reach on whole-file rows (§3.6); families | cell 5 natural counterpart exists only via §3.5's reassignment |
| **E noise** (3 sources, 21.6 h) | 4.00 s clips (CompSpoof), 58.4% hold a 4 s span; 86.7% `hard_cut`; 7.3% carry speech; RIRS isotropic set is impulse responses | `restrict_noise` (i) or (ii) at G4; cell-9 viability 11,901 rows ([EDA/05 E5b](../EDA/05-pool-e-noise.md#e5b---answered-2026-09-16-11901-files-are-defensibly-cell-9-and-2113-are-not)) | music presence never evidenced — every cell-9 count is an upper bound |
| **telephone channel** | no telephony source on disk (`cfad-codec`, `cfad-noisy`, `codecfake` blocked) | comes entirely from the normalise stage A-S4 (8 kHz leg + G.711 implemented; AMR/OPUS not) | X4 forensics blocked at n = 3 — the chain is an assumption |
| **`mixed` partition** | `compspoof-v2` is on disk, declared, **never enumerated** — and it carries our exact two-component label structure | not an EDA item; a strategy input if cells 6/7 need whole-file material | probe it (M tier only, minutes) before D3 is frozen |

🔴 **Per-head reading.** *Music heads (0.27 + most of 0.45):* §3.1 and §3.5 (cell-5 rows) are the
whole game; §3.6 decides whether the head can be validated. *Voice heads (0.18):* §3.4 and §3.3;
the voice heads are clean of publisher confounds (X1d) and not clean of the silence artefact
(A6b). *Presence heads (0.10):* set by the cell mix and C1; nothing in this plan moves them except
the cell-5 reassignment, which is what makes "voice present, music present, both real" a real
population.

---

## 5 — Order, cost, and the verification that closes each thread

| step | thread | needs | cost | closes |
|---|---|---|---|---|
| 1 | §2 harness, `baseline` | nothing | ~½ day | H1/H2/H3 measured; the baseline audit |
| 2 | §3.1 crop policies P1–P4 | step 1 | ½ day | L1, L2 on the stream |
| 3 | §3.4 silence, §3.2 level, §3.3 DC/HP, §3.7 codec menu | step 1 | ½ day together | L4; P-A1; `band_hz`; A-S3 menu |
| 4 | §3.5 manifest actions; pool-C VAD at 100% | — (decode ≤ 3 h, background) | 1 day | G-EDA6 count; G4/G7 packs |
| 5 | §3.6 split columns, family table, music-family count | §3.5 | ½ day | G-EDA3/4, VG1 |
| 6 | §3.8 vectors, `aug_strength` | — | ½ day | B7 |
| 7 | **D1–D4** written; harness re-run on `run_v1.yaml` | all | ½ day | the gates below |

⚠️ **Budget.** ~4½ engineer-days serial, against a leaderboard close on **2026-09-29** (10 days
from writing). Steps 2 and 3 are independent and run the same day; step 4's decode runs in the
background from day one. Target: **strategy frozen in 3 days**, so training gets the week.

**Verification of the whole** — the strategy is accepted when, on the stream drawn from
`configs/run_v1.yaml` over the changed manifest:

1. §2's audit: every head **< 0.60**, both holdouts, worst stratum — including `coverage`,
   `music_take_s`, onset exposure and effective lead silence;
2. `training.audit.run_audit` — I1, I1b, I2, I2b, I3, I8, I21 pass (the stream changed: I1b re-run);
3. **G-EDA7** measured, not `na`: every filter's rate per label stated (the 4 s floor, `restrict_noise`,
   `degenerate`) with the tolerance it is held to;
4. **G4** parity: no pool > 5% dropped without review; salvage within 10 pp across cells;
5. **G-EDA6** re-run at 100% coverage of pool C: contradicted count after reassignment;
6. **G-EDA3/G-EDA4** on the built fold table; the music-family count against the ≥ 8 floor;
7. the human gates that fired — **G3** (any threshold), **G4**, **G6** (level, DC/HP, silence),
   **G7** (reassignment) — each with its review pack ([data/10 §5](../data/10-preprocessing-and-filtering.md#5--human-review-gates)).

⚠️ **What passing does not prove.** X1 and this harness see the features they are given. A
confound carried by something unmeasured — a per-generator phase artefact, a mastering chain — passes
cleanly ([EDA/07 §4](../EDA/07-order-and-gates.md#4---known-limits-of-this-plan)). The leak
tripwires in the training loop (music fake unseen-generator < 3% EER, voice < 1%: *"suspect the
split, not the model"*) are the second line, and VG4's T3-pair gap the third.

---

## 6 — What this plan cannot answer, stated

* **The test set.** X3/X4 are blocked at n = 3. Every symmetric choice in §3.2–§3.3 is a
  robustness choice, not a match. The four-submission LB decomposition
  ([validation/05](../validation/05-lb-probe-plan.md)) is the only external read, and it is spent
  once.
* **`FILE_FAKE` (0.45).** Auditable only over the spec stream (X5) — §2 covers it *through* the
  component features, which is the most the corpus allows.
* **Sung voice.** Not evidenced by anything on disk; SONICS is labelled by its publisher's
  `no_vocal = False`, and cell-5 natural songs will be evidenced only as far as a speech VAD hears
  sung vocals (28.1% — [EDA/00 §4e](../EDA/00-harness.md#4e---what-the-vad-can-and-cannot-evidence)).
* **PANNs.** Declined (32 kHz model, conflates singing with Music). Music presence stays asserted.
* **A third pool-C publisher and a second pool-D source** are acquisition decisions, not analysis.

---

## 7 — Kaggle and reference methods used here, indexed

| method | source | used in |
|---|---|---|
| E8 order: split → sampler/loss → architecture | ★ `[BirdCLEF playbook 2026]` [kaggle/05](../kaggle/05-transferable-playbook.md) | the plan's scope |
| A2/E-S4 map every field to feature / split key / leakage risk | ★ playbook · [data/07](../data/07-eda-plan.md) | done — `column_roles.parquet`; consumed in §1, §3.6 |
| A3 data memo before tuning | ★ playbook | done — [EDA/data_memo](../EDA/data_memo.md) |
| A4/E-A2 adversarial validation | ★ `[G2Net 2021, 3rd]` [kaggle/03](../kaggle/03-weak-signal-anomaly.md) | §3.8 between sources within a label; VG3 later |
| A5/E-A6 statistics-T low-information filter | ☆ `[BirdCLEF 2024, 1st]` (unverified) | `stat_t_*` columns exist in `signal.parquet`; **not** adopted as a filter — P1 says noise is the domain; used only as a candidate mask feature in §3.5 if masks are needed |
| A6 duration-vs-label trap (`for-2sec`) | ☆ `[FoR]` | §3.1 |
| A7 automate dedup / label-map / corrupt / fold-leak; manually review | ★ playbook | §3.5, §3.6; G7 review packs |
| B7 augmentation strength by source cleanliness | ☆ `[Bengali.AI 2023]` | §3.8 → `aug_strength` |
| B3/A-A4 SigmoidConcatMixer | ★ `[Freesound 2019, 1st]` | §3.1 P3, §3.5 V-B2 joins |
| C4/A-A3 low-SNR-skewed mixing | ★ `[G2Net 2021, 3rd]` | already in `run_default.yaml` (`gain_db_mean −3.6`); unchanged |
| C7 per-class cap 500 / DOSS domain cap | ★ `[BirdCLEF 2024, 3rd]` · [papers/05](../papers/05-generalization.md) | §3.6 |
| D2 real-class PSD whitening | ★ `[G2Net 2021, 3rd]` | §3.3, noted for a later front-end experiment |
| E2/E4 hybrid grouped folds, generated once | ★ playbook | §3.6 |
| F1/F2 per-tier loss, two-stage clean→noisy | ★ `[Freesound 2019, 1st]` · ☆ `[HMS 2024]` | §3.5 `label_confidence` |
| Silero VAD at 0.4/0.5 | ★ `[BC25 separation]` [kaggle/06 §9](../kaggle/06-notebook-code.md) | done at 0.5 with the 0.4 shift recorded; §3.5 extends coverage |
| gain ±6 dB, silence insertion | ★ `[BC2026 Distilled-SED]` [kaggle/06 §5](../kaggle/06-notebook-code.md) | §3.2, §3.4 |
| ❌ pseudo-labelling the test set, cross-file normalisation, test-set EDA | rule 2.3 / 2.4 | not used; the harness measures our own stream |

---

*Numbers quoted here are from the parquets named in §1 as of 2026-09-19; where a figure in this
page and the artifact disagree, the artifact wins. The 🔷 items in §2.1 are code readings with
arithmetic, and the harness exists to replace them with measurements.*
