# 01 — Split Scheme

🔴 The whole task is generalization to **unseen generators** ([survey README](../survey/README.md) #6).
A random split will report ~0.99 and teach us nothing. Every rule below exists to make the local
number *harder* than the leaderboard, not prettier.

## 1. Grouping keys

All keys come from the [provenance ledger](../data/08-build-plan.md#provenance-ledger), which is
written at acquisition/generation time. [`E-S4`](../data/07-eda-plan.md) assigns every metadata
field to exactly one of `feature` / `split key` / `leakage risk` **before** any fold is built.

| Key | Ledger source | Rule |
|---|---|---|
| **`artifact_family`** | derived from `generator_model` + `generator_version` | Disjoint across slices. **Assigned before generation.** |
| **`source_name`** | `source_name` | Disjoint — no MUSDB18 track, Jamendo artist or LibriTTS speaker on both sides |
| **`speaker_ref_id`** / `artist_id` | `speaker_ref_id` | Disjoint within `source_name` |
| **`pair_id`** | `pair_id` (T1/T2/T3 twins) | A real file and its resynthesized twin land in the **same** slice |
| **`dup_group`** | [`E-B5`](../data/07-eda-plan.md) hash + near-duplicate pass | Whole group to one slice |

### 🔴 `artifact_family`, not `generator_model`

[09 R4](../data/09-risks-and-checks.md): many open TTS systems share a vocoder or neural codec —
HiFi-GAN and EnCodec appear underneath dozens of nominally distinct models. Holding out
"XTTS-v2" while training on three other HiFi-GAN systems is **not** a generator-disjoint split.

Assignment is by the *artifact-producing stage*, resolved in this priority order:

```
1. neural codec         (EnCodec, DAC, SoundStream, Mimi, …)
2. vocoder              (HiFi-GAN, BigVGAN, WaveNet, Vocos, …)
3. generative backbone  (diffusion / flow-matching / AR-LM / GAN)
4. end-to-end waveform  (no separable stage → family = the model itself)
```

A model is tagged with the **first** stage that applies. Two models sharing a stage at that level
share a family and can never be split across slices. The mapping lives in
`configs/artifact_families.yaml`, one row per generator, with the ledger's `generator_model`
as the join key. It is **frozen before the first fold is built** and reviewed with you once, as a
new review gate alongside the [G1–G8](../data/10-preprocessing-and-filtering.md) set — it is a
judgement call with no ground truth, and it must not be re-litigated per experiment.

⚠️ Unknowable for closed commercial systems. Any generator whose internals we cannot establish is
tagged `family = unknown_<model>` and treated as its own family — which is the conservative
direction (it may silently share artifacts with a training family, so a *good* score on it is
weak evidence; a *bad* score is still real evidence).

## 2. The four slices

| Slice | Purpose | Composition | Opened |
|---|---|---|---|
| **TRAIN** | fitting | ~65% of artifact families, disjoint sources | freely |
| **VAL** | every tuning decision, model selection | ~25% of families, disjoint sources | freely |
| **SHADOW** | domain-shift stress test only | **VAL generators × unseen acoustic conditions** | per milestone |
| **PROBE** | final sanity before a submission decision | ~10% of families, seen in **neither** TRAIN nor VAL | **≤3 times total** |

The three generator-partitioned slices sum to 100%; SHADOW is not a generator partition and draws
its files from VAL's families (see below), so it does not consume budget from the other three.

★ `[BirdCLEF playbook 2026]` E3: keep a shadow split *purely* for domain-shift stress testing,
distinct from the tuning split. The reason to keep SHADOW separate from VAL is that VAL gets
overfit by hundreds of small decisions; SHADOW answers a different question — *does the ordering
of our candidates change under channel shift?*

### SHADOW is a condition axis, not a generator axis

SHADOW holds two kinds of row, and the distinction matters for how each is read:

**S-a · Channel re-renders (paired).** The same VAL content re-rendered under a channel TRAIN has
never seen — same generators, same sources, same underlying renders. Every such row carries
`shadow_of = <val file_id>`, so the VAL→SHADOW drop is attributable to the channel and nothing
else. Drawing these from different source files instead would confound content with condition and
waste the measurement.

**S-b · Held-out content slices (unpaired).** Properties that cannot be created by re-rendering —
sung voice, Korean. These are carved out of VAL's families at fold-construction time and read as
absolute numbers, not as a paired delta.

| Condition | Kind | Rendering |
|---|---|---|
| Telephone chain | S-a | narrowband + G.711/AMR-NB, per [`A-S4`](../data/06-augmentation-spec.md) |
| Low-bitrate MP3 | S-a | 64/96 kbps round-trip, per [`A-S3`](../data/06-augmentation-spec.md) |
| Stereo | S-a | including mono-duplicated "fake stereo" ([`E-B4`](../data/07-eda-plan.md)) |
| Low component SNR | S-a | the quiet-vocal case, −15…−3 dB |
| Duration extremes | S-a | 4 s and 60 s, the boundaries of the [test range](../competition/01-overview.md) |
| Sung voice | S-b | the [R5](../data/09-risks-and-checks.md) gap |
| Korean | S-b | unknown test share — measured, not assumed |

Conditions applied to SHADOW must be **absent from TRAIN augmentation** for that milestone's
measurement to mean anything. Once a condition is folded into training, it is retired from SHADOW
and a new one takes its place; the retirement is logged.

### PROBE is sealed

PROBE exists to catch the failure that VAL and SHADOW cannot: we have tuned against VAL so hard
that VAL has become a training set. Budget is **3 openings for the entire competition**, enforced
by [VG6](04-audit-gates.md#vg6--probe-budget). Suggested spend: first end-to-end model, corpus
freeze ([G8](../data/10-preprocessing-and-filtering.md)), final candidate selection.

## 3. Fold construction

Not a single split — **grouped k-fold over artifact families**, so every family is validated on
exactly once and we get fold variance for free (★ E2: hybrid grouped + stratified is "usually the
strongest practical option"; random stratification is "fast and often optimistic, good only as a
sanity baseline").

| Head | Families available ([08 volumes](../data/08-build-plan.md#volume-targets)) | Scheme |
|---|---|---|
| Voice | ≥20 | **grouped 5-fold**, ~4 families per fold |
| Music | ≥5 | ⚠️ see below |

### 🔴 The music head cannot support the planned split

With the current floor of **≥5 music generator families**, a 5-fold grouped split puts **one
family in each validation fold**, and the PROBE slice — which needs families seen in neither
TRAIN nor VAL — **cannot be carved out at all**. Since Music is the highest-weighted component
head (0.27, vs Voice 0.18), this is the weakest point in the validation design.

Three options, in preference order:

| # | Option | Cost | Consequence |
|---|---|---|---|
| **1** | 🔴 **Raise the music-family floor to ≥8** — 5 TRAIN / 2 VAL rotating / 1 sealed PROBE | more synthesis in [Phase B](../data/08-build-plan.md) | The scheme works as designed |
| 2 | **Leave-one-family-out** over 5 families, no music PROBE | none | Per-fold EER is a 1-family estimate — very high variance; PROBE blind spot on the head that matters most |
| 3 | Accept a random music split | none | ❌ Measures nothing. Recorded only so it isn't rediscovered |

**Recommendation: option 1.** ≥8 families is a Phase B target change, not a Phase C one — raised
here because Phase C is where the constraint becomes visible. Until it is resolved, music-head
numbers carry an explicit variance caveat in the [experiment ledger](03-decision-protocol.md#5-experiment-ledger).

### Stratification within the group constraint

Within the family-disjoint partition, balance folds on: **cell** (1–9, [taxonomy](../data/02-label-taxonomy.md)),
**source_name**, **duration bucket**, **channel count**. Greedy assignment — take families in
descending size, place each into the fold whose current composition is furthest from target.

Emitted **once** to `folds.parquet` (★ E4 — fold assignment generated once and reused everywhere):

```
file_id · slice(train|val|shadow|probe) · shadow_kind(a|b|null) · shadow_of(file_id|null)
fold(0-4|null) · artifact_family
source_name · speaker_ref_id · pair_id · dup_group · cell · assigned_at · scheme_version
```

Regenerating it requires bumping `scheme_version`; every experiment records the version it ran
against, and results across versions are never compared.

## 4. Size floors

Measured, not guessed — 1,000-replication simulation of the official EER estimator
([02 §7](02-metric-harness.md#7-verified-properties)):

| VAL samples per class | 95% CI width on EER=0.10 | Resolves a 1-pt gap? (paired, ρ=0.8) |
|---:|---:|---|
| 150 | ±3.4 pts | no |
| 300 | ±2.5 pts | no |
| 600 | ±1.7 pts | 86% — coin-flippy |
| **1,200** | **±1.2 pts** | **95%** ✅ |
| 3,000 | ±0.7 pts | ~100% |

**Floor: ≥1,200 samples per class, per head, per fold pool** — i.e. ≥1,200 fake and ≥1,200 real
*voice-present* files for the Voice pool, and the same for the Music pool. Composed on the fly
([06](../data/06-augmentation-spec.md)) with a fixed evaluation seed, so this is a sampling
decision, not a storage one.

⚠️ The masked pools make this bite: the Voice EER pool contains only voice-present files
(cells 1,2,5,6,7,8) and the Music pool only music-present ones (cells 3,4,5,6,7,8). Sizing VAL by
total file count will silently under-fill both.

## 5. Reporting

Pooled numbers are **not reportable on their own**. Every experiment emits:

| Breakdown | Why |
|---|---|
| Per **cell** (1–9) | A good pooled number hides a collapsed cell — cells 6/7 are the whole reason two fake heads exist |
| Per **artifact_family** | Feeds [`E-A9`](../data/07-eda-plan.md) generator difficulty; identifies where budget goes |
| Per **fold** (mean + sd) | ★ E5: *"a model with slightly lower mean but lower fold variance is often a better final candidate"* |
| **T3-pairs only** | The corpus-identity leakage detector — [VG4](04-audit-gates.md#vg4--corpus-identity-leakage) |
| Sung vs spoken | [09 R5](../data/09-risks-and-checks.md) |
| By component SNR bucket | The hybrid-stems finding: detection tracks stem energy |

## 6. Rejected

| Approach | Why not |
|---|---|
| Random / stratified-random split | ❌ Reports ~0.99, measures memorization of generators |
| Splitting by `generator_model` name | ❌ Shared vocoders and codecs leak across the boundary ([09 R4](../data/09-risks-and-checks.md)) |
| Holding out by **source corpus only** | Insufficient — the generator axis is the one being tested |
| Using DACON's 3 dummy files as a validation slice | Far too small for any metric; they are forensic input to [`E-S1`](../data/07-eda-plan.md), nothing else |
| Tuning on SHADOW | Destroys the only unbiased domain-shift read we have |
