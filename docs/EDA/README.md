# EDA — Per-Pool Analysis Plan

**Written 2026-09-11.** What to measure in each pool, why it is worth measuring, and which
processing knob each answer sets.

This is the *execution* plan. The *method catalog* is [data/07](../data/07-eda-plan.md) (tiered
E-S1…E-C4) and the *processing catalog* is [data/10](../data/10-preprocessing-and-filtering.md)
(P-/F-/V- items). Neither is organized by pool, and neither can be run — they say what a method is
for, not what to point it at. This directory closes that gap: every task below names a pool, a
concrete computation, an output file, and a gate.

| File | Pool | Size in S3 | Sources |
|---|---|---|---|
| [01](01-pool-a-real-voice.md) | **A** — real voice | 111.0 GiB | common-voice-en/ko, libritts-r, zeroth-korean, ljspeech |
| [02](02-pool-b-fake-voice.md) | **B** — fake voice | 50.9 GiB | wavefake, mlaad, **ctrsvdd** (sung) |
| [03](03-pool-c-real-instrumental.md) | **C** — real instrumental | 12.6 GiB | fma, mtg-jamendo (+ musan `music/`) |
| [04](04-pool-d-fake-instrumental.md) | **D** — fake instrumental | 12.0 GiB | fakemusiccaps |
| [04](04-pool-d-fake-instrumental.md) | **cell 8** — whole-file AI songs | 30.4 GiB | **sonics** (`row_kind: whole_file`) |
| [05](05-pool-e-noise.md) | **E** — non-musical sound | 11.5 GiB | musan, rirs-noises (+ compspoof `env_sources/*/bonafide/`) |
| [06](06-cross-pool.md) | cross-pool | — | the shortcut audit, duplicates, the test chain |
| [00](00-harness.md) | the shared harness | — | what every script emits, and the two measurement planes |
| [07](07-order-and-gates.md) | execution | — | phase order, cost, gates, and what each finding changes |
| [08](08-real-run.md) | **the real run** | — | wave priority over the 15 acquired sources, verified layouts, and the four defects found checking it |
| [09](09-next-steps.md) | **Next steps** — the plan of record | — | what remains, in dependency order |
| [memo](data_memo.md) | 🔴 **The data memo** — X7, the exit condition | — | inventory, class imbalance, domain shift, and the leakage hypothesis |

`compspoof-v2` (111.8 GiB) spans pools and is treated per-component; its `env_sources` halves are
covered in [05](05-pool-e-noise.md).

🔴 **SONICS is not a pool.** It was acquired as `pool: D`; its own `fake_songs.csv` reports
`no_vocal = False` for all 49,074 rows, and `POOL_LABELS["D"]` asserts `voice_present = 0`. It is
**cell 8** (mixed F/F) and `validate_manifest` requires `pool = null` on a whole-file row. It is the
corpus's first whole-file source, which is a gift and a trap at once — see
[04](04-pool-d-fake-instrumental.md). ⚠️ And **CtrSVDD is fake *sung voice*, not fake music**: the
rules classify vocals as voice, so it is pool **B**, and pool D remains FakeMusicCaps alone.

---

## Why EDA is worth doing here, specifically

Three reasons, in descending order of size.

**1. The licence gate makes the pool small and hard to grow.** ([data/01](../data/01-rules-check.md))
When you cannot buy score with more data, you buy it by knowing exactly what you have. DOSS is the
evidence: 0.2k h domain-balanced beat 6.4k h naive, 2.77% vs 3.29% EER
([papers/05](../papers/INDEX.md)). **Balance is an EDA output.** You cannot cap a domain you have
not counted.

**2. Our corpus is assembled from sources that were never meant to sit beside each other, and
`FILE_FAKE` is nearly aligned with `which archive it came from`.** Pool A is five speech corpora;
pool B is two fake-speech corpora; pool C is two music corpora; pool D is one TTM corpus. Every one
of them was produced by a different recording chain, at a different sample rate, in a different
container, normalized differently. A classifier that learns *the archive* scores beautifully in CV
and nothing on the leaderboard — and it is the single failure mode this project has documented
most often and defended against least, because it lives in the data rather than the code.
☆ `[LLM-Detect-AI 2024, efficiency prize]`: off-distribution data *"caused severe data drift which
further increased the CV/LB gap"*.

**3. Every finding has a lever.** This repo's pipeline is unusually parameterized — the sampler,
the render chain, the transform registries, the fold builder and the audit all read config. An EDA
finding here does not become a note in a doc; it becomes a value in `configs/run_*.yaml`, a
`validity_mask`, a `domain_key`, or a row dropped by `build_test_corpus.py`. [07](07-order-and-gates.md)
tabulates the mapping.

---

## 🔴 What "feature engineering" means in this project — and what it does not

The user-facing goal is *"data processing and feature engineering."* Half of that is the wrong
shape for this competition, and the repo already has the evidence. Stating it once so no effort is
spent in the wrong direction:

**❌ Handcrafted acoustic features as model input are rejected**, and not on taste. CtrSVDD's own
baselines: raw waveform **13.75%** EER vs MFCC **26.67%** ([survey/03](../survey/03-sota-singing-mixed.md)).
The SSL frontend beats every handcrafted representation throughout this literature. That is Tier X
in [data/07](../data/07-eda-plan.md), and it stands: *use EDA to decide what **data** to add and
how to **process** it, not what features to compute.*

**✅ What EDA legitimately produces that the model or the sampler consumes:**

| Product | Consumed by | Manifest / sidecar column |
|---|---|---|
| **Per-file container forensics** — container, codec, bitrate, encoder tag, channel count, native sample rate, near-Nyquist shelf | A candidate legal side-signal, and the shortcut audit's input | `container`, `orig_sr`, `orig_channels` + new forensics columns |
| **Split keys** — speaker / artist / album / generator / duplicate group | `training.folds.build_folds` | `speaker_ref_id`, `source_name`, `artifact_family`, `dup_group`, `domain_key` |
| **Validity masks** — where in a file the labelled component is actually evidenced | the runtime crop sampler (V-A4) | `validity_mask_ref` |
| **Label-confidence tier** — exact / asserted / inferred, assigned from the *evidence checks*, not from the archive | per-tier loss and two-stage clean→noisy training (F1/F2) | `label_confidence` |
| **Per-source cleanliness score** | ★ `[Bengali.AI 2023]` B7 — condition augmentation strength on source cleanliness | `aug_strength` |
| **Paired real/fake indices** | the T3 twin construction and the E-A8 artifact gallery | `pair_id` |

All seven exist in `training.manifest.REQUIRED_COLUMNS` today and are **unfilled or filled at the
wrong granularity**. Measured on the 2,177-row smoke corpus: `pair_id` is null on 2,177 of 2,177
rows, `validity_mask_ref` on 2,177 of 2,177, `aug_strength` is 0.0 everywhere, `dup_group` is
populated on only the 176 rows the RIRS/MUSAN leak forced, and `label_confidence` carries two
values assigned **per source** (`reported` 1,477 / `exact` 700) rather than from any per-file
evidence. **Filling them is the deliverable.**

⚠️ **The one genuinely open feature question is the metadata leak**, flagged as A5 in
[architecture/09](../architecture/09-open-questions.md) and unresolved: per-file metadata is legal
under rule 2.4 and *may* separate REAL from FAKE almost for free — or be a pure CV mirage that
inverts on the real test set. EDA cannot settle it (we have 3 dummy files), but EDA is what builds
the extractor and measures how large the mirage is in *our* corpus. See [06](06-cross-pool.md).

---

## Two rules every task below obeys

**🔴 R1 — Measure at two planes: native, and post-chain.** Every spectral or level statistic is
computed twice: once on the file as the publisher shipped it, and once after the exact
`resample → downmix → normalize` path the model will see (16 kHz mono float32). *The comparison
between the two planes is the analysis*, not either number alone. Every question this project has
about band-limiting — E-A1's 16 kHz survivability, the resampler shelf, whether a generator
fingerprint lives above 8 kHz — is a difference between those planes, and a single-plane EDA
answers none of them.

**🔴 R2 — A filter is a distribution-shift decision; a transform is a symmetry obligation.**
([data/10 P2](../data/10-preprocessing-and-filtering.md)) We cannot filter the 1,200 test files, so
every training-side drop moves TRAIN away from TEST. Every task that recommends dropping rows must
state what the drop does to the pool's balance, and any transform it recommends must be applied to
the test path by the identical code. ⚠️ An asymmetric filter — one whose *rate* differs by label —
manufactures the exact shortcut this whole plan exists to prevent.
