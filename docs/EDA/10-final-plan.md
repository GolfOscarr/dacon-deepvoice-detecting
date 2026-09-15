# 10 — The final EDA plan: every task, its status, and what closes it

**Written 2026-09-16.** [09](09-next-steps.md) was a *remaining-work* plan written at one moment and
its eight steps are done — but it was never the full task inventory, and reporting "the EDA is
complete" from it was wrong. This page is the inventory: **all 43 declared tasks** from
[01](01-pool-a-real-voice.md)–[06](06-cross-pool.md), each with its status and the evidence for it.

Status was read off artifacts and doc sections, not recalled. Five states:

| | meaning |
|---|---|
| ✅ **answered** | a written answer with numbers exists |
| 📊 **data exists** | measured and on disk; no write-up. A `groupby` away, **no decode** |
| ▶️ **runnable** | not run, and nothing blocks it |
| ⛔ **blocked** | needs an input we do not hold |
| ⏭️ **Phase 2** | out of the EDA's scope by design |

**25 answered · 7 data-exists · 2 runnable · 7 blocked · 2 Phase 2 = 43.**

> **Steps 1 and 2 closed 2026-09-16.** B1, B4, D6, B6, E5 and X2 all have written answers.
> **Only step 3 remains** -- `_shared/roles.md` and the six 📊 census write-ups -- and it does
> not decode. C5 (step 4) is optional and [§2.4](#step-4--c5s-envelope-half-optional-and-probably-unnecessary) recommends closing it on reasoning.

⚠️ `A9`, `C8`, `D8`, `D9` are **result sections**, not tasks, and are not counted here.

---

## 1 — The inventory

### Pool A — real voice

| | task | status | evidence / blocker |
|---|---|---|---|
| A1 | Native-format census | 📊 | `files.parquet`, 100%. Never tabulated per source |
| A2 | Near-Nyquist rolloff profile | ✅ | [00 §4d](00-harness.md), `signal_bandwidth.parquet` |
| A3 | Speaker inventory / grouping atoms | ✅ | [06 X6b](06-cross-pool.md), `grouping_report.parquet` — zeroth 115, ljspeech declared 1 |
| A4 | Common Voice subset design | ⛔ | `common-voice-en/ko` not on disk |
| A5 | Duration census vs 4 s / 60 s | ✅ | [01 A9](01-pool-a-real-voice.md), `signal_duration.parquet` |
| A6 | Silence statistics vs pool B | 📊 | `signal_duration.parquet` has silence/lead/tail per partition; the **A-vs-B comparison the task asks for was never written**. Its point is the ASVspoof silence shortcut |
| A7 | Loudness, DC, clipping census | ✅ | [01 A9](01-pool-a-real-voice.md), `signal_level.parquet` |
| A8 | Pool-membership verification | ✅ | `signal_content.parquet`, G-EDA6 |

### Pool B — fake voice

| | task | status | evidence / blocker |
|---|---|---|---|
| B0 | CFAD registered | ✅ | [02 B0](02-pool-b-fake-voice.md) |
| **B1** | **WaveFake ↔ LJSpeech pairs** | ✅ | **Answered 2026-09-16**, [02 B1b](02-pool-b-fake-voice.md). 500 pairs, 4,000 files, 0 failures. The artifact **survives 16 kHz at 82–115%** and `melgan` ↔ `multi_band_melgan` correlate at **−0.186** |
| B2 | MLAAD generator / language inventory | 📊 | `group_key` holds 54 language×generator groups; not tabulated |
| B3 | Format census against pool A | 📊 | same data as A1 |
| B4 | Degenerate-output screen (F-S4) | ✅ | **2026-09-16**, [02 B4b](02-pool-b-fake-voice.md). No failure population: pool D flags **0.40%**, the lowest of any pool. ⚠️ the looping detector needs a decode |
| B5 | Duration / speaker-proxy census | ✅ | [02 B0c](02-pool-b-fake-voice.md) |
| B6 | Cross-generator separability | ✅ | **2026-09-16**, [02 B6b](02-pool-b-fake-voice.md). Pool B median best-AUC **0.904**; pool D's `mustango` at **1.000** on duration alone |
| B7 | CtrSVDD | ⛔ | source blocked (bonafide label only in train) |

### Pool C — real instrumental

| | task | status | evidence / blocker |
|---|---|---|---|
| C1 | Licence-allowlist intersection | ✅ | computed — G-EDA1 **fail**, 2,907 of 8,000 outside. ⚠️ the *decision* is open, the measurement is not |
| C2 | Silero VAD sweep | ✅ | [03 C2b](03-pool-c-real-instrumental.md) — 12.6%, and a floor |
| C3 | Artist / album grouping atoms | ⛔ | needs FMA `tracks.csv`; the metadata archive is unfetched. `eda keys` flags it every run |
| C4 | Genre census, C↔D matching | ⛔ | same archive |
| C5 | Clip boundary / duration morphology | ▶️ | duration half answered by [03 C8](03-pool-c-real-instrumental.md). The **onset/offset envelope classification** — hard-cut / fade / natural start — was never computed and needs a decode |
| C6 | Codec provenance, the mp3 question | 📊 | `files.parquet`; not tabulated |
| C7 | MUSAN `music/` as pool C | ✅ | structural in `configs/eda.yaml` |

### Pool D — fake instrumental

| | task | status | evidence / blocker |
|---|---|---|---|
| D1 | Archive inventory, real-half exclusion | ✅ | [08 §D1](08-real-run.md) — no MusicCaps real half |
| D2 | Per-model census | 📊 | per *source* exists; per generator directory not tabulated |
| D3 | Near-Nyquist per model, both planes | 📊 | `signal_bandwidth.parquet` is per source, not per generator |
| D4 | Family-count shortfall vs fold builder | ⏭️ | needs a built fold table |
| D5 | Caption / prompt census vs pool C | ⛔ | MusicCaps captions not fetched |
| D6 | Degenerate-generation screen (F-S4) | ✅ | **2026-09-16**, [04 D6b](04-pool-d-fake-instrumental.md). ⚠️ pool C flags **3.6×** more than D -- a drop would manufacture "clipping means REAL" |
| D7 | SONICS as a whole-file source | ✅ | `configs/eda.yaml`, [04 D7](04-pool-d-fake-instrumental.md) |

### Pool E — non-musical sound

| | task | status | evidence / blocker |
|---|---|---|---|
| E0 | The 2026-09-12 decisions | ✅ | [05 E0](05-pool-e-noise.md) and E0b–E0e |
| E1 | Cross-source duplicate sweep | ✅ | `duplicates.parquet`, [05 E1b](05-pool-e-noise.md) |
| E2 | Noise recordings vs impulse responses | ✅ | [05 E0d](05-pool-e-noise.md) — split by `name_glob`, IRs blocked. ⚠️ Closed by a **different method** than the spec's energy-decay classification, and the per-row classification was never computed. The question it existed for — "are there IRs in pool E?" — is settled |
| E3 | Duration census vs the sampler floor | ✅ | [05 E0e](05-pool-e-noise.md) — 58.4% |
| E4 | Independent-group inventory | ✅ | 10,722 groups, `grouping_report.parquet` |
| E5 | Cell-9 viability, `PRESENT=0` | ✅ | **2026-09-16**, [05 E5b](05-pool-e-noise.md). **11,901 viable (83.8%)**; **2,113** assert `PRESENT=0` while carrying voice |

### Cross-pool

| | task | status | evidence / blocker |
|---|---|---|---|
| X1 | The shortcut audit | ✅ | [06 X1b/X1c/X1d](06-cross-pool.md), `shortcut_audit.parquet` |
| X2 | The metadata-leak question | ✅ | **2026-09-16**, [06 X2b](06-cross-pool.md). `music_fake` **AUC 1.000** from metadata, **0.209 / 0.0000008** under an archive holdout. **Decision: neutralise** |
| X3 | Adversarial validation | ⛔ | needs the organizers' test audio |
| X4 | Dummy-file forensics | ⛔ | 🔴 needs `TEST_0000–0002.wav`. **Not on disk** — searched the corpus, the repo and S3. Marked *blocking*; it leaves `normalize` unparameterized in the render path |
| X5 | Composition confound, spec stream | ⏭️ | Phase 2 — X1 reads the corpus, this reads the spec stream |
| X6 | Metadata role assignment | ▶️ | keys done ([X6b](06-cross-pool.md)); the artifact it asks for — `_shared/roles.md`, every column assigned feature / split-key / leakage-risk — **does not exist** |
| X7 | The data memo | ✅ | [`data_memo.md`](data_memo.md) |

### ⚠️ Sub-parts not delivered inside otherwise-answered tasks

A ✅ above means the task's **question** has a written numeric answer. Five asked for something
extra that was not produced, none of which changes a conclusion:

| task | what is missing |
|---|---|
| A5 | the duration counts were not repeated *after silence trimming* |
| A7 | `lufs_integrated` is deliberately deferred (`eda/extract/level.py`); the between- vs within-source variance ratio was not computed |
| A8 | the PANNs half — VAD only ([09 §7](09-next-steps.md) records why PANNs was declined) |
| E4 | groups counted; `build_folds` feasibility not run — it needs the fold builder (Phase 2) |
| X1 | audited **per head**, not additionally per cell |

---

## 2 — The plan

Everything runnable, in one session. **The only decode was B1's, measured at 10.7 min** — steps 2 and 3 read
columns already on disk. C5 is the one exception and §2.4 says why it is optional.

### Step 1 — B1, the paired vocoder experiment ✅ **done 2026-09-16**

The only unconfounded real/fake comparison in the corpus: same speaker, same utterance, same source
recording, one vocoder apart. Everything else is confounded by speaker, text, archive or chain.

1. Choose ~500 LJSpeech utterance ids present in all 7 `ljspeech_*` WaveFake directories.
2. Decode real + 7 fakes for each — ~4,000 files, ~7 audio hours, **~15 min**.
3. Per vocoder: mean log-mel difference `E[mel(fake) − mel(real)]`, both planes.
4. Populate `pair_id` so G-EDA4 has an input when the fold table exists.

**Done when** a per-vocoder mel-difference table is saved and written into [02 B1](02-pool-b-fake-voice.md).
✅ **Met.** Measured at **0.70 audio-hours/min**, 10.7 min — the `1.33` in [09](09-next-steps.md)
is optimistic for a tier that runs S + V + C over one decode. ⚠️ Item 4 is only partly met:
`pair_id` is populated for the 500 drawn utterances, not for all 13,100 (see [02 B1b](02-pool-b-fake-voice.md)).

### Step 2 — the four free screens ✅ **done 2026-09-16**

All over columns already on disk. No decode.

| | what |
|---|---|
| **B4 / D6** | degenerate-output screen (F-S4): digital silence, full-scale clipping, single-frame files, near-constant spectra. `mel_bands_flat` and `clipping_ratio` are the inputs |
| **B6** | cross-generator separability over the S-tier columns, `group_key`-grouped |
| **E5** | cell-9 viability: how much of pool E can supply `PRESENT=0` at 4 s |
| **X2** | the metadata-leak question, against the chain-plane features |

### Step 3 — `_shared/roles.md`, and the census write-ups

X6's actual deliverable. Every column in `files.parquet` and `signal.parquet` assigned to exactly
one of **feature** / **split key** / **leakage risk**, with missingness and cardinality. Folding in
the six 📊 tasks — A1, B2, B3, C6, D2, D3 — as the per-source census tables they each asked for.

**Done when** `_shared/roles.md` exists and no column is unassigned.

### Step 4 — C5's envelope half: optional, and probably unnecessary

C5's remaining half — classify each pool-C clip as hard-cut / fade / natural start — needs a decode
of pool C's 2,660 files (~59 audio hours, **~45 min**) because the onset envelope is not among the
stored columns.

⚠️ **It is very likely already answered.** [03 C8](03-pool-c-real-instrumental.md) measured pool C
at **30.003 s median with p05 29.977** — FMA ships fixed-length excerpts, which are hard cuts by
construction. Run it only if the crop policy turns out to depend on the distinction; otherwise
record that reasoning and close it.

---

## 3 — Explicitly out of scope

Declared here so nobody re-opens them.

| | why |
|---|---|
| A4, B7, C3, C4, D5 | the input is not on disk. Each needs a fetch decision, not analysis |
| **X3, X4** | need the organizers' test audio, which we do not have. 🔴 X4 is marked *blocking* and stays open — `normalize` is unparameterized until it lands, and that is a **known, accepted gap** |
| D4, X5 | Phase 2 — they read a fold table and a spec stream, neither of which exists yet |
| G-EDA2 | structural. X1 reads the corpus on disk; every fix is render-time. It will report 1.000 whatever is fixed |
| G-EDA4, G-EDA7 | need a fold table and a filter respectively |

---

## 4 — After this, the EDA is closed

The exit artifacts are [`data_memo.md`](data_memo.md) and
[`RESULTS_FOR_ANALYSIS.md`](RESULTS_FOR_ANALYSIS.md), both of which must be updated with steps 1–3
before the EDA is declared done. They currently say the EDA is complete, which
[§1](#1--the-inventory) shows it is not.
