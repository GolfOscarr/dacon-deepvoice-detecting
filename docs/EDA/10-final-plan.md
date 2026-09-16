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

**33 answered · 0 data-exists · 0 runnable · 7 blocked · 2 Phase 2 = 42.**

🔴 **Every runnable task is answered.** What remains is out of scope by construction (7 blocked,
2 Phase 2) plus the two exit artifacts.

> **Steps 1-4 closed 2026-09-16.** B1, B4, D6, B6, E5, X2, X6, A1, B2, B3, C6, D2, D3 and **C5**
> all have written answers.
>
> ⚠️ **C5 was nearly closed on reasoning instead of measured**, and measuring it found a
> **0.869-AUC** shortcut on the 0.27-weight music head that the reasoning could not have reached --
> see [§2.4](#step-4--c5s-envelope-half--done-2026-09-16--and-the-recommendation-below-was-wrong).
>
> What remains before the EDA is closed: updating [`data_memo.md`](data_memo.md) and
> [`RESULTS_FOR_ANALYSIS.md`](RESULTS_FOR_ANALYSIS.md), which both still claim it is complete.

⚠️ `A9`, `C8`, `D8`, `D9` are **result sections**, not tasks, and are not counted here.

---

## 1 — The inventory

### Pool A — real voice

| | task | status | evidence / blocker |
|---|---|---|---|
| A1 | Native-format census | ✅ | **2026-09-16**, [01 A1b](01-pool-a-real-voice.md). `fakemusiccaps` is the corpus's only `pcm_f32le` source -- one column identifies pool D |
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
| B2 | MLAAD generator / language inventory | ✅ | **2026-09-16**, [02 B2b](02-pool-b-fake-voice.md). 534 leaves, 54 languages, 205 generators. **Korean exists**: 12 generators, 359 files, 0.92 h |
| B3 | Format census against pool A | ✅ | **2026-09-16**, [01 A1b](01-pool-a-real-voice.md). Pools A and B are **format-indistinguishable** -- no container confound between real and fake voice |
| B4 | Degenerate-output screen (F-S4) | ✅ | **2026-09-16**, [02 B4b](02-pool-b-fake-voice.md). No failure population: pool D flags **0.40%**, the lowest of any pool. ⚠️ the looping detector needs a decode |
| B5 | Duration / speaker-proxy census | ✅ | [02 B0c](02-pool-b-fake-voice.md) |
| B6 | Cross-generator separability | ✅ | **2026-09-16**, [02 B6b](02-pool-b-fake-voice.md). Pool B median best-AUC **0.904**; pool D's `mustango` at **1.000** on duration alone |
| B7 | CtrSVDD | ⛔ | source blocked (bonafide label only in train) |

### Pool C — real instrumental

| | task | status | evidence / blocker |
|---|---|---|---|
| C1 | Licence-allowlist intersection | ✅ | **2026-09-16**, [03 C1b](03-pool-c-real-instrumental.md). The 2,907 are **23 DENY + 2,884 ND**, and ND was resolved usable by #417333 A5 the same day the allowlist was built. ⏭️ **stale, not restrictive** — deferred with a trigger |
| C2 | Silero VAD sweep | ✅ | [03 C2b](03-pool-c-real-instrumental.md) — 12.6%, and a floor |
| C3 | Artist / album grouping atoms | ⛔ | needs FMA `tracks.csv`; the metadata archive is unfetched. `eda keys` flags it every run |
| C4 | Genre census, C↔D matching | ⛔ | same archive |
| C5 | Clip boundary / duration morphology | ✅ | **2026-09-16**, [03 C5b](03-pool-c-real-instrumental.md). 🔴 the boundary shortcut is **real and inverted**: `onset_level_deficit_db` separates C from D at **AUC 0.869** |
| C6 | Codec provenance, the mp3 question | ✅ | **2026-09-16**, [01 A1b](01-pool-a-real-voice.md). Not "is it mp3": `sonics` 36.8 kbps against `fma` 265.7 kbps, non-overlapping |
| C7 | MUSAN `music/` as pool C | ✅ | structural in `configs/eda.yaml` |

### Pool D — fake instrumental

| | task | status | evidence / blocker |
|---|---|---|---|
| D1 | Archive inventory, real-half exclusion | ✅ | [08 §D1](08-real-run.md) — no MusicCaps real half |
| D2 | Per-model census | ✅ | **2026-09-16**, [04 D2b](04-pool-d-fake-instrumental.md). 231 generators. All 5 of pool D's have a **single duration each**, over all 27,605 files |
| D3 | Near-Nyquist per model, both planes | ✅ | **2026-09-16**, [04 D3b](04-pool-d-fake-instrumental.md). 🔴 **the chain is a no-op for pool D** -- both planes identical; `near_nyquist_ratio` spans 58x |
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
| X6 | Metadata role assignment | ✅ | **2026-09-16**, [06 X6b](06-cross-pool.md). `_shared/roles.md`: **118 columns, 0 unassigned**. `files.parquet` contributes **zero** features |
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

### Step 3 — `_shared/roles.md`, and the census write-ups ✅ **done 2026-09-16**

X6's actual deliverable. Every column in `files.parquet` and `signal.parquet` assigned to exactly
one of **feature** / **split key** / **leakage risk**, with missingness and cardinality. Folding in
the six 📊 tasks — A1, B2, B3, C6, D2, D3 — as the per-source census tables they each asked for.

**Done when** `_shared/roles.md` exists and no column is unassigned.
✅ **Met.** 118 columns, 0 unassigned, generated by `python -m eda.cli roles` and regenerable.
⚠️ It reports **five** roles, not three: `label` and `diagnostic` were added because 40 of the 118
columns fit none of X6's original three, and forcing them would have been false ([06 X6b](06-cross-pool.md)).

### Step 4 — C5's envelope half ✅ **done 2026-09-16 — and the recommendation below was wrong**

⚠️ **This section previously recommended closing C5 on reasoning rather than spending the decode**,
on the grounds that [03 C8](03-pool-c-real-instrumental.md) had pool C at 30.003 s median with p05
29.977 — FMA ships fixed-length excerpts, which are hard cuts by construction. **That reasoning was
wrong**, and it is left here rather than deleted because the way it was wrong is the lesson.

C5's hypothesis is a *comparison*: real music cut from track centres against generated music that
begins from silence. The duration evidence speaks only to pool C's half of it and says nothing
whatever about pool D's onset morphology — so the inference covered one side of a two-sided claim.

**Measured** ([03 C5b](03-pool-c-real-instrumental.md)): the shortcut is real, it runs **opposite**
to the predicted direction — pool D opens at its own level and pool C opens 43 dB below it — and
`onset_level_deficit_db` separates the two at **AUC 0.869** against a 0.60 gate. That is a second
shortcut the size of duration's 0.852, on the same 0.27-weight head, and independent of it.

The whole corpus was measured rather than pool C alone, because the crop policy C5 proposes is a
*transform* and R2 makes it a symmetry obligation. Cost: 58,885 files, 1 failure, **~5 min** at
76 audio-hours/min — the 45-minute estimate above assumed the S tier's rate, and an envelope pass
skips the spectral, VAD and vector work entirely.

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

## 4 — ✅ The EDA is closed, 2026-09-16

**33 of 33 runnable tasks answered.** What remains is out of scope by construction: 7 blocked on
inputs we do not hold, 2 deferred to Phase 2 (§3).

The exit artifacts are [`data_memo.md`](data_memo.md) and
[`RESULTS_FOR_ANALYSIS.md`](RESULTS_FOR_ANALYSIS.md), and both have been updated with steps 1-4.

⚠️ **Two decisions are measured and deliberately left open**, because they are calls rather than
computations — they are not EDA work and closing the EDA does not close them:

| | the number |
|---|---|
| `G-EDA1/allowlist/fma` | **23** genuinely DENY; the other **2,884** are ND and already usable ([03 C1b](03-pool-c-real-instrumental.md)). ⏭️ deferred: the gain is volume, and volume is not pool C's constraint |
| `G-EDA6` reassignment | **2,442** rows contradicting their asserted components, of which **2,113** are pool-E files asserting `VOICE_PRESENT = 0` while carrying speech |

🔴 **`eda gates` still exits non-zero and that is correct.** `G-EDA2` is structural and cannot go
green in the EDA at all (§3); `G-EDA3` reports a genuine count, not a gap; the other two are the
decisions above.
