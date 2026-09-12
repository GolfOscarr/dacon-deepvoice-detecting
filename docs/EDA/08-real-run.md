# 08 — The Real Run: Order, Priority, and What Had to Be Fixed First

**Written 2026-09-11**, after verifying every claim below against the bucket and the code.
[07](07-order-and-gates.md) gives the phase order in the abstract; this is the executable version
against the 15 sources actually in S3.

> ⚠️ **Nothing in the store is extracted.** All ~340 GiB are publisher archives.
> `scripts/fetch_from_s3.py --extract` is the prerequisite for every line below.

---

## 1 — Four things that were wrong, and are now fixed

Each was found by checking a plan claim against the thing it assumed. None would have raised; all
four fail quietly.

| # | Defect | Consequence | Fix |
|---|---|---|---|
| 1 | **`extract_archives` was not recursive** (`payload.iterdir()`) | SONICS nests its ten zips under `payload/fake_songs/`, so the walk yielded the *directory*, `is_file()` was False, and **30.4 GiB extracted to nothing while the run reported success** | `rglob("*")`, with a test |
| 2 | **A file that looks like an archive and is not unpacked was skipped silently** | CompSpoof ships `development.tar.gz.part_aa..ae`; neither `is_tarfile` nor `is_zipfile` recognises a part, so `n` came back smaller and the caller saw a successful extraction of a corpus that is not there | warn, naming the files. Docs and CSVs are deliberately **not** warned about — a noisy warning trains the reader to ignore it |
| 3 | 🔴 **`configs/eda.yaml` named the allowlist column `track_id`; both shipped CSVs call it `id`** | A mis-named column rejects every row. G-EDA1 would have reported **FAIL on all of pool C** for a reason that is not a licence | column is `id`; a mis-named one now reports **`na`**, not `FAIL` |
| 4 | 🔴 **MTG-Jamendo's allowlist ids are *paths*** — `14/214.mp3` — while FMA's are bare integers (`2`) | Stem-matching compares `214` against `14/214.mp3`: **every Jamendo track rejected** | third key `relpath`; `int_stem` keeps FMA's zero-padded `000002.mp3` → `2` |

⚠️ **And one trap left in place because it cannot be fixed here.**
`fetch_from_s3.py --pool X` filters on the **bucket's** `acquisition.json`, which records the pool
each source was *acquired* as. SONICS is recorded there as `pool: D` — the assignment
[04](04-pool-d-fake-instrumental.md) corrects. So **`--pool D` pulls SONICS as well as
FakeMusicCaps**. Select by **name**, never by pool, until the acquisition records are reconciled.

---

## 2 — Priority 0: no audio at all (~20 MB, one afternoon)

Everything here unblocks a gate or a decision and costs almost nothing.

| What | Where | Unblocks |
|---|---|---|
| `fma_allow.csv`, `jamendo_allow.csv` | `s3://<bucket>/dacon-deepfake-detection/data/licences/` (6.6 + 4.6 MB) | **C1**, and G-EDA1, which reports `na` without them |
| `ctrsvdd/2024/payload/{train,dev}.txt` | 7.3 MB | Pool-B artifact families and speaker keys **without extracting 18.9 GiB** |
| `sonics/hf-2025/payload/*.csv` | `fake_songs.csv` 126 MB (already pulled) | Cell-8 domain key (`source × algorithm`), genre for **D5**, the authors' split |
| **X4 dummy files** `TEST_0000–0002.wav` | ⚠️ **From DACON — not in S3, and not in this repo.** Verified absent from both | `signal_chain.yaml`, and `normalize`, which is a **no-op until it lands** |

### 🔴 What Priority 0 actually found — 7.3 MB, before extracting 18.9 GiB

`ctrsvdd/payload/{train,dev}.txt` is the whole argument for doing this tier first.

```
128,029 rows  ->  109,313 deepfake   18,716 bonafide   (14.6% REAL sung voice)
114 speakers  ->  all 114 appear on BOTH sides of the label
8 attack families (A01-A08) x 6 source corpora, entangled:
    jvsmusic has only A06; m4singer has 7 of 8
```

**CtrSVDD is not a fake-voice source. It is a fake-voice *benchmark*, and 14.6% of it is real.**
Registered as a flat `pool: B` it would label **18,716 real singers `voice_fake = 1`** — the SONICS
defect in a second costume, and again invisible until someone asked.

⚠️ **And `exclude` cannot fix this one.** `train_set.zip` is flat —
`train_set/CtrSVDD_0053_T_0038470.flac` — so the label exists only in the `.txt`, keyed by utterance
id. `SourceSpec` matches on *paths*. This needs a label-file join, which does not exist yet, so
CtrSVDD is **`blocked:`** in [`configs/eda.yaml`](../../configs/eda.yaml): `enumerate_source` raises
with the reason rather than returning files. A source we cannot label correctly fails loudly instead
of being quietly deleted from the config and rediscovered later.

Three things to carry into the wave-2 work:

- 🔴 **All 114 speakers are on both sides of the label**, so a speaker-disjoint fold is *mandatory*.
  The same singer as real and as fake is memorisable **directly on the label axis**, and
  `speaker_ref_id` must come from the `.txt`, never from a path.
- **Attack family and source corpus are entangled**, so a family-disjoint split and a
  corpus-disjoint split are not independent choices.
- ℹ️ **The bonafide half is not waste.** Pool A holds *no* sung real voice at all
  ([data/04](../data/04-sources.md) lists MUSDB18-HQ stems and Opencpop; neither is acquired), so it
  fills a genuine gap once it can be labelled.

### ✅ D1 answered, and a second hazard found with it

FakeMusicCaps extracted first, so [04](04-pool-d-fake-instrumental.md)'s **blocking licence gate**
could be closed immediately. Its central directory holds six top-level entries:

```
MusicGen_medium 5521 · audioldm2 5521 · musicldm 5521 · mustango 5521 · stable_audio_open 5521
__MACOSX 27,605
```

✅ **No real-audio half.** MusicCaps' YouTube-derived originals are not in this archive, so nothing
needs excluding on licence grounds and `27,605 = 5 x 5,521` matches the acquisition record exactly.

🔴 **`__MACOSX` is the hazard.** It is the macOS resource-fork sidecar and it mirrors the tree as
`._name.wav` — **27,605 entries, exactly as many as the real audio**, all carrying the `.wav`
suffix. `enumerate_source` would have taken every one, and because *a decode failure is a row, not
an exception*, they would have landed as 27,605 `probe_ok = False` rows: **half the pool-D census,
silently, as garbage**. Excluded by name, because R2 makes every filter a visible decision rather
than a global rule.

⚠️ **Read the archive, not the extraction, while extraction is running.** A first look at the
unpacked tree showed four generators and suggested one was missing; the zip's own central directory
showed five, and `stable_audio_open` simply had not been written yet. Mid-flight directory listings
are not evidence.

### 🔴 MLAAD would have enumerated to zero, and nothing would have said so

**The store has two source shapes, and nothing recorded the difference.** Most sources ship as
archives and land under `--extract`. MLAAD does not: it is synced with its
`fake/<language>/<generator>/` tree **intact**, deliberately, because flattening it destroys the
generator-disjoint split axis — that was one of the five acquisition defects
([data/12](../data/12-acquisition-status.md)). So `extract_archives` found no archive,
**created an empty `interim/mlaad/v9/`**, and returned 0.

The directory *existed*, so no `FileNotFoundError` fired. `enumerate_source` returned an empty list,
and the best generator-diversity asset in the corpus — 535 generator directories — would have
contributed **nothing** to the census.

⚠️ **Accuracy matters here: the fetch log did say so** — `[xtr ] mlaad: 0 archive(s) unpacked`,
right between two sources reporting `1 archive(s) unpacked`. The signal existed and was not a
failure, which is the more interesting version of the problem: a line nobody reads is not a check.
Making it raise is what turns it into one.

Two fixes, and the second is the one that generalises:

* **`SourceSpec.stage`** — `interim` (unpacked) or `raw` (synced, root carries the `payload/`
  segment). MLAAD is `stage: raw` and enumerates 16,006 files.
* 🔴 **`enumerate_source` now raises when a source yields zero files.** A registered source that
  matches nothing is a misconfiguration — wrong root, wrong suffixes, everything excluded — and it
  is silent *because the directory exists*. This catches the class without anyone looking.

### The pattern across all three wave-1 findings

| Source | What was silent | Would have cost |
|---|---|---|
| `fakemusiccaps` | `__MACOSX` carries 27,605 `._*.wav` entries | Half the pool-D census as undecodable rows |
| `ctrsvdd` | 14.6% is `bonafide`, labelled only in a `.txt` | 18,716 real singers labelled `voice_fake = 1` |
| `mlaad` | Unarchived source, empty extraction directory | The whole of pool B's generator diversity |

**None raised. All three needed someone to ask.** That is the repo's standing lesson
([pipelines/05](../pipelines/05-invariants.md)) arriving in the data layer, and it is the argument
for running the M tier wave by wave rather than once at the end: each of these was visible within
minutes of the source landing, and each would have been buried in a single end-of-corpus run.

---

## 3 — Wave 1: chosen for coverage, not size (~56 GiB)

🔴 **The governing constraint.** [X1](06-cross-pool.md) is a *per-head population* statement, and
`shortcut_audit` reports `nan` — which `eda.gates` turns into **`na`** — for any head with one
class present. **No gate can fire until both sides of all four heads are on disk.** Within that,
music first: 0.27 plus most of 0.45, and it is the head with no literature.

| # | Source | GiB | Supplies | Why this rank |
|---|---|---|---|---|
| 1 | `fakemusiccaps` | 12.0 | **D** (`music_fake` +) | **D1 is a blocking licence gate** — the real-half exclusion must be decided before ingest, and G-EDA1 reports `na` until it is |
| 2 | `fma` | 7.5 | **C** (`music_fake` −) | Closes the music head. Needs Priority 0's allowlist |
| 3 | `mlaad` | 5.1 | **B** (`voice_fake` +) | 535 generator directories — by far the best diversity-per-GiB in the store |
| 4 | `ljspeech` | 3.0 | **A** (`voice_fake` −) | Tiny, and WaveFake's real twin: it sets up **B1** |
| 5 | `zeroth-korean` | 9.6 | **A** | Korean anchor; 105 speakers, the only real speaker-group source at this size |
| 6 | `sonics`, 2 of 10 parts | ~7 | **cell 8** | Exercises the whole-file path and the C↔cell-8 duration/bitrate confound. ⚠️ Needs fix #1 above |
| 7 | `musan` + `rirs-noises` | 11.5 | **E**, and **C** via `musan-music` | E1's known leak lives here |

**~56 GiB — 16% of the store — and it answers X1, E1, X6 and G-EDA1–5 across every head.**

⚠️ **What can invalidate it**: if C1's intersection leaves `fma_small` too thin, pool C cannot
support the music head and `mtg-jamendo` (5.1 GiB) moves into wave 1.

### Verified layouts

Confirmed by reading the archive headers out of S3 rather than by assumption, because a wrong
internal prefix makes `enumerate_source` raise:

```
musan.tar.gz        -> musan/music/fma/music-fma-0001.wav
rirs_noises.zip     -> RIRS_NOISES/pointsource_noises/noise-free-sound-0423.wav
```

🔴 Both confirm [05](05-pool-e-noise.md) E1's prediction **in the filenames themselves**: MUSAN's
music partition is literally `music/fma/`, and RIRS's point-source noise is literally
`noise-free-sound-*`. The `configs/eda.yaml` roots for both are correct as written.

---

## 4 — Wave 2: volume and diversity (~84 GiB)

| Source | GiB | What it adds |
|---|---|---|
| `wavefake` | 26.9 | **B1** — the corpus's only unconfounded real/fake comparison, and the highest-value analysis in the plan |
| `ctrsvdd` | 18.9 | Sung fake voice, and **B7**'s 16 kHz confound |
| `sonics`, remaining 8 parts | ~23 | The rest of cell 8 |
| `libritts-r` | 10.0 | Pool A speaker diversity |
| `mtg-jamendo` | 5.1 | Second real-music source, for source-disjoint folds |

---

## 5 — Deferred, deliberately

| Source | GiB | Why not yet |
|---|---|---|
| `common-voice-en` | 88.1 | Extract **`validated.tsv` only**. A4's subset decision is made from metadata, and taking it whole makes CV-en ~60% of pool A — the MLAAD-cap mistake in a second costume |
| `compspoof-v2` | 111.8 | Largest in the store, `mixed`, and its components need per-directory pool assignment first. ⚠️ Also the split-archive case in fix #2 — it will warn rather than unpack |

---

## 6 — The loop

```bash
V=/data/project/private/dacon-venvs/dacon311/bin/python
R=/data/project/private/dacon-corpus          # weka, 93 T free -- NOT / (456 G)

python3 scripts/fetch_from_s3.py fakemusiccaps fma mlaad ljspeech zeroth-korean \
    musan rirs-noises --dest $R/raw --extract $R/interim

$V -m eda probe        # M tier: ffprobe + sha256, 100%, resumable
$V -m eda consolidate  # parts -> <partition>/files.parquet
$V -m eda analyze      # X1 + E1 + grouping, then the gates; exit 1 on any fail
```

🔴 **Run the loop after *each* wave, not once at the end.** [06](06-cross-pool.md)'s standing rule
exists for exactly this: X1 and E1 are minutes against a table, and finding the corpus-identity
confound after wave 1 costs one re-plan instead of three.

⚠️ **Two operational facts.**
`--dest` and `--extract` both retain their output, so budget **2× per source** or delete the
payload after a successful extraction. And extract to `/data/project/private` — that is weka with
**93 T** free; `/` has 456 G and is not the place for this.

💰 Egress is billed per read. Pull once and keep it; `--verify-only` re-checks local files without
transferring anything.

---

## 6b — Three more defects, found by *running* the loop

[Section 1](#1--four-things-that-were-wrong-and-are-now-fixed) lists what checking the plan found.
These three only appeared once the verbs were pointed at a real, partially-fetched corpus.

| Defect | Symptom | Fix |
|---|---|---|
| **`eda probe` died on the first unfetched source** | `FileNotFoundError` on `common-voice-en`. Staged, wave-by-wave fetching is the whole shape of this document, so the tool could not do its actual job | Three states, one fatal: **absent** (normal mid-plan) and **blocked** (a recorded decision) are reported and skipped; **empty** stays fatal. Ends with a counted summary |
| **`eda consolidate` returned 1 on an unprobed partition** | `cell8` has no parts until SONICS lands | `NotProbed` is its own type, skipped and counted, distinct from a part that exists but never completed |
| 🔴 **A gate passed on an empty set** | `G-EDA1/allowlist/mtg-jamendo` reported **`all 0 rows permitted` → PASS**. Jamendo is not fetched; the gate checked nothing and passed | `na`, with the reason. The `na`-vs-`pass` confusion reappearing one level down — expect it wherever a new check meets an empty set |

---

## 7 — What this run cannot answer

Carried forward from [07 §4](07-order-and-gates.md) and unchanged: three dummy files make every
test-chain statement `n = 3`; a spectral screen is not an EER; X1 cannot prove the absence of a
shortcut it does not measure; and pool D is still one source of five families. Wave 1 adds one
more: **SONICS is the only whole-file source**, so every cell-8 statement is a statement about
Suno and Udio.
