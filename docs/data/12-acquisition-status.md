# 12 — Acquisition Status

What is in the S3 raw store, what is not, and exactly what each missing thing needs.

Machine-readable twin: [`scripts/sources.yaml`](../../scripts/sources.yaml), which is what
[`scripts/fetch_to_s3.py`](../../scripts/fetch_to_s3.py) actually enforces. This page is the
human view of the same table — if the two disagree, the YAML is right and this page is stale.

**Updated 2026-09-10.** 7 sources acquired, 122.6 GiB in S3; MLAAD in flight.

```
s3://<bucket>/dacon-deepfake-detection/data/raw/<source>/<version>/
    payload/            bytes exactly as the publisher served them, not unpacked
    _meta/acquisition.json    provenance: licence, verdict, sha256 per artifact
    _meta/checksums.sha256
    _meta/DONE          idempotence marker; a re-run skips the source
```

The bucket and prefix are configurable — `--bucket`/`--prefix` on either script, or the
`DACON_S3_BUCKET` / `DACON_S3_PREFIX` environment variables — so nothing below assumes a
particular account or region.

ℹ️ **Storage and compute here are personal resources**, provisioned and paid for by the
participant for this competition. Nothing in this repository depends on any employer's
infrastructure.

⚠️ **Check that the bucket's region matches wherever training runs.** Ingress is free, so the
initial write is cheap, but every read back out is billed as egress, and training reads the corpus
repeatedly. Retargeting is one flag — `--bucket`/`--prefix`, or `DACON_S3_BUCKET`/`DACON_S3_PREFIX`
— and is far cheaper to do before the corpus grows than after.

---

## 1 — The gate, in one paragraph

🔴 DACON requires submitting the actual training files and stated that data whose licence
restricts 제3자 제공 **cannot be used as competition training data at all** (#417280). So a
licence verdict is not paperwork that follows a download, it is the precondition for one. The
fetcher refuses any source whose `redistribution` verdict is not `ok`, and refuses to accept an
`ok` verdict that is not marked `verified_at_origin` — reported-from-a-secondary-source is how
[11](11-source-inventory.md) describes every row it lists, and that is explicitly not
verification. Overriding the gate takes `--i-have-verified <name>`, which names the source so it
cannot be set once and forgotten.

---

## 2 — Acquired

Licence read on the dataset's own page on the date shown, then fetched. Every row has an
`acquisition.json` carrying the verdict and a sha256 per artifact.

| Source | Pool | Size | Licence | Verified at origin |
|---|---|---|---|---|
| **common-voice-en** `cv-26.0` | A | 88.1 GB | CC0-1.0 | MDC API; **sha256 matched the publisher's `6809228e…`** |
| **musan** `openslr-17` | E | 10.3 GiB | CC BY 4.0 | [openslr.org/17](https://www.openslr.org/17/) |
| **libritts-r** `openslr-141` | A | 9.8 GiB | CC BY 4.0 | [openslr.org/141](https://www.openslr.org/141/) |
| **zeroth-korean** `openslr-40` | A | 9.6 GiB | CC BY 4.0 | [openslr.org/40](https://www.openslr.org/40/) |
| **ljspeech** `mdc-1.1` | A | 3.0 GB | CC0-1.0 | MDC API |
| **rirs-noises** `openslr-28` | E | 1.2 GiB | Apache-2.0 | [openslr.org/28](https://www.openslr.org/28/) |
| **common-voice-ko** `cv-26.0` | A | 0.2 GB | CC0-1.0 | MDC API; sha256 matched |
| | | **122.6 GiB** | | |

**Pool A and Pool E are covered.** Korean (Zeroth + CV-ko), English read (LibriTTS-R, LJSpeech),
English crowd-sourced (CV-en), plus noise and room impulse responses. That is enough to build the
presence heads and the augmentation chain.

⬜ **Pools B, C and D are still empty.** No fake voice, no real music, no fake music.

Two things to remember when unpacking:

- ⚠️ **Never use MUSAN's `music` partition as noise** ([06](06-augmentation-spec.md)) — it would
  put real music into the noise role and corrupt the music-present labels.
- 🔴 **Record `source_name` at speaker granularity as you unpack**, not after. The fold builder
  needs speaker-disjoint splits and [08](08-build-plan.md) records this as the one Phase B
  decision with no second chance. ⚠️ Common Voice English is 88 GB against a 20 h budget, so the
  `client_id` subsetting is a real decision waiting at the unpack stage, not a solved one.

### In flight

**mlaad** `v9` — capped, see [§4b](#4b--the-mlaad-cap-a-corpus-decision-not-a-download-setting).

---

## 2b — What the transfers cost, and the five defects they exposed

🔴 **None of these was found by a test going red.** All five surfaced only because a transfer was
being watched while it ran, which is the same pattern
[docs/pipelines/05](../pipelines/05-invariants.md) records for the training loop.

| Defect | Symptom | Fix |
|---|---|---|
| **`curl --retry` with `-C -`** | Common Voice English went **backwards, 74 GB → 45 GB**. `-C -` fixes its resume offset when curl starts, so an internal retry re-requests from that stale offset and discards everything since | curl no longer retries; `download_with_resume` starts a fresh curl per attempt |
| **Orphaned child process** | `pkill -f "fetch_to_s3.py --all"` matched only the Python parent. Its curl survived at `PPID=1` and kept writing into the same path as the replacement run — **two writers, different offsets** | kill children first; every monitoring tick now counts `PPID=1` curls |
| **Corruption invisible to size** | The two writers produced **118.3 GB against an expected 94.6 GB — 125%**, unresumable because `-C -` then asks past EOF | oversized partial ⇒ delete and restart; short download ⇒ fatal; **publisher sha256 verified** where one exists |
| **Partial discarded on failure** | The `finally` block deleted staging on every failure, so a transfer that died at 66% restarted from zero — four times | keep staging on failure, clean only on success |
| **HF resolved one URL at a time** | MLAAD planned **99,411 artifacts**, flattening `fake/am/Edge-TTS/x.wav` into a basename — destroying the generator directory, which is the generator-disjoint split axis | mirror the repo with structure intact, upload with one `aws s3 sync` |

⚠️ **The HF API's `siblings` field undercounts.** It reports 99,411 files for MLAAD; the repo holds
**534,539**. `list_repo_files` is the number that matches reality, and the difference was 174 GB
against an assumed 30.

## 3 — Queued, cleared, not yet fetched

**22 sources scheduled, 367 GB.** All licence-verified at origin; the fetcher runs them in
registry order and writes a `DONE` marker per source, so an interrupted run resumes for free.

| Source | Pool | Size | Licence |
|---|---|---|---|
| **mlaad** v9 | B | ~5 GB capped | CC BY-NC 4.0 — 175 TTS families, the generator-diversity asset |
| **compspoof-v2** | mixed | — | CC BY-NC 4.0 — our exact label structure, cells 6/7 |
| **musdb18-hq** | mixed | 21.1 GB | educational/NC — isolated stems, cells 5–8 |
| **partialspoof** | B | 9.2 GB | CC BY 4.0 — segment labels, trains `frame_max` |
| **asvspoof2021-la** | B | 7.2 GB | ODC-BY — the telephony asset |
| **wavefake** | B | 26.9 GB | CC BY-SA 4.0 — matched T3 twins against our LJSpeech |
| **cfad** | B | 33.2 GB | CC BY 4.0 |
| **timit-tts** | B | 6.7 GB | CC BY 4.0 |
| **rfp** | B | 37.6 GB | CC BY-NC 4.0 |
| **codecfake** | B | 32.1 GB | CC BY-NC-ND — ND, see [§6](#6---nd-unblocked-2026-09-08--the-biggest-single-change) |
| **st-codecfake** | B | 39.3 GB | CC BY-NC-ND |
| **ctrsvdd** | B | 31.6 GB | CC BY-NC-ND — 260 h sung fake at 16 kHz |
| **scenefake** | B | 5.8 GB | CC BY-NC-ND |
| **mtg-jamendo** | C | 5.4 GB/shard | per-track, allowlist only |
| **fma** | C | 7.2 GB | per-track, allowlist only |

### Deferred — cleared, deliberately not scheduled

| Source | Size | Why |
|---|---|---|
| **asvspoof5** | 132.6 GB | Cleanly ODC-BY, and 142 GB against a ~70 h Pool B. Same DOSS reasoning as the MLAAD cap: fetch by name after deciding which subset earns its place |
| **vctk** | 10.9 GB | Deferred on **throughput, not rights**. DataShare throttled it 1.1 MB/s → 73 KB/s mid-transfer, a 14 h tail that was blocking Pool B crown jewels. Licence is clean; the 7.3 GB partial is preserved and `fetch_to_s3.py vctk` resumes |

`defer` exists to keep two different questions apart: whether we are *allowed* to take a source,
and whether we *want* it now. Folding the second into the licence verdict would have marked
ASVspoof 5 `review`, which is false.

---

## 4 — Per-track licence pass — done

FMA and MTG-Jamendo are single downloads whose **audio is not under a single licence**. Both are
now partitioned by [`scripts/filter_track_licences.py`](../../scripts/filter_track_licences.py),
and the resulting lists live at `s3://<bucket>/dacon-deepfake-detection/data/licences/`.

| Corpus | Tracks | ✅ allow | ⚠️ NoDerivatives | ❌ deny |
|---|---|---|---|---|
| **FMA** | 109,727 | 62,121 (56.6%) | 44,652 (40.7%) | 2,954 (2.7%) |
| **MTG-Jamendo** | 55,701 | 36,728 (65.9%) | 18,887 (33.9%) | 86 (0.2%) |

The partition follows the rule text rather than the feel of the licence: CC0, public domain, BY,
BY-SA, BY-NC and BY-NC-SA all **allow**, because none restricts 제3자 제공. FMA's
*"FMA-Limited: Download Only"* (2,619 tracks) is the clear **deny**, and **unknown defaults to
deny** — an unrecognised licence string is not evidence of permission.

🔴 **The ND third is held out, not merged either way.** ND permits provision of the *unmodified*
file, but our pipeline's entire design is augmentation and composition
([06](06-augmentation-spec.md)), and ND plausibly bars the derivative we would actually train on.
That is the same legal read CtrSVDD is waiting on — one decision unlocks ~63,000 tracks across
both corpora.

⚠️ **Two traps found doing this.** MTG-Jamendo's licence is **not in `raw.tsv`** — that carries
only TRACK_ID / ARTIST_ID / ALBUM_ID / PATH / DURATION / TAGS, so a filter built from the TSV
would silently pass every track. It lives in `audio_licenses.txt` at the repo **root**, not under
`data/` with the TSVs. And FMA's metadata zip is bzip2-compressed, which system `unzip` refuses
(`need PK compat. v4.6`) while Python's `zipfile` reads it fine.

**The archive being cleared does not clear a track.** Only ids in `fma_allow.csv` /
`jamendo_allow.csv` may enter the corpus, and `fma_small` is an 8,000-track subset, so intersect
the allow list with whatever subset is actually downloaded.

---

## 4b — The MLAAD cap: a corpus decision, not a download setting

MLAAD is the only source so far where *how much to take* and *how to take it* are the same
question, and getting it wrong is silent.

```
[list] mueller91/MLAAD ...
[cap ] 534539 files in 535 dirs -> 16025 selected (cap=30/dir, seed=0)
```

MLAAD v9 is **1002.9 h over 534,539 files ⇒ ~6.75 s/file**, and [04](04-sources.md) budgets
**30 h** from it. So the cap is arithmetic, not taste:

| cap/dir | files | audio | verdict |
|---|---|---|---|
| none | 534,539 | 1002.9 h | ~174 GB, ~32 h of wall clock |
| 300 | 160,205 | 301 h | 🔴 10× the budget — **what was shipped first** |
| **30** | **16,025** | **30.1 h** | ✅ current |

🔴 **cap=300 would have made MLAAD ~80% of a 70 h Pool B**, which is precisely the domain
imbalance DOSS says decides the result — 0.2k h domain-balanced beat 6.4k h naive (2.77% vs 3.29%
EER). The first cap avoided the "don't take all 174 GB" mistake and walked straight into its
second-order version, because the hours arithmetic had not been done.

**The cap is per leaf directory** because MLAAD lays out `fake/<language>/<generator>/<file>` — the
leaf *is* the generator, i.e. the DOSS domain key. All 535 `language × generator` domains survive;
none is taken whole. A global cap would have kept all of the first few generators and none of the
rest, the opposite of what generator diversity needs.

⚠️ **The selection is part of the corpus definition, not a fetch detail.** It is seeded and written
to `_meta/selection.json`, because [#417333 A6](../competition/05-talkboard-qa.md) makes
reproducibility from *originals + code + seed* the thing the 2nd-stage submission rests on. Change
the cap or the seed and it is a different corpus.

`test_mlaad_cap_matches_the_documented_hour_budget` fails the build if the cap drifts more than
~1.5× off 30 h.

---

## 5 — Corrections to [11](11-source-inventory.md), found while wiring the fetcher

Both were "reported from a secondary source" in the inventory, and both were wrong in the
direction that matters.

### MUSDB18-HQ — cleared for use

🔴 The inventory calls this "access request required" and the longest pole. **Both halves were
wrong**, and the second one was my own over-caution rather than the inventory's.

There is nobody to ask: the Zenodo file serves anonymously (22.7 GB, HTTP 200). And the licence,
though awkward-looking, clears — because the test is the **rule text**, not how restrictive the
licence feels. #417280 answer 1 opens with the sentence that settles it:

> **운영진 검증 목적의 제출은 일반적인 공개 혹은 재배포와는 구분됩니다.** 다만, 라이선스상
> **제3자 제공 자체가 제한되는 데이터라면** 대회 학습 데이터로 사용할 수 없습니다.

Submitting to DACON for verification is *not* redistribution. The one bar is a licence that
restricts **제3자 제공 itself** — and MUSDB18-HQ has no such clause. It restricts *commercial
use*, which is a different thing, and [#417212](../competition/05-talkboard-qa.md) answers that exact shape
(CC-BY-NC, CC-BY-NC-SA) with **네 사용 가능합니다**. [#417310](../competition/05-talkboard-qa.md) adds that
external public data keeps its own licence and is not swallowed by the winning-work exclusivity
clause.

For the record, the archive is four upstream grants in one: 100 tracks from DSD100 (via the
'Mixing Secrets' library), 46 from MedleyDB (CC BY-NC-SA 4.0), 2 from Native Instruments, 2 from
The Easton Ellises (CC BY-NC-SA 3.0). That is a **compliance duty, not a bar** — the components
with no stated licence are published by the dataset's own authors as part of a public corpus, and
nothing in them restricts provision.

⚠️ Two duties follow. Attribute MUSDB18-HQ and its upstreams in the 데이터 구성 보고서. And note
that the 46 MedleyDB tracks are **SA**: [docs/competition/05](../competition/05-talkboard-qa.md)
flags ShareAlike propagating into the model deliverable as a real tension, so prefer them as
training *input* over shipping derived stems.

**The lesson, recorded because this row was wrong twice in opposite directions:** clear a source
against the rule text, not against how restrictive its licence sounds. `test_musdb18_was_cleared_against_the_rule_text` pins it.

### VCTK — no click-through at all, and now acquired

The 4 KB HTML page seen earlier was a **wrong bitstream path**, not an interstitial.
`https://datashare.ed.ac.uk/download/DS_10283_3443.zip` serves anonymously (11.7 GB), and the
record's own `license_text.txt` is the full **CC BY 4.0** legalcode. Verified at origin
2026-09-08 and promoted to `ready`.

The same bulk-URL pattern works for **ASVspoof 2019** (`DS_10283_3336`, 25.3 GB) — added to the
registry as `needs_review`, because its licence is not exposed as a bitstream and ships inside
the zip. The honest order there is download, read the LICENSE, then record the verdict; reading
a licence is not using the data.

---

## 6 — 🔴 ND unblocked 2026-09-08 — the biggest single change

[#417333 A5](../competition/05-talkboard-qa.md) answered the question these rows were waiting on:

> **말씀하신 방식도 가능하며**, 이경우에는 **원본 파일과 코드로 재현될 수 있어야합니다.**

ND data may be used **and augmented**, provided the result is reproducible from 원본 파일 + 코드.
That is what our pipeline already does — sampling is separate from rendering and composed output
is never stored ([`training/`](../../training/AGENTS.md)) — so the grant matches the design rather
than constraining it.

| Source | Pool | Size | Why it matters |
|---|---|---|---|
| **codecfake** | B | 32.1 GB | Inventory crown jewel #4. Modern TTS generates directly from neural codecs, skipping the vocoder; vocoder-trained detectors are structurally blind to that family |
| **st-codecfake** | B | 39.3 GB | Source-tracing companion |
| **ctrsvdd** | B | 31.6 GB | 260 h of sung fake **already at 16 kHz**, 14 SVS/SVC methods. Without it every song's vocal reads the same way |
| **scenefake** | B | 5.8 GB | Tampers the acoustic *scene* while leaving speech genuine — the one public analogue of our real-voice + fake-background cells |
| FMA + MTG-Jamendo ND buckets | C | — | ~63,000 tracks previously held out |

⚠️ **Usable only in the granted form.** Train from the **originals**, augment at training time,
ship originals + code + seed. Never redistribute a derivative. NC still applies on top: attribute
in the 데이터 구성 보고서. `test_nd_sources_record_the_reproduce_from_original_duty` fails the
build if an ND row is cleared without recording that condition.

## 6b — Still blocked

| Source | Verdict | Why |
|---|---|---|
| **ai-hub** | ❌ | Usage policy bars third-party provision without approval. Unblocks only if the NIA inquiry succeeds |
| **odss** | ❌ | Zenodo record states no licence at all. Unknown defaults to deny — an absent grant is not evidence of permission |
| **scraped-audio** | ❌ | No licence permits provision to DACON, and scraped clips carry no component-level ground truth, so they cannot produce `VOICE_FAKE` vs `MUSIC_FAKE` labels even in principle |
| **commercial generative APIs** | ❌ | ToS forbid using outputs to build detection models. [#417333 A7](../competition/05-talkboard-qa.md) confirms DACON will not adjudicate — the call is ours |

## 7 — The G2 backlog, and one outage

**⚠️ Zenodo was unreachable on 2026-09-08 ~20:00 KST** — HTTP 504 then connection failure, on
records that had answered 200 twenty minutes earlier, while other hosts stayed fine. It is a
Zenodo-side outage, not a credential problem. Two consequences:

- The G2 licence audit of **Codecfake, ASVspoof 2021, PartialSpoof, WaveFake, SceneFake,
  ASVspoof 5, CFAD, RFP** could not run. All are Zenodo-hosted and need no credential, so this is
  purely a matter of retrying when Zenodo is back.
- **MUSDB18-HQ's download will fail while the outage lasts.** That is handled rather than fatal:
  the fetcher reports `[FAIL]` for that source, continues with the rest, and writes **no `DONE`
  marker** — so a later re-run picks it up with nothing lost.

**ASVspoof 2019** (`DS_10283_3336`) is on Edinburgh DataShare, not Zenodo, and serves anonymously
at 25.3 GB. It stays `needs_review` for a different reason: its licence is not exposed as a
bitstream — `LICENSE.txt`, `license.txt`, `LICENCE.txt` and `README.txt` all return the DSpace SPA
shell — so it ships inside the zip. The honest order is download, read the LICENSE, record the
verdict. Reading a licence is not using the data.

## 7b — Not a download at all

Roughly half of Pools B and D are **self-generated** ([05](05-synthesis-plan.md)) — 8–12 open TTS
families, VC/SVC, SVS, vocoder and codec resynthesis of Pool A, and ACE-Step for fake music. That
needs **GPU compute, not credentials**; `HF_TOKEN` already covers pulling the open weights.

## 8 — Running it

### Acquiring into S3 (this machine)

```bash
python3 scripts/fetch_to_s3.py --list              # the registry and its verdicts
python3 scripts/fetch_to_s3.py --plan --all        # resolve URLs, upload nothing
python3 scripts/fetch_to_s3.py --all               # every scheduled source
python3 scripts/fetch_to_s3.py vctk asvspoof5      # a deferred source, by name
```

Peak local disk is **one artifact**, not the corpus: each is staged, hashed, uploaded, deleted.
Downloads resume via HTTP Range across whole-transfer failures, and a source whose `DONE` marker is
already in S3 is skipped, so a re-run after an interruption costs only what it did not finish.

⚠️ **Stopping it: kill the children first.** `pkill -f "fetch_to_s3.py --all"` matches only the
Python parent and leaves its `curl` orphaned at `PPID=1`, still writing into the same path as
whatever replaces it — which is how the 118 GB corruption happened.

```bash
PID=$(pgrep -f "scripts/fetch_to_s3.py --all"); pkill -P $PID; kill $PID
```

### Pulling onto the training machine

[`scripts/fetch_from_s3.py`](../../scripts/fetch_from_s3.py) is the consumer side. It needs **AWS
credentials and nothing else** — not a checkout of this repo at a matching revision — because it is
driven by what the bucket actually holds rather than by `sources.yaml`.

```bash
python3 scripts/fetch_from_s3.py --list                              # what is in the store
python3 scripts/fetch_from_s3.py --all    --dest /data/raw           # everything complete
python3 scripts/fetch_from_s3.py --pool B --dest /data/raw           # one pool
python3 scripts/fetch_from_s3.py mlaad musdb18-hq --dest /data/raw
python3 scripts/fetch_from_s3.py --all --dest /data/raw --extract /data/interim
python3 scripts/fetch_from_s3.py --all --dest /data/raw --verify-only   # no transfer
```

Three behaviours worth knowing:

🔴 **Verification is the default, not a flag.** Every file is re-hashed against the source's
`_meta/checksums.sha256` and a mismatch is a failure with exit code 1, not a warning. This is the
lesson from acquisition: a two-writer collision once produced a file 125% of its true size, and on
a different day would have produced one of exactly the right length with the wrong contents.
**Size is not evidence.**

⚠️ **A source with no `_meta/DONE` is skipped.** The marker is written last, after payload and
checksums, so its absence means the upload was interrupted and the payload may be short.
`--include-incomplete` takes it anyway and says so. As of 2026-09-10 `compspoof-v2` is exactly this
case — 3 stray files from a failed run.

💰 **Egress is the real cost.** Pull once and keep it; `--verify-only` re-checks local files
without transferring anything. If the bucket and the training machine are in different regions,
co-locating them first is worth more than any tuning in this script.

Tests: [`tests/test_fetch_sources.py`](../../tests/test_fetch_sources.py) (42, the acquisition side)
and [`tests/test_fetch_from_s3.py`](../../tests/test_fetch_from_s3.py) (12, the consumer side) —
each invariant paired with the mutation that breaks it.
