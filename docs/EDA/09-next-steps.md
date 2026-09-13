# 09 — Next steps

Written 2026-09-13, after wave 1 + CFAD. This is the **plan of record** for what remains; the state
it starts from is measured, not remembered, and every command below was run verbatim before this
file was committed.

Branch is **`feat/eda`**. `main` sits at `origin/main` and the EDA lands on it as a PR.

---

## 0 — Where the EDA actually is

```bash
V=/data/project/private/dacon-venvs/dacon311/bin/python
$V -m eda.cli gates            # re-read the saved analyses; no decode
```

**M tier: complete.** 264,085 files, 19 runnable sources, 6 partitions, `probe_ok` 264,082 — the 3
failures are known-unreadable files in pool C, recorded rather than dropped.

| A | B | C | D | E | cell8 | mixed |
|--:|--:|--:|--:|--:|--:|--:|
| 74,846 | 89,706 | 8,660 | 27,605 | 14,194 | 49,074 | not probed |

**S tier: pool E only** — 14,194 rows of 264,085, **5.4%**. Every S-tier claim in these documents is
a pool-E claim until step 2 lands.

**Gates**: `G-EDA5` pass · `G-EDA1/exclude` ×4 pass · `G-EDA1/allowlist/fma` fail (a licence
partition, not a defect) · `G-EDA1/allowlist/mtg-jamendo` `na` (unfetched) · `G-EDA2` fail
(structural — see below) · `G-EDA3` fail, 7 of 13 sources · `G-EDA4`, `G-EDA6`, `G-EDA7` `na`.

**6 sources blocked**, each with a measured reason: `ctrsvdd`, `rirs-pointsource`,
`rirs-isotropic-rir`, `cfad-codec`, `cfad-noisy`, `codecfake`.

⚠️ **`G-EDA2` cannot be made green by the EDA and must not be chased.** X1 reads `files.parquet` —
the corpus as it sits on disk — while every fix is a *render-time* transform. It will report
1.000 on `music_fake` no matter what is fixed. Closure belongs to `audit_specs` over the spec
stream (X5, Phase 2).

---

## 1 — Publisher keys into the census ⚠️ do this first

**What.** Join each publisher's own group key into `files.parquet` as a column, starting with
SONICS' `algorithm` from `payload/fake_songs.csv`.

**Why it is first, and not a tidy-up.** Two reasons, and the second is the load-bearing one.

It is what `G-EDA3` has asked for in every run — *"no depth reaches the floor, needs the publisher's
own key"* — for **7 of 13** sources: `fakemusiccaps`, `ljspeech`, `musan-music`, `musan-noise`,
`musan-speech`, `rirs-isotropic-noise`, `sonics`.

And it decides **the draw** for the three sampled partitions. `sample.stratify_by` is
`("source_name",)`, so at 2,000 per source:

| partition | source | rows | real groups inside it |
|---|---|--:|--:|
| B | `cfad-fake` | 73,700 | **11 generators** |
| B | `mlaad` | 16,006 | 54 language×generator |
| A | `cfad-real` | 38,600 | **6 corpora** |
| A | `zeroth-korean` | 22,720 | 115 path groups |

2,000 drawn from `cfad-fake` as one stratum leaves eleven generators in whatever proportion the draw
happened to pick. Widening `stratify_by` to the group key fixes that — and it **must** be decided
before step 2, because the draw is part of the corpus definition: redrawing after a decode
invalidates every S-tier number taken from the old draw, and `Sample.fingerprint` now refuses to let
that happen silently.

⚠️ This argument does **not** apply to `cell8` or `D`. Both are measured in full, so `stratify_by`
does not touch them. An earlier draft of this plan justified step 1's ordering by cell8's draw;
the full-coverage decision below made that reason obsolete, and the reason above replaces it.

**Verified already**: the SONICS join matches **49,074 of 49,074** rows, with the CSV's own
`duration` agreeing with ffprobe to **0.000 s**. It yields 5 generator groups —
chirp-v3.5 (19,057), udio-120s (18,745), udio-30s (4,903), chirp-v3 (4,285),
chirp-v2-xxl-alpha (2,084) — against the 1 that path depth finds.

**Also check for a key**: MLAAD (generator is already in the path), FMA (`genre`, `artist` in its
metadata), MTG-Jamendo (when fetched). CFAD needs nothing — its key *is* the path.

**Done when** `eda analyze`'s grouping report stops saying *"needs the publisher's own key"* for
SONICS, `G-EDA3` counts 5 groups there instead of 1, and a decision is recorded on whether
`stratify_by` widens to the new column.

⚠️ `eda sample` and `eda signal` both skip a partition with no `files.parquet` and exit 0 —
`mixed` is unprobed and will print one line to stderr on every run. That is the expected state, not
a failure.

---

## 2 — The S tier over everything else

```bash
$V -m eda.cli sample          # draws every partition that has none
$V -m eda.cli signal          # ~139 min, resumable at 500-file parts
```

**89,765 files to decode, about 2 h 20 min.** Measured rate: pool E's 14,102-row pass took
**21 m 04 s** on 32 threads — **11.2 files/s**. The table below is computed at a deliberately
conservative 10.7, so it over-estimates by ~5%. `user + sys` was 41 minutes against 21 of wall
clock, so the GIL holds the speedup near 2× and this is I/O- and decode-bound, not parallel-bound.

| partition | population | to decode | how | est |
|---|--:|--:|---|--:|
| cell8 | 49,074 | **49,074** | full | 76 min |
| D | 27,605 | **27,605** | full | 43 min |
| A | 74,846 | 6,426 | 2000/source | 10 min |
| B | 89,706 | 4,000 | 2000/source | 6 min |
| C | 8,660 | 2,660 | 2000/source | 4 min |
| E | 14,194 | — | **done** | — |

🔴 **`cell8` is measured in full**, decided 2026-09-13 and now in `configs/eda.yaml`. Same argument
as pool D and a stronger version of it: 49,074 rows from **one publisher and five generators**, 19%
of the corpus, and the only partition that is `1` on *all four heads*. A 2,000-file sample would
draw from a single `source_name` stratum and let five generators whose durations run 32.90 s to
240.08 s land in whatever proportion the draw happened to pick.

**Done when** every partition has a `signal.parquet`, `load_signal(cfg)` joins them to the M tier
one-to-one, and the decode-failure count is reported per partition (it is printed even when zero).

---

## 3 — The vector half

`ltas[128]`, `mel_band_skew[128]`, `mel_band_kurt[128]` → `<partition>/vectors.npz`, keyed by
`file_id`. The scalar half shipped in `a0f4b70`; this is the other half of the same pass and the
reason `vectors.npz` is a separate artifact from `signal.parquet` ([00 §1](00-harness.md)).

These are the TISMIR music features, *recorded because they are expected to die at 16 kHz*. The
method for testing that is now established rather than hypothetical: the paired round-trip
experiment in [00 §4c](00-harness.md#4c---r1s-premise-measured-on-pool-e-it-holds) holds content
constant and measures what the chain removes. Run the same design on the mel statistics.

---

## 4 — The content tier: VAD + PANNs

The largest remaining build, and the only thing **`G-EDA6`** waits on. Silero VAD at thresholds 0.5
and 0.4, PANNs top-10 tags with scores.

⚠️ **Both must be vendored, not `torch.hub.load`ed** — the eval server is offline
([competition/02](../competition/02-submission.md)). And PANNs conflates singing with Music, so the
tag is evidence, never a label.

---

## 5 — WaveFake, to unblock B1

26.9 GB, already in the store, not on disk. [07 §Phase 1](07-order-and-gates.md) names **B1 first**
for this phase: the WaveFake↔LJSpeech join is the only unconfounded real/fake comparison in the
corpus — same speaker, same utterances, one vocoder apart.

The fetch is I/O-bound and independent of steps 1–4, so it can run in the background from the start.

```bash
python3 scripts/fetch_from_s3.py wavefake \
    --dest /data/project/private/dacon-corpus/raw \
    --extract /data/project/private/dacon-corpus/interim
```

---

## What this plan does not fix, and why

| | |
|---|---|
| `G-EDA2` | Structural. Needs X5 over the spec stream (Phase 2), not more EDA |
| `G-EDA4` | Needs a built fold table (Phase 2) |
| Pool E's 4 groups vs a floor of 6 | A fetch question — no source on disk supplies a 5th |
| Pool C is one source | `mtg-jamendo` is in the store, unfetched; its allowlist gate is a vacuous `na` until it lands |
| `codecfake` | Cannot be extracted here: `zip -s 0` writes 10,737,418,467 bytes from 32,060,882,354 of pieces and exits 0. Needs a newer Info-ZIP, 7-Zip, or an unsplit re-upload |
| `st-codecfake` | No `_meta/` in S3 at all — no `DONE`, no checksums. Needs re-uploading before it can be verified |

---

## The three findings this plan is built on

Carry these forward; they are what the remaining work is *for*.

**1. There is one confound — corpus identity — with fifteen proxies.**
[06 X1b](06-cross-pool.md#x1b---x1-run-on-the-full-corpus-it-is-one-confound-wearing-fifteen-hats):
every metadata column except `n_streams` separates the corpus alone. The fix is a property, not a
knob list — every file leaves the render chain having been through one identical encode, tags
stripped.

**2. Duration survives source-grouping, and that makes it the dangerous one.**
[06 X1c](06-cross-pool.md#-x1c--with-13-sources-grouped-auc-becomes-measurable-and-one-head-refuses-to-collapse):
the voice heads collapse to chance when a publisher is held out (0.488, 0.467); `music_present` does
not (0.991), and duration alone holds **0.933 grouped**. Every music source we hold is long and
every voice source is short, so holding out one leaves the property intact — while the test set is
4–60 s for both. Source-grouped validation, the tool we use to catch shortcuts, actively hides this
one. And [02 B0](02-pool-b-fake-voice.md) shows it is not only cross-publisher: CFAD's own real and
fake halves differ at AUC 0.725 on duration.

**3. R1's premise holds.** The 16 kHz chain destroys the native bandwidth fingerprint
([00 §4c](00-harness.md#4c---r1s-premise-measured-on-pool-e-it-holds)). The resampler's own skirt is
real at AUC 0.940 on *identical* audio and invisible at 0.611 once content varies — a train/test
domain difference worth one cheap transform, not a shortcut, and not something to rank above
duration and level.

---

## Working practice

Commit to **`feat/eda`** whenever there is a solid advance. Stage explicit paths — never
`git add -A`. Keep the gates green first:

```bash
$V -m pytest tests/test_eda.py tests/test_eda_signal.py \
             tests/test_config.py tests/test_fetch_from_s3.py -o addopts="" -q
$V scripts/mutate_eda.py $V        # every invariant paired with a mutation observed to fail
python3 scripts/check_links.py     # after every doc edit
```

⚠️ **Do not run the full pytest suite while editing `configs/eda.yaml`** — several tests assert on
the shipped config and editing it mid-run produces failures that are artifacts of the race.
⚠️ **`scripts/mutate_eda.py` edits files in place**; do not run it concurrently with anything else
that reads them, and run the suite green first — a mutant that dies against an already-red test is
not a kill.
