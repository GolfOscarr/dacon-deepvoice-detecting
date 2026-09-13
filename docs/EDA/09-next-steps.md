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

**M tier: complete.** 382,068 files, 14 probed sources, 6 partitions, `probe_ok`
382,065 — the 3 failures are known-unreadable files in pool C, recorded rather than dropped. Updated 2026-09-13 when
WaveFake landed (step 5) — 117,983 files after its duplicated half was excluded
([02 B0b](02-pool-b-fake-voice.md)).

| A | B | C | D | E | cell8 | mixed |
|--:|--:|--:|--:|--:|--:|--:|
| 74,846 | 207,689 | 8,660 | 27,605 | 14,194 | 49,074 | not probed |

**S tier: pool E only** — 14,194 rows, **3.7%** — until the step-2 pass lands. Every S-tier claim in
these documents is a pool-E claim until then.

**Gates**: `G-EDA5` pass · `G-EDA1/exclude` ×5 pass · `G-EDA1/allowlist/fma` fail (a licence
partition, not a defect) · `G-EDA1/allowlist/mtg-jamendo` `na` (unfetched) · `G-EDA2` fail
(structural — see below) · `G-EDA3` fail, **5 of 14** sources, in two separated clauses
([06 X6b](06-cross-pool.md#-x6b--the-publishers-key-joined-what-path-depth-was-hiding)) ·
`G-EDA4`, `G-EDA6`, `G-EDA7` `na`.

**6 sources blocked**, each with a measured reason: `ctrsvdd`, `rirs-pointsource`,
`rirs-isotropic-rir`, `cfad-codec`, `cfad-noisy`, `codecfake`.

⚠️ **`G-EDA2` cannot be made green by the EDA and must not be chased.** X1 reads `files.parquet` —
the corpus as it sits on disk — while every fix is a *render-time* transform. It will report
1.000 on `music_fake` no matter what is fixed. Closure belongs to `audit_specs` over the spec
stream (X5, Phase 2).

---

## 1 — Publisher keys into the census ✅ done 2026-09-13

Landed in `cc79913`. `eda.groupkeys` joins each publisher's own key into `files.parquet` at
`consolidate` time — no re-probe, nothing decodes — and `grouping_report` prefers it over path
depth. The measured result, and the two corrections it produced, are
[06 X6b](06-cross-pool.md#-x6b--the-publishers-key-joined-what-path-depth-was-hiding).

```bash
$V -m eda.cli keys             # per source: the key, its count, and the evidence
```

**The `stratify_by` decision, recorded.** It does **not** widen. `stratify_by: [source_name]` keeps
the per-source budget and a new `sample.spread_by: [group_key]` divides that budget evenly inside
each source. Same 15,086 sampled files; `cfad-fake` goes from 17–108 per generator to 74–75 and
`zeroth-korean` covers 115 speakers instead of 114. Widening `stratify_by` instead would have
multiplied the budget by the group count — 54,000 files from `cfad-fake` alone.

Still open, and deliberately: **FMA's artist key.** Its `tracks.csv` has `artist` and `album`; the
metadata archive is not fetched, so pool C currently groups on FMA's numbered shard directories,
which are not artists. `eda keys` says so on every run.

---

## 2 — The S tier over everything else

```bash
$V -m eda.cli sample          # draws every partition that has none -- ✅ drawn 2026-09-13
$V -m eda.cli signal          # ~143 min, resumable at 500-file parts
```

**The draw is recorded.** A 6,426 · B 6,000 · C 2,660 · D 27,605 (full) · cell8 49,074 (full), each
with its fingerprint in `<partition>/sample.json`. ⚠️ Redrawing now invalidates every S-tier number
taken from it, and `Sample.fingerprint` refuses to let that happen silently.

**91,765 files to decode, about 2 h 25 min.** Measured rate: pool E's 14,102-row pass took
**21 m 04 s** on 32 threads — **11.2 files/s**. The table below is computed at a deliberately
conservative 10.7, so it over-estimates by ~5%. `user + sys` was 41 minutes against 21 of wall
clock, so the GIL holds the speedup near 2× and this is I/O- and decode-bound, not parallel-bound.

| partition | population | to decode | how | est |
|---|--:|--:|---|--:|
| cell8 | 49,074 | **49,074** | full | 76 min |
| D | 27,605 | **27,605** | full | 43 min |
| A | 74,846 | 6,426 | 2000/source | 10 min |
| B | 207,689 | 6,000 | 2000/source | 9 min |
| C | 8,660 | 2,660 | 2000/source | 4 min |
| E | 14,194 | — | **done** | — |

🔴 **`cell8` is measured in full**, decided 2026-09-13 and now in `configs/eda.yaml`. Same argument
as pool D and a stronger version of it: 49,074 rows from **one publisher and five generators**, 19%
of the corpus, and the only partition that is `1` on *all four heads*. A 2,000-file sample would
draw from a single `source_name` stratum and let five generators whose durations run 32.90 s to
240.08 s land in whatever proportion the draw happened to pick. (Step 1's `spread_by` would now
even that out — but a 4% sample of the only all-four-heads partition is still a 4% sample.)

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

## 5 — WaveFake, to unblock B1 ✅ done 2026-09-13

26.9 GB fetched, sha256 verified, extracted and probed. **117,983 files**, and the 16,283-file
duplicate half it ships is excluded with its reason — [02 B0b](02-pool-b-fake-voice.md) has the
measurement and what it means for B1's pairing.

B1 itself is still to run: it is an S-tier analysis and waits on step 2's decode.

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
