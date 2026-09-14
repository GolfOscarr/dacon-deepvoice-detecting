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

**S tier: complete.** 58,885 rows over six partitions, **0 decode failures**, scalars and
`[128]`-wide vectors on both planes. The sampled partitions are A 6,426 · B 6,000 · C 2,660 ·
cell8 2,000; D and E are measured in full.

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

## 2 + 3 — The S tier, both halves ✅ done 2026-09-14

**58,885 rows over six partitions, 0 decode failures**, scalars in `signal.parquet` and
`[128]`-wide vectors in `vectors.npz`, keyed position for position. 302 audio hours at a measured
1.33 audio-hours/min on 32 threads.

Two things the run corrected about the plan itself:

* 🔴 **The estimate was in files and the cost is in audio hours.** A cell8 file is 131 s and a
  pool-E file is 4 s. Measured in full, cell8 alone is 1,970.6 audio hours — **~25 hours**, 90% of
  the S tier's cost for 19% of its rows. It left `full_pools`; its 2,000-file draw is now **400 per
  generator exactly**, which `spread_by` makes safe and which was the whole reason full coverage had
  been chosen.
* Steps 2 and 3 are **one pass**. `vectors.npz` comes out of the same decode, so running them apart
  means opening every file twice. Learned by doing it and stopping four minutes in.

⚠️ And one defect it exposed: `stage` is a config property that `files.parquet` does not carry, so
every `stage: raw` source resolved against `interim/`. MLAAD is the only one, so pool B silently
lost a third of its S tier while the other five partitions looked perfect. Fixed in `7dadab4`;
pool B re-run clean.

---

## 5 — WaveFake, to unblock B1 ✅ done 2026-09-13

26.9 GB fetched, sha256 verified, extracted and probed. **117,983 files**, and the 16,283-file
duplicate half it ships is excluded with its reason — [02 B0b](02-pool-b-fake-voice.md) has the
measurement and what it means for B1's pairing.

B1 itself is still to run: it is an S-tier analysis and waits on step 2's decode.

---

## 6 — Reading the S tier ✅ done 2026-09-14

```bash
$V -m eda.cli report           # duration, bandwidth, level -- no decode
```

The S tier had been measured and unread: `analyze` stops at `files.parquet`. Three findings, each
written up where it belongs:

| finding | where | what it decides |
|---|---|---|
| 🔴 **Duration survives the chain.** `music_present` holds **0.852** after a whole-publisher holdout, against a 0.60 gate. Pool D is *every file exactly 10.000 s*, pool C is 30 s | [06 X1d](06-cross-pool.md#-x1d--x1-on-the-decoded-audio-the-duration-shortcut-survives-the-chain), [03 C8](03-pool-c-real-instrumental.md), [04 D8](04-pool-d-fake-instrumental.md) | the crop policy |
| **The voice heads are clean** — 0.586 and 0.401 grouped, collapsing as X1c predicted | [06 X1d](06-cross-pool.md#-x1d--x1-on-the-decoded-audio-the-duration-shortcut-survives-the-chain) | — |
| **Ten of fourteen sources lose nothing to the chain**; only `fma`, `wavefake`, `ljspeech`, `mlaad` have content above 8 kHz at all | [00 §4d](00-harness.md#4d---r1-on-the-whole-corpus-ten-of-fourteen-sources-lose-nothing) | whether any >8 kHz feature is worth engineering |
| **22.6 dB of median-RMS spread**, and all three MUSAN partitions peak-normalised to full scale | [01 A9](01-pool-a-real-voice.md) | normalise, and at which stage |
| **41% of pool B is under 4 s**; only 10.3% hold a 4 s non-silent span | [02 B0c](02-pool-b-fake-voice.md) | usable material for the sampler |
| **E1 pass 2 fell short** — LTAS is a timbre fingerprint, not an identity one — but following it found CompSpoof sharing **292 parent recordings across its two splits** | [05 E1b](05-pool-e-noise.md) | grouping; now fixed, 6 → 10,710 groups |

---

## 7 — The content tier: VAD + PANNs ⬅ the largest remaining build

The largest remaining build, and the only thing **`G-EDA6`** waits on. Silero VAD at thresholds 0.5
and 0.4, PANNs top-10 tags with scores.

⚠️ **Both must be vendored, not `torch.hub.load`ed** — the eval server is offline
([competition/02](../competition/02-submission.md)). And PANNs conflates singing with Music, so the
tag is evidence, never a label.

---

⚠️ **It needs a second decode pass, ~4 h.** The models were not vendored when step 2 ran, so the
scalars and vectors came out of a decode that could not also run them. Scope it to the partitions
that need it -- C for [C2](03-pool-c-real-instrumental.md), E for
[E2](05-pool-e-noise.md), then A and B -- rather than all six.

---

## 8 — X7, the data memo ✅ done 2026-09-14

**[`docs/EDA/data_memo.md`](data_memo.md)** — inventory, schema, class imbalance, domain shift, the
leakage hypothesis, and the risk list, every figure measured and named with the command that
reproduces it. It is the EDA's exit condition ([06 X7](06-cross-pool.md)) and the input document for
the processing-strategy phase.

The three things it asks you to carry forward:

1. **Duration is the shortcut, and only the sampler can fix it** — nothing in the render chain
   changes a file's length.
2. **Above 8 kHz is not available** — ten of fourteen sources have nothing there to begin with.
3. **Level and metadata are publisher fingerprints** — metadata is already handled by the render
   chain; level is not, yet.

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
