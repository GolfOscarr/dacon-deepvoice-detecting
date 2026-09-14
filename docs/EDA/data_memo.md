# The data memo — X7, the EDA's exit condition

**Written 2026-09-14 against commit `4aa40e0`.** Every number here was measured, not carried
forward; the command that reproduces each one is named beside it. This is
[06 X7](06-cross-pool.md#x7--the-data-memo-e-s3--tier-s-the-exit-condition), and the standard it is
written to is the playbook's: *"do not advance to hyperparameter tuning until the data memo explains
class imbalance, domain shift, and the first leakage hypothesis."* Those three have their own
sections, and the third is the one to read first.

```bash
V=/data/project/private/dacon-venvs/dacon311/bin/python
$V -m eda.cli sources | probe | consolidate | keys   # the M tier and the group keys
$V -m eda.cli sample | signal                        # the S tier, both halves
$V -m eda.cli report                                 # duration, bandwidth, level
$V -m eda.cli analyze | gates                        # X1, E1, grouping, the gates
```

---

## 1 — Inventory

**382,068 files · 2,676 audio hours · 14 runnable sources · 16,700 grouping atoms.**
`probe_ok` 382,065 — the three failures are known-unreadable files in pool C, recorded rather than
dropped (R2).

| partition | role | files | hours | sources | groups |
|---|---|--:|--:|--:|--:|
| A | real voice | 74,846 | 194.8 | 4 | 132 |
| B | fake voice | 207,689 | 302.8 | 3 | 83 |
| C | real instrumental | 8,660 | 109.2 | 2 | 238 |
| D | fake instrumental | 27,605 | 77.3 | 1 | 5,521 |
| E | non-musical sound | 14,194 | 21.6 | 3 | 10,722 |
| cell 8 | whole-file AI songs | 49,074 | **1,970.6** | 1 | 5 |

🔴 **One source is 74% of the corpus's audio.** SONICS' 49,074 songs are 1,970.6 of the 2,676 hours,
at a median of 131 s against pool E's 4 s. Counting *files* hides this completely, and it is why the
S-tier cost estimate was wrong by an order of magnitude until it was recounted in hours
([09 §2](09-next-steps.md)).

| source | partition | files | hours | groups | key |
|---|---|--:|--:|--:|---|
| `cfad-real` | A | 38,600 | 57.5 | 14 | path |
| `zeroth-korean` | A | 22,720 | 52.9 | 115 | path |
| `ljspeech` | A | 13,100 | 23.9 | **1** | declared single |
| `musan-speech` | A | 426 | 60.4 | 2 | path |
| `wavefake` | B | 117,983 | 198.7 | **2** | publisher |
| `cfad-fake` | B | 73,700 | 69.5 | 27 | path |
| `mlaad` | B | 16,006 | 34.6 | 54 | path |
| `fma` | C | 8,000 | 66.6 | 156 | path |
| `musan-music` | C | 660 | 42.6 | 82 | publisher |
| `fakemusiccaps` | D | 27,605 | 77.3 | 5,521 | publisher |
| `compspoof-env-bonafide` | E | 13,172 | 14.6 | 10,710 | publisher |
| `musan-noise` | E | 930 | 6.2 | 2 | path |
| `rirs-isotropic-noise` | E | 92 | 0.8 | 10 | publisher |
| `sonics` | cell 8 | 49,074 | 1,970.6 | 5 | publisher |

**6 sources are blocked**, each with a measured reason: `ctrsvdd`, `rirs-pointsource`,
`rirs-isotropic-rir`, `cfad-codec`, `cfad-noisy`, `codecfake`. **5 are declared and not on disk**:
`common-voice-en`, `common-voice-ko`, `libritts-r`, `mtg-jamendo`, `ctrsvdd` — which is both absent
and blocked, so 14 of the 20 declared sources are runnable and probed.

⚠️ `compspoof-v2` is a sixth exception and a different one: it **is** on disk, declared into the
`mixed` partition, and has never been enumerated. It is absent from every number above.

---

## 2 — Schema

Three tables, because the tiers have different populations *and* different shapes
([00 §1](00-harness.md)).

| artifact | tier | population | shape |
|---|---|---|---|
| `<partition>/files.parquet` | M | **100%**, 382,068 rows | **30** columns: ffprobe, sha256, mp3 header, `group_key` |
| `<partition>/signal.parquet` | S | sampled/full, 58,885 rows | **81** columns, the measured ones suffixed `_native` / `_chain` |
| `<partition>/vectors.npz` | S | the same 58,885, keyed by position | `ltas`, `mel_band_skew`, `mel_band_kurt`, `[128]` each, both planes |

**S-tier coverage**: A 6,426 · B 6,000 · C 2,660 · cell8 2,000 drawn at 2,000 per source spread
evenly over `group_key`; **D 27,605 and E 14,194 in full**. 0 decode failures.

**Two measurement planes (R1).** One decode, measured twice: `native` as published, `chain` after
the 16 kHz resample and channel policy. Every S-tier row is its own control, which is what makes a
`native − chain` difference a property of the transform rather than of two populations.

---

## 3 — Class imbalance

Counted on the M tier. A head is `not scored` where the metric ignores it — a pool-A row has no
`music_fake`, and `POOL_LABELS` says so with `None` rather than a 0.

| head | metric weight | scored | positive | rate | not scored |
|---|--:|--:|--:|--:|--:|
| `voice_present` | 0.05 | 382,068 | 331,609 | **0.868** | 0 |
| `music_present` | 0.05 | 382,068 | 85,339 | **0.223** | 0 |
| `voice_fake` | 0.18 | 331,609 | 256,763 | **0.774** | 50,459 |
| `music_fake` | **0.27** | 85,339 | 76,679 | **0.899** | 296,729 |

⚠️ **The two heaviest heads are the two most imbalanced, in opposite directions.** `music_fake`
carries 0.27 of the metric and is 90% positive over only 85,339 scored rows — the corpus holds
8,660 real instrumental files against 76,679 fake ones. `music_present` is 22% positive.

⚠️ And these are **corpus** rates, not manifest rates. The composed majority's labels do not exist
until the sampler has drawn a spec, so the numbers above describe the material, not the training
distribution. Rebalancing is a sampler decision and belongs to Phase 2.

---

## 4 — Domain shift

The test set is **1,200 files, 4–60 s, 16 kHz** ([competition/01](../competition/01-overview.md)).
Measured against it on the chain plane (`eda report`):

| partition | p05 | median | p95 | **in 4–60 s** | under 4 s | over 60 s | holds a 4 s span |
|---|--:|--:|--:|--:|--:|--:|--:|
| A | 2.47 | 6.96 | 411.3 | 0.761 | 0.174 | 0.066 | 0.182 |
| B | 2.03 | 4.725 | 13.8 | **0.586** | **0.414** | 0.000 | **0.103** |
| C | 29.98 | 30.00 | 296.1 | 0.757 | 0.001 | 0.242 | 0.992 |
| D | **10.00** | **10.00** | 10.24 | 1.000 | 0.000 | 0.000 | 0.946 |
| E | 4.00 | 4.00 | 6.3 | 0.980 | 0.014 | 0.007 | 0.584 |
| cell 8 | 32.8 | 120.0 | 239.0 | **0.202** | 0.000 | **0.798** | 0.999 |

Three shifts, in order of how much they cost:

1. **Duration does not match, and it differs *between* pools.** Pool D is 10 s to three decimals,
   pool C is 30 s, cell 8 is two minutes. See §5 — this is also the leakage hypothesis.
2. **41% of pool B is shorter than the test set's minimum**, and only 10.3% of it holds a 4 s
   non-silent span — the least usable material in the corpus.
3. **Level.** Median RMS spans **22.6 dB** across sources on the chain plane, and all three MUSAN
   partitions sit at peak −0.000265 dBFS with 86–88% of files clipping: peak-normalised to full
   scale, a publisher fingerprint rather than a property of the audio
   ([01 A9](01-pool-a-real-voice.md)).

**Bandwidth is *not* a shift**, and that is a finding rather than an absence. Ten of fourteen
sources are natively at or below 16 kHz, so the chain removes nothing from them; only `fma`
(−10,389 Hz), `wavefake` (−3,220), `ljspeech` (−2,853) and `mlaad` (−2,530) lose anything, and those
four are split across three pools ([00 §4d](00-harness.md#4d---r1-on-the-whole-corpus-ten-of-fourteen-sources-lose-nothing)).

---

## 5 — 🔴 The leakage hypothesis

**Duration predicts the music labels, it survives a whole-publisher holdout, and the render chain
cannot touch it.**

The evidence, in the order it was found:

| | measurement | where |
|---|---|---|
| M tier, full corpus | every metadata column except `n_streams` separates the corpus alone | [06 X1b](06-cross-pool.md#x1b---x1-run-on-the-full-corpus-it-is-one-confound-wearing-fifteen-hats) |
| M tier, source-grouped | voice heads collapse (0.488, 0.467); `music_present` does not (0.991), and **duration alone holds 0.933** | [06 X1c](06-cross-pool.md) |
| **S tier, chain plane, source-grouped** | `music_present` **0.852**, carried by `longest_valid_span_s` 0.943 and `duration_s_decoded` 0.922 | [06 X1d](06-cross-pool.md#-x1d--x1-on-the-decoded-audio-the-duration-shortcut-survives-the-chain) |
| the cause | pool D is **10.000 s** for every file; pool C is 30.003 s | [04 D8](04-pool-d-fake-instrumental.md), [03 C8](03-pool-c-real-instrumental.md) |

Against a gate of **AUC < 0.60**.

**Why it is the dangerous one.** Source-grouped validation is the tool we use to catch shortcuts,
and it *hides* this one: every music source we hold is long and every voice source is short, so
holding out one publisher leaves the property intact. A model can score beautifully in CV by
measuring length, and the test set is 4–60 s for both classes.

**What the chain does remove, for contrast**: `music_fake` falls 0.999 native → 0.898 chain, and the
features that drop out are `mel_bands_flat` and `effective_bandwidth_hz` — the native sample rate
wearing acoustic names. The rate fingerprint dies at 8 kHz. Duration does not.

⚠️ **A second leakage axis, structural rather than statistical**: near-duplicate and sibling content
across fold boundaries. Found and fixed once already — CompSpoof shares **292 parent recordings
between its two splits, 1,060 files**, which the path-derived key did not catch
([05 E1b](05-pool-e-noise.md)). `G-EDA4` is the gate for this and it cannot run until a fold table
exists.

---

## 6 — Risk list

| risk | state | what would close it |
|---|---|---|
| 🔴 Duration shortcut on the music heads | **open, measured at 0.852** | a crop/pad policy that makes pool C and pool D indistinguishable in length; then re-run X1d |
| 🔴 `G-EDA2` reports 1.000 on `music_fake` | **structural, will not go green here** | X5 over the spec stream (Phase 2). X1 reads the corpus on disk; every fix is render-time |
| Level as a publisher fingerprint | open, 22.6 dB spread | a normalisation decision, and the stage to apply it at |
| `G-EDA3`: 5 of 14 sources under 6 groups | 2 have no publisher key (`musan-noise`, `musan-speech`); 3 are **genuinely short** (`ljspeech` 1, `sonics` 5, `wavefake` 2) | nothing — the last three are facts about the sources. They cannot be rotated in a fold table |
| `G-EDA1/allowlist/fma` fails, 2,907 of 8,000 | open | a licence decision (C1), not a computation |
| Pool C is one usable source | open | `mtg-jamendo` is in the store, unfetched |
| `G-EDA6` — component evidence | `na` | the content tier: VAD + PANNs, vendored ([09 §7](09-next-steps.md)) |
| `G-EDA4` — pairs vs fold boundaries | `na` | a built fold table (Phase 2) |
| `G-EDA7` — filter-rate symmetry | `na` | a filter. Phase 0 applies none, by design (R2) |
| E1 pass 2 is not a real content fingerprint | open, and **known to be weak** | a landmark/chromaprint fingerprint, or waveform cross-correlation over the LTAS shortlist |
| `codecfake` unextractable; `st-codecfake` has no `_meta/` | blocked | a newer Info-ZIP or an unsplit re-upload |

---

## 7 — What this memo does not cover

Stated rather than left as a silence:

* **No content-tier measurement.** Whether pool C is actually instrumental (C2), where the valid
  4 s spans are (E2), and every PANNs tag are unmeasured. `G-EDA6` is `na` for that reason.
* **`FILE_FAKE` is not audited.** A composed file's fake status does not exist until the sampler
  draws a spec, so the file head — **0.45 of the metric** — is X5's question, over the spec stream,
  not this corpus's.
* **The `mixed` partition is unprobed.** `compspoof-v2` is declared and never enumerated.
* **S-tier numbers are sampled for A, B, C and cell 8.** The draw is recorded with its seed and
  fingerprint in `<partition>/sample.json`; D and E are complete.

---

## 8 — The three things to carry into processing strategy

1. **Duration is the shortcut, and only the sampler can fix it.** Nothing in the render chain
   changes a file's length.
2. **Above 8 kHz is not available.** Ten of fourteen sources have nothing there to begin with, so
   any feature engineered up there works on four sources split across three pools.
3. **Level and metadata are publisher fingerprints.** Metadata is already handled — every file
   leaves the render chain through one identical encode with tags stripped. Level is not, yet.

---

*Regenerate the numbers with the commands at the top. This memo is versioned with the corpus: if
`eda analyze` or `eda report` disagrees with a figure here, the figure here is stale and the
artifact wins.*
