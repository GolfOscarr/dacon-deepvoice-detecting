# The data memo — X7, the EDA's exit condition

**Written 2026-09-14; updated 2026-09-15 (content tier), 2026-09-16 (steps 1-4 of
[10](10-final-plan.md)) and 2026-09-17 (A6, and the last two sub-parts).** Every number here was measured, not carried forward; the command that
reproduces each one is named beside it. This is
[06 X7](06-cross-pool.md#x7--the-data-memo-e-s3--tier-s-the-exit-condition), and the standard it is
written to is the playbook's: *"do not advance to hyperparameter tuning until the data memo explains
class imbalance, domain shift, and the first leakage hypothesis."* Those three have their own
sections, and the third is the one to read first.

```bash
V=/data/project/private/dacon-venvs/dacon311/bin/python
# read-only, no decode -- one verb per line, because they are not a pipeline
for verb in sources probe consolidate keys sample report analyze gates \
            b1 screens roles envelope reassign; do
  $V -m eda.cli "$verb"
done
$V -m eda.cli signal          # the S tier: ~8 h. Not to be re-run (10 section 4)
$V -m eda.cli envelope --run  # C5's decode, ~5 min over the corpus
```

⚠️ **Completeness claims in this repository have been wrong before.** The 2026-09-15 version of this
memo said the EDA was complete on the strength of [09](09-next-steps.md)'s eight steps; 09 was a
remaining-work plan and never the task inventory. [10 §1](10-final-plan.md) is the inventory —
**34 answered · 7 blocked · 2 Phase 2 = 43** — and it is the page to check before believing this
one. ⚠️ That page's own header was wrong from 2026-09-16 to 2026-09-17 (it read *"33 … = 42"*
while A6 was still outstanding), so **count the status markers, not the header**.

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
| `<partition>/signal.parquet` | S + C | sampled/full, 58,885 rows | **90** columns — S measured on both planes and suffixed, C on the chain plane only and unsuffixed |
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

**There are four: three statistical and one structural, on both scored heads.** The playbook asks
for *the first* leakage hypothesis; this corpus has several that matter, and collapsing them into
one is the error that would leave most of the problem in place. L1 and L2 are within **0.02 AUC** of
each other on the same 0.27-weight head and are **independent** — one is the file's length, the
other its first 20 ms.

| | shortcut | what it is | AUC | fixed by |
|---|---|---|--:|---|
| **L1** | **duration** | the file's *length* | **0.852** (whole-publisher holdout) | a crop policy |
| **L2** | **onset deficit** | the file's first **20 ms** | **0.869** (pool C vs D) | a **random-offset** crop |
| **L3** | **sibling content** | the same recording either side of a fold | *structural* | the fold builder (`G-EDA4`, Phase 2) |
| **L4** | **leading silence** | the file's first *seconds* | **0.639** (same utterance, one vocoder apart) | the `A-A11` silence-edit augmentation |

⚠️ **L4 is on the *voice* head, and it is the one that survives pairing.** See
[01 A6b](01-pool-a-real-voice.md): 0.612 corpus-wide, **0.717** inside one publisher, 0.639 between
LJSpeech and its own vocoded copies. `silence_lead_s` / `silence_tail_s` are **0.0** in
`configs/run_default.yaml` today, so no RNG draw happens; this measurement says make them non-zero,
symmetrically across pools A and B (R2).

🔴 **A crop that fixes L1 does not necessarily fix L2.** Cropping every music file to a common
length removes duration and leaves the file's *beginning* exactly where it was. Only drawing the
crop from a **random offset inside** the file removes both — and R2 makes that symmetric by
construction only if pools C and D both go through it.

### L1 — duration

The evidence, in the order it was found:

| | measurement | where |
|---|---|---|
| M tier, full corpus | every metadata column except `n_streams` separates the corpus alone | [06 X1b](06-cross-pool.md#x1b---x1-run-on-the-full-corpus-it-is-one-confound-wearing-fifteen-hats) |
| M tier, source-grouped ⚠️ **dated** | voice heads collapse (0.488, 0.467); `music_present` does not (0.991), and **duration alone holds 0.933** | [06 X1c](06-cross-pool.md) |
| **S tier, chain plane, source-grouped** | `music_present` **0.852**, carried by `longest_valid_span_s` 0.943 and `duration_s_decoded` 0.922 | [06 X1d](06-cross-pool.md#-x1d--x1-on-the-decoded-audio-the-duration-shortcut-survives-the-chain) |
| the cause | pool D is **10.000 s** for every file; pool C is 30.003 s | [04 D8](04-pool-d-fake-instrumental.md), [03 C8](03-pool-c-real-instrumental.md) |

Against a gate of **AUC < 0.60**.

⚠️ **The X1c row is a dated snapshot and today's artifact disagrees with it on purpose.** X1c was
run on **264,085 rows across 13 sources on 2026-09-12, before WaveFake landed**. Today's
`shortcut_audit.parquet` (382,068 rows, 14 sources) reads `voice_present` **0.324**, `voice_fake`
**0.333**, `music_present` **0.583** source-grouped. The row is kept because it is the order the
evidence arrived in; it is **not** a current claim. The live number for L1 is the S-tier one below,
and it lives in `screen_metadata_leak.parquet` as `auc_source_grouped_chain`.

**Why it is the dangerous one.** Source-grouped validation is the tool we use to catch shortcuts,
and it *hides* this one: every music source we hold is long and every voice source is short, so
holding out one publisher leaves the property intact. A model can score beautifully in CV by
measuring length, and the test set is 4–60 s for both classes.

**What the chain does remove, for contrast**: `music_fake` falls 0.999 native → 0.898 chain, and the
features that drop out are `mel_bands_flat` and `effective_bandwidth_hz` — the native sample rate
wearing acoustic names. The rate fingerprint dies at 8 kHz. Duration does not.

### L2 — the first 20 ms

**Pool C opens 43.35 dB below its own median level; pool D opens 0.90 dB below it.** Measured over
all 58,885 drawn files on the chain plane
([03 C5b](03-pool-c-real-instrumental.md)); `onset_level_deficit_db` separates the two pools at
**AUC 0.869** and the interquartile ranges do not overlap (C p25 34.53, D p75 6.93).

⚠️ **The direction is the opposite of what [03 C5](03-pool-c-real-instrumental.md) predicted.** The
spec assumed real music is cut from track centres (hard) and generated music begins from silence.
In fact the **real** pool carries the editing artefact — a short de-click ramp at the excerpt cut —
and the **generated** pool carries none, because a generation simply begins. The mechanism C5
identified is right; its sign was assumed rather than measured.

Two alternative explanations were ruled out before this was published: **mp3 decoder padding**
(`musan-music`, a wav source, shows the corpus's largest deficit at 77.04 dB) and **loudness
compression** (|r| ≤ 0.33 against `crest_factor_db_chain`).

🔴 **Why it is dangerous in the same way L1 is:** it is not acoustic, no spectral analysis would
surface it, and it survives the render chain untouched — resampling does not move a file's first
frame.

### L3 — structural, and not statistical

⚠️ Near-duplicate and sibling content across fold boundaries. Found and fixed once already — CompSpoof shares **292 parent recordings
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
| `G-EDA1/allowlist/fma` fails, 2,907 of 8,000 | ⏭️ deferred | **23 DENY + 2,884 ND**; ND resolved usable by #417333 A5. The allowlist is stale, not restrictive ([03 C1b](03-pool-c-real-instrumental.md)) |
| Pool C is **two** sources, one of them small | open | `fma` 8,000 files / 66.6 h and `musan-music` 660 / 42.6 h. Both usable -- the risk is **publisher diversity**, not usability. `mtg-jamendo` would be a third and is in the store, unfetched ([03 C1b](03-pool-c-real-instrumental.md)) |
| 🔴 `G-EDA6` — component evidence | **fail: 2,442 of 58,884 rows contradict their assertions** | ✅ **decomposed** into three problems with opposite actions ([07 G-EDA6b](07-order-and-gates.md)); acting on it is a **manifest** change. Worst source rate: pool C at 12.6%, and that is a floor |
| Sung voice cannot be evidenced | open, **measured** | a singing-aware detector. A speech VAD finds vocals in 28.1% of files the publisher says all have them ([00 §4e](00-harness.md#4e---what-the-vad-can-and-cannot-evidence)) |
| `G-EDA4` — pairs vs fold boundaries | `na` | a built fold table (Phase 2) |
| `G-EDA7` — filter-rate symmetry | `na` | a filter. Phase 0 applies none, by design (R2) |
| E1 pass 2 is not a real content fingerprint | open, and **known to be weak** | a landmark/chromaprint fingerprint, or waveform cross-correlation over the LTAS shortlist |
| `codecfake` unextractable; `st-codecfake` has no `_meta/` | blocked | a newer Info-ZIP or an unsplit re-upload |

---

## 7 — What this memo does not cover

Stated rather than left as a silence:

* **No PANNs tags.** Decided 2026-09-15 rather than skipped: it is trained at 32 kHz against our
  16 kHz chain plane and conflates singing with Music. The voice assertions are evidenced by VAD;
  the **music** assertions rest on energy and spectral proxies, not on a tagger.
* **Sung voice is not evidenced at all.** The content tier measures speech, and the corpus's sung
  sources (`sonics`, and `ctrsvdd` when unblocked) are outside it
  ([00 §4e](00-harness.md#4e---what-the-vad-can-and-cannot-evidence)).
* **`FILE_FAKE` is not audited.** A composed file's fake status does not exist until the sampler
  draws a spec, so the file head — **0.45 of the metric** — is X5's question, over the spec stream,
  not this corpus's.
* **The `mixed` partition is unprobed.** `compspoof-v2` is declared and never enumerated.
* **S-tier numbers are sampled for A, B, C and cell 8.** The draw is recorded with its seed and
  fingerprint in `<partition>/sample.json`; D and E are complete.

---

## 8 — What to carry into processing strategy

1. 🔴 **Two shortcuts, not one, and a crop must remove both.** L1 is the file's length, L2 its
   first 20 ms. **Only a random offset inside the file removes both**, applied identically to pools
   C and D or it manufactures the cue it is meant to erase (R2).
2. 🔴 **Nothing in `files.parquet` is a feature.** All 30 M-tier columns are a leakage risk, a split
   key or the label — `roles.md` assigns every one of the corpus's 118 columns
   ([06 X6b](06-cross-pool.md)). Metadata scores AUC 1.000 in CV and **0.0000008** under an archive
   holdout: it does not merely fail to transfer, it **inverts**.
3. **Above 8 kHz is not available.** Ten of fourteen sources have nothing there, so any feature
   engineered up there works on four sources split across three pools.
4. 🔴 **Do not high-pass — and decide it together with DC removal.** Five of seven vocoder
   signatures live below 72 Hz ([02 B1b](02-pool-b-fake-voice.md)), so a rumble filter would delete
   the largest discriminative feature pool B has. ⚠️ But `cfad/pwg` is identifiable at **AUC 0.998**
   from `dc_offset` alone ([02 B6b](02-pool-b-fake-voice.md)), so DC removal would *erase* a
   generator id. The two pull in opposite directions and are one decision, not two.
5. **`artifact_family` is measured, not named.** `melgan` ↔ `melgan_large` correlate at 0.895;
   `melgan` ↔ `multi_band_melgan` at **−0.186** — same name, opposite artefacts. A fold builder that
   groups by model name merges two families and splits the one real one.
6. **Level is a publisher fingerprint and is *not* neutralised by the chain** — 22.6 dB of
   median-RMS spread. Metadata is neutralised by the render chain; level is not, yet.
7. **No degenerate-generation population exists to drop**, and a drop would be asymmetric anyway:
   pool C flags **3.6×** more often than pool D ([02 B4b](02-pool-b-fake-voice.md)). ⚠️ But the VAD
   finds **23** that the level/clipping screen structurally could not
   ([07 G-EDA6b](07-order-and-gates.md)) — the right detector for *"the TTS produced no speech"* is
   a VAD.
8. 🔴 **Turn the silence augmentation on.** `silence_lead_s` / `silence_tail_s` are **0.0** in
   `configs/run_default.yaml`, so no RNG draw happens at all. L4 clears the gate in every
   configuration measured and is **strongest where the controls are tightest**
   ([01 A6b](01-pool-a-real-voice.md)). Symmetric across pools A and B, or it becomes the cue it
   inoculates against. ⚠️ Changes the drawn stream — needs an I1b re-run.
9. 🔴 **Do not trim silence**, and the reason is now arithmetic as well as principled. Trimming
   costs pool A **7.78 pp** of the 4-60 s window against pool B's **4.12 pp**, so it is an
   **asymmetric transform** (R2) that shifts the real/fake balance toward the label; pool E would
   lose **27.28 pp** ([01 A5b](01-pool-a-real-voice.md)). It also does nothing for L1 — the music
   pools are untouched.

---

## 9 — The two decisions that were open, and where they landed

⚠️ **Both were open when this section was written and neither is now.** Kept as a section because
each still needs an *action* somewhere outside the EDA, and because the shape of each answer is the
useful part.

| | what | the number |
|---|---|---|
| ~~fma licence allowlist~~ | ⏭️ **answered and deferred**, [03 C1b](03-pool-c-real-instrumental.md): the 2,907 are **23 DENY + 2,884 ND**, and ND is usable per #417333 A5. Regenerating is ~10 min and adds **+57% volume, 0 publishers, 0 shortcut dilution** — deferred until pool C volume binds | 23 unusable |
| **G-EDA6 reassignment** | ✅ **decomposed**, [07 G-EDA6b](07-order-and-gates.md). Three problems, opposite actions | **1,342** → cells 5/8 · **1,032** pool-E restricted from the noise layer · **23** generation failures · 45 not contradictions |

---

*Regenerate the numbers with the commands at the top. This memo is versioned with the corpus: if
`eda analyze` or `eda report` disagrees with a figure here, the figure here is stale and the
artifact wins.*
