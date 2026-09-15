# 06 — Cross-Pool Analyses

The five pool plans measure *what each pool is*. These measure *what the pools are relative to each
other* — which is where the failure mode that actually kills this kind of project lives. None of
them can be run inside a single pool, and all of them must be **re-run after every corpus change**.

---

## X1 — 🔴 The shortcut audit (E-S2) — Tier S, the blocking gate

**Compute.** A logistic regression on **metadata-only** features — no learned audio representation
— predicting each of the five labels, and each cell, in turn.

Feature set (all from [00 §4](00-harness.md), at both planes):
`duration_s`, `orig_sr`, `orig_channels`, `container`, `codec`, `bitrate`, `lufs_integrated`,
`peak_dbfs`, `crest_factor`, `dc_offset`, `clipping_ratio`, `silence_ratio`, `lead_silence_s`,
`tail_silence_s`, `effective_bandwidth_hz`, `near_nyquist_ratio`, `band_energy[8]`,
`spectral_flatness`, `statistics_T`.

Report **AUC per head and per cell**, plus per-feature coefficients and a permutation importance
ranking.

**Why it is meaningful.** This is the single most important number in the entire EDA. It detects
*"the failure mode that kills this kind of project — a confound separating real/fake with no
acoustic content"* ([data/07](../data/07-eda-plan.md)). Our corpus is built from archives that
were never meant to sit beside each other; A1, B3, C6, D2 each predict a specific confound, and
this is where they sum. ☆ `[LLM-Detect-AI 2024, efficiency prize]` is the cautionary case: adding
off-distribution data *"caused severe data drift which further increased the CV/LB gap."*

**What we discover.** Not just whether a shortcut exists, but **which feature carries it** — the
permutation ranking tells you what to neutralize, and neutralizing the wrong thing is worse than
neutralizing nothing.

**🔴 The gate is AUC < 0.60**, per head, per cell. Above it: neutralize before training anything.
The neutralization is always a **transform applied symmetrically to every pool** (R2) — a codec
round-trip, a resampler draw, a crop policy, a loudness normalization — never a feature deletion,
because the feature will still be there in the audio.

⚠️ **Run it on the population, not on a stratified sample.** Stratifying by `source_name` destroys
exactly the effect being measured ([00 §3](00-harness.md)).

⚠️ **And run it per cell, not only per head.** A confound can be absent marginally and present
inside a presence stratum — the same structure as the marginal-composedness gap that
[data/02](../data/02-label-taxonomy.md) found and fixed: *"`P(composed|FAKE) = P(composed|REAL)` can
be satisfied while composedness still predicts the label inside a presence stratum."* A marginal
AUC of 0.52 is not evidence of safety.

---

## X1b — 🔴 X1 run on the full corpus: it is one confound wearing fifteen hats

Measured 2026-09-12 over **151,693 rows / 10 sources** — pools A-E plus `cell8`, the first audit
with whole-file rows in it.

| head | n | AUC |
|---|---|---|
| `music_fake` | 85,339 | **1.0000** |
| `music_present` | 151,693 | 0.99996 |
| `voice_fake` | 101,326 | 0.9611 |
| `voice_present` | 151,693 | 0.9919 |

⚠️ **This corrects how the earlier result was summarised, not the result itself.** The wave-1
handoff reported the shortcut as *"three metadata columns — duration, format, channels — and
nothing underneath them"*. The cumulative ablation it was drawn from actually removed five groups
before reaching 0.500, and re-running the audit one column at a time shows why that compression is
dangerous. **Every metadata column except `n_streams` separates the corpus on its own:**

| feature alone | voice_present | music_present | voice_fake | music_fake |
|---|---|---|---|---|
| `bit_rate` | **0.941** | 0.672 | 0.799 | 0.640 |
| `codec_name` | 0.890 | **0.998** | 0.954 | 0.705 |
| `sample_fmt` | 0.835 | **0.997** | 0.877 | 0.705 |
| `container` | 0.793 | 0.891 | **0.954** | 0.642 |
| `encoder` | 0.783 | 0.834 | 0.877 | **0.986** |
| `file_bytes` | 0.743 | 0.880 | **0.898** | 0.870 |
| `duration_s` | 0.658 | **0.946** | 0.860 | 0.603 |
| `lame_version` | 0.763 | 0.811 | 0.877 | **0.904** |
| `xing_tag` | 0.663 | 0.834 | 0.877 | 0.642 |
| `lame_tag` | 0.702 | 0.811 | 0.877 | 0.587 |
| `bits_per_raw_sample` | 0.659 | 0.717 | 0.813 | 0.500 |
| `orig_sr` | 0.541 | 0.701 | 0.558 | **0.962** |
| `orig_channels` | 0.636 | 0.503 | 0.500 | **0.957** |
| `id3_tag` | 0.579 | 0.547 | 0.500 | **0.962** |
| `n_streams` | 0.500 | 0.500 | 0.500 | 0.500 |
| **all fifteen removed** | **0.500** | **0.500** | **0.500** | **0.500** |

**What survives from the earlier diagnosis**: removing every metadata feature puts all four heads at
exactly 0.500. There is no shortcut outside the columns we measure, and the audit is complete rather
than merely alarming.

**What does not**: the idea that three knobs close it. `encoder` alone predicts `music_fake` at
0.986, `bit_rate` alone predicts `voice_present` at 0.941, and `file_bytes` alone predicts
`voice_fake` at 0.898 — none of which is duration, container, or channel count. A transform plan
aimed at those three would leave every row of that table in place.

### 🔴 X1c — with 13 sources, grouped AUC becomes measurable, and one head refuses to collapse

Re-run 2026-09-12 after CFAD landed: **264,085 rows, 13 sources**. Enough source groups that
holding one out no longer leaves a single-class fold, so `auc_source_grouped` is real for three of
the four heads for the first time.

| head | n | AUC | **source-grouped** |
|---|--:|--:|--:|
| `voice_present` | 264,085 | 0.991 | **0.488** |
| `voice_fake` | 213,626 | 0.867 | **0.467** |
| `music_present` | 264,085 | 1.000 | **0.991** |
| `music_fake` | 85,339 | 1.000 | n/a — 4 groups |

**The voice heads collapse to chance.** That is the corpus-identity diagnosis confirmed the strong
way: hold out the publisher and the metadata tells you nothing about whether a voice is present or
generated.

**`music_present` does not collapse.** One feature at a time, grouped:

| feature alone | AUC | grouped |
|---|--:|--:|
| `duration_s` | 0.975 | **0.933** |
| `file_bytes` | 0.946 | **0.921** |
| `codec_name` | 0.997 | 0.875 |
| `container` | 0.855 | 0.750 |
| `orig_sr` | 0.455 | 0.167 |

🔴 **Duration survives source-grouping because it is a property of the domain, not of the
publisher.** Every music source we hold is long — FMA 30.0 s, FakeMusicCaps 10.0 s, SONICS 33–240 s,
musan-music 216 s — and every voice source is short — CFAD 3–5 s, LJSpeech 6.8 s, MLAAD 7.1 s,
zeroth 7.9 s. Holding out one music source leaves the others, still long.

**Why that is worse than a corpus fingerprint, not better.** Source-grouped validation is the tool
this project uses to catch shortcuts, and this is a shortcut it cannot catch: it looks like a
genuine, generalising signal *within our corpus*. In the **test set** it is absent — every file is
4–60 s whether it is music or speech — so a model that learns "long ⇒ music" scores well in
validation and loses the presence heads on the leaderboard. It is the one confound so far that
source-grouping actively hides.

The crop policy ([03 C5](03-pool-c-real-instrumental.md), [04 D2](04-pool-d-fake-instrumental.md))
is therefore not a tidy-up. It is what makes the presence heads honest, and
[02 B0](02-pool-b-fake-voice.md) shows it has to hold *within* a source too — CFAD's own real and
fake halves differ at AUC 0.725 on duration alone.

---

### The sharpest case: three header fields identify cell 8 exactly

`(orig_sr, orig_channels, container)` over the whole corpus:

| tuple | A | B | C | D | E | cell8 |
|---|--:|--:|--:|--:|--:|--:|
| `16000 / 1 / mp3` | 0 | 0 | 0 | 0 | 0 | **49,074** |
| `16000 / 1 / wav` | 426 | 0 | 660 | 27,605 | 7,626 | 0 |
| `22050 / 1 / wav` | 13,100 | 16,006 | 0 | 0 | 0 | 0 |
| `44100 / 2 / mp3` | 0 | 0 | 7,490 | 0 | 0 | 0 |
| `16000 / 1 / flac` | 22,720 | 0 | 0 | 0 | 0 | 0 |

**One tuple covers every cell-8 row and nothing else.** Cell 8 is `1` on *all four heads* and is
32% of the corpus, so those three fields alone label a third of it perfectly — which is where
`orig_sr` 0.962, `orig_channels` 0.957 and `id3_tag` 0.962 on `music_fake` come from. `encoder`
is the same story with a finer edge: **`LAME3.100` appears on all 49,074 SONICS files and on no
other file in the corpus**, while `fma`'s mp3s carry eighteen *other* LAME versions.

⚠️ Adding a whole-file partition sourced from one publisher makes the shortcut *worse*, not
better, and that is a property of the acquisition rather than of SONICS. A second cell-8 source
from a different publisher would do more for this gate than any transform.

🔴 **The right reading is that there is exactly one confound — corpus identity — and fifteen
columns are each a proxy for it.** The source-grouped collapse says the same thing from the other
side. The fix is correspondingly not a list of knobs but a property: **every file must leave the
render chain having been through one identical encode** — one container, one codec, one bit depth,
one bitrate policy, tags stripped — so that no header field can name its source. Anything less
leaves a column that still can.

---

## 🔴 X1d — X1 on the **decoded audio**: the duration shortcut survives the chain

Run 2026-09-14 over all 58,885 S-tier rows (`eda.analyze.signal.signal_audit`).
X1, X1b and X1c all read `files.parquet` — metadata, which every fix is a render-time
transform against. This asks the version that cannot be transformed away: **is the label still
predictable from the audio itself, after the 16 kHz chain the competition mandates?**

Run on the `chain` plane, because that is the only plane a model ever sees.

| head | n | ungrouped | **source-grouped** | |
|---|--:|--:|--:|---|
| `voice_present` | 58,885 | 0.933 | **0.586** | collapses |
| `music_present` | 58,885 | 0.902 | **0.852** | 🔴 **survives** |
| `voice_fake` | 14,426 | 0.884 | **0.401** | collapses |
| `music_fake` | 32,265 | 0.898 | unmeasurable | 4 source groups |

**`music_present` holds 0.852 after a whole publisher is held out**, against a gate of 0.60 — and
its top features name the cause: `longest_valid_span_s` **0.943**, `duration_s_decoded` **0.922**.
[X1c](#-x1c--with-13-sources-grouped-auc-becomes-measurable-and-one-head-refuses-to-collapse)
found duration at 0.933 grouped on the M tier and predicted exactly this: resampling does not
change how long a file is, so nothing in the render chain touches it.

The two voice heads collapse — 0.586 and 0.401, the second below chance — which is the same
answer X1c gave and is the good news in this table.

### What the chain *does* remove

| head | native | chain | |
|---|--:|--:|---|
| `music_fake` | **0.999** | 0.898 | −0.10 |
| `voice_present` | 0.956 | 0.933 | −0.02 |
| `music_present` | 0.931 | 0.902 | −0.03 |
| `voice_fake` | 0.894 | 0.884 | −0.01 |

The 0.10 the chain takes off `music_fake` is visible in which features leave the top five:
`mel_bands_flat_native` (0.876) and `effective_bandwidth_hz_native` (0.795), both of which are the
**native sample rate** wearing an acoustic name. That is R1's premise confirmed a second way —
[00 §4d](00-harness.md#4d---r1-on-the-whole-corpus-ten-of-fourteen-sources-lose-nothing) has the
per-source version. `duration_s_decoded` sits at 0.946 on *both* planes.

### ⚠️ Two holdouts, and they answer different questions

| holdout | asks | `music_present` |
|---|---|--:|
| `source_name` | does this survive an **unseen publisher**? | 0.852 |
| `group_key` | does this survive an unseen **clip or speaker** from a publisher already in training? | **0.902** |

`group_key` is **not** the stricter one, and reading it as such is the mistake to avoid. Holding
out one FakeMusicCaps clip leaves 5,520 others in training, so an archive-level confound is still
fully available — which is why it scores *higher*.

⚠️ The group-keyed figure moves when a key is corrected: it was 0.894 when first measured and is
**0.902** after `compspoof-env-bonafide` went from 6 groups to 10,710
([05 E1b](05-pool-e-noise.md)). The source-grouped 0.852 did not move, because the archive holdout
does not depend on the key. That asymmetry is the point of this table. It is what makes `music_fake` measurable at all
(5,764 groups against 4 sources, giving **0.892**), and that number must never be quoted as an
archive holdout.

⚠️ A high AUC here is not automatically a defect — real and vocoded audio genuinely differ, and a
detector is supposed to find that. What makes a number a **shortcut** is surviving the archive
holdout on a feature the test set does not share: pool C is 30 s and pool D is 10 s
([03 C8](03-pool-c-real-instrumental.md), [04 D8](04-pool-d-fake-instrumental.md)) while the test
set is 4–60 s for both.

---

## X2 — 🔴 The metadata-leak question (A5) — Tier A, and the one genuinely open feature question

**Compute.** X1's model, but treated as a **candidate submission feature** rather than as a
diagnostic: measure its AUC on our own generator-disjoint, source-disjoint split, then measure it
again after pushing every file through the test-chain normalizer (16 kHz, mono, the containers the
test set uses). Report the drop. Extract the same features from the 3 dummy test files and check
whether they are even *in range*.

**Why it is meaningful.** [architecture/09](../architecture/09-open-questions.md) A5 and
[PROGRESS](../../PROGRESS.md) both flag this as unresolved and decision-changing: per-file metadata
is **legal under rule 2.4** — it is computed from one file, with no cross-file statistics — and may
separate REAL from FAKE almost for free. *"If the leak is real it dominates every architecture
decision."* It may equally be a pure CV mirage: our metadata reflects **our** archives, and DACON
built theirs.

**What we discover.** The size of the mirage in our own corpus, and — the only real evidence
available — whether the dummy files' metadata is consistent with a chain that preserves any of it.
🔴 If DACON normalized every test file to 16 kHz through one pipeline, then container and rate
carry *their* pipeline, not the generator's, and the entire feature is worth zero on the
leaderboard while scoring beautifully in CV.

**⚠️ X1 and X2 are the same measurement read in opposite directions**, and confusing them is easy.
X1 wants the AUC **low** (no shortcut). X2 asks whether a high AUC is exploitable. They cannot both
be satisfied by the same number: **a metadata AUC high enough to be worth shipping is, by
construction, a shortcut that will not generalize.** Resolve it in X1's direction — neutralize —
unless E-S1 produces positive evidence from the dummy files that the test chain preserves the
signal. Record the decision either way; it is the kind of thing that gets silently re-litigated.

---

## X3 — Adversarial validation (E-A2) ⚠️ Tier A

**Compute.** Train a classifier to separate TRAIN from VAL, then TRAIN from a **proxy-eval** slice
built by pushing held-out generators through the test-chain normalizer. Report AUC and the ranked
discriminating features.

**Why it is meaningful.** ★ `[G2Net 2021, 3rd]` used it to confirm train/test similarity at
**AUC ≈ 0.5**. A high AUC here means our own splits differ systematically — corpus-identity
leakage ([data/09](../data/09-risks-and-checks.md) R2) — and that the VAL score is measuring
domain transfer rather than generator transfer.

**What we discover.** Whether artifact-family disjointness has accidentally produced
*source* disjointness too, which would make every VAL number pessimistic in a way that does not
transfer to the leaderboard.

⚠️ We cannot run this against DACON's test set — 3 dummy files. Against our own splits only. That
is a real limitation, and it is why the **shadow split** matters: `shadow_of` / `shadow_b` are the
repo's instrument for domain-shift stress testing and have **never been built**
([PROGRESS](../../PROGRESS.md)). X3 is what they are for.

---

## X4 — 🔴 Dummy-file forensics (E-S1) — Tier S, blocking, and it unblocks the render chain

**Compute.** On `TEST_0000–0002.wav`: `ffprobe` headers and every tag; LTAS with the rolloff shape
at the top of the band; near-Nyquist resampler shelf; LUFS / peak / DC; dither noise floor;
leading/trailing silence; inter-channel correlation. Write `_shared/signal_chain.yaml`.

**Why it is meaningful.** *"The **only** direct evidence of the organizers' signal chain"*
([data/07](../data/07-eda-plan.md)). Everything symmetric depends on reading it right: a rolloff
near ~3.4 kHz would reveal how they represent 전화채널; the presence or absence of loudness
normalization decides P-A1; the silence policy decides P-A2.

**What it changes — immediately.** 🔴 `normalize` in the render path is **unparameterized until
this lands**, so A-S1, the highest-leverage augmentation step, is *structurally present and doing
nothing today*. Three files, an afternoon, and it turns a no-op back into the step it was designed
to be.

⚠️ **Three files is three files.** Anything read from them is an `n = 3` inference, marked ☆ at
best. Treat the signal chain as a hypothesis with a confidence, not as ground truth, and prefer
policies that are *robust* to getting it wrong over policies that are optimal if it is right.

---

## X5 — The composition confound, over the spec stream ⚠️ Tier S

**Compute.** Run `training.audit.audit_specs` against the real corpus once it is built, and read
off the per-cell composed fraction `f_c`, the marginal `P(composed | FILE_FAKE)`, and the
within-presence-stratum version. No audio decode is required — `sample_spec()` is pure and
label-free, so auditing a stream costs nothing.

**Why it is meaningful.** [data/02](../data/02-label-taxonomy.md)'s composition trap is the one
confound this project has already solved *in design* — `f8`, `balance_marginal_composedness`, and
the C1/C3 checks in `SamplerConfig.__post_init__`. But the reference cell mix *was itself trapped
when first published* (`P(mixed|FAKE)=0.667` vs `P(mixed|REAL)=0.300`, so "is a mixed file"
predicted FAKE at 0.769), and the shipped mix was solved against a **synthetic** corpus whose pool
sizes differ from the real one.

**What we discover.** Whether the solved mix survives contact with the real pool sizes. It may not:
`f_c` equality requires enough whole-file material in cells 5 and 8 to match the composed fraction,
and [C1](03-pool-c-real-instrumental.md)'s licence intersection may leave too little real music to
do it with.

**⚠️ Do not read this as optional because the code enforces it.** The enforcement is on the *mix*,
not on whether the pool can supply it — a mix that satisfies C1 and C3 and cannot be drawn from the
corpus fails later, in `build_folds`, with a message about fold feasibility rather than about
composedness.

---

## X6 — Metadata role assignment (E-S4) ⚠️ Tier S

**Compute.** Every column in `eda/*/files.parquet` assigned to exactly one of **feature** /
**split key** / **leakage risk**, with missingness and cardinality quantified. Freeze it as
`_shared/roles.md`.

**Why it is meaningful.** ★ `[BirdCLEF playbook 2026]`. A column used as a feature and as a split
key at the same time is a leak by construction; a column with 90% missingness used as a split key
silently collapses its groups. This is what determines the grouping keys *before* any fold is
built, and [data/12](../data/12-acquisition-status.md) records what happens when it is decided
late: a `source_name` at the wrong granularity validates clean and makes `build_folds` infeasible
at every fold count.

**What we discover.** The frozen list, and the cardinality of every proposed grouping atom —
which is [A3](01-pool-a-real-voice.md), [B2](02-pool-b-fake-voice.md), [C3](03-pool-c-real-instrumental.md),
[D1](04-pool-d-fake-instrumental.md) and [E4](05-pool-e-noise.md) collected into one table that
`build_folds` can be run against before a single file is copied.

---

## 🔴 X6b — the publisher's key, joined: what path depth was hiding

Run 2026-09-13 (`eda.groupkeys`, `eda keys`). X6 asks where the grouping key lives; `eda analyze`
had been answering with **path depth** and saying on every run that seven of thirteen sources
*"need the publisher's own key"*. That single sentence was covering three different situations, and
only one of them was work.

| | before | after | what it was |
|---|--:|--:|---|
| `sonics` | 1 | **5** | `algorithm` in `fake_songs.csv`; 49,074 of 49,074 join |
| `fakemusiccaps` | 5 | **5,521** | the MusicCaps clip, not the generator |
| `musan-music` | 5 | **82** | `ANNOTATIONS` field 4, the artist |
| `rirs-isotropic-noise` | 0 | **10** | the room in the filename, channels merged |
| `wavefake` | — | **2** | the voice, not the vocoder ([02 B0b](02-pool-b-fake-voice.md)) |
| `ljspeech` | 1 | **1** | one speaker. No key exists, and none can |
| `musan-noise` / `musan-speech` | 2 / 2 | 2 / 2 | collection only; MUSAN publishes no finer key |

**G-EDA3 went from 7 of 13 short to 5 of 14**, and — more useful than the count — it now separates
two clauses that were one list: *"2 with no publisher key, path depth only"* against *"3 keyed and
genuinely short — this is the count, not a gap"*. SONICS reported as 1 group read exactly like
LJSpeech reported as 1 group, and they are opposite problems.

### Two of these are corrections, not refinements

🔴 **FakeMusicCaps' atom is the clip, and path depth found the generator.** The pool is 5,521
MusicCaps clips rendered by five models — 5,521 files in each of the five directories, 5,521
distinct stems in their union, exactly. Grouping by generator therefore puts the *same clip* in all
five groups: hold out `musicldm` and its 5,521 clips are still in the training fold four times
over, rendered by the other four models. The publisher's key here **lowers** the count from 5 to
5,521 groups of five, and taking the larger number would have been taking the wrong one. This is
the 0.27-weight pool.

🔴 **WaveFake is two voices wearing ten vocoders**, and it shares its key with `ljspeech` across
the source boundary — 137,366 files in one group, spanning pools A and B. See
[02 B0b](02-pool-b-fake-voice.md).

### The key is on every row, including the path-derived ones

⚠️ An earlier revision left `group_key` null wherever no provider existed, which read as tidy and
was useless for the case that motivated the work: `cfad-fake`'s eleven generators live in the path
and nowhere else. Every row now carries a key; `group_key_kind` says whether it came from the
publisher, from a declared single atom, or from the path — and for the last, `eda keys` prints the
depth and the reason that depth is the publisher's key.

⚠️ **A path key is the directory prefix, never the name at that depth.** With `train/spk1` and
`test/spk1`, keying on `spk1` merges two speakers the publisher kept apart — the trap this
document's X6 names, and the one place it could have been walked into.

### What it changed about the draw

`sample.spread_by: [group_key]`, decided 2026-09-13. `stratify_by` still sets the budget — 2,000
files per source — and the spread decides how that budget is spent inside one. Same 15,086 sampled
files either way; what changes is which:

| | before | after |
|---|---|---|
| `cfad-fake`, 27 generators | 17–108 each | **74–75** |
| `zeroth-korean`, 115 speakers | 1–66, and 114 covered | **3–18, all 115** |
| `fma`, 156 shards | — | 2,000 over all 156 |

⚠️ Widening `stratify_by` instead would have multiplied the budget by the group count: 54,000
files from `cfad-fake` alone, against a whole-corpus decode budget of ~92,000.

⚠️ The allocation is water-filling, **smallest group first**. Walking in key order spends the
remainder only forwards and runs out — measured on `fma`, 1,953 of a 2,000 budget, 47 files
quietly unmeasured because the groups that could have absorbed them had already been passed.

---

## X7 — The data memo (E-S3) ⚠️ Tier S, the exit condition

**Compute.** One page: file inventory, schema, label counts per cell, suspected leakage variables,
risk list — with the numbers from X1–X6 inline.

**Why it is meaningful.** ★ `[BirdCLEF playbook 2026]`: *"Do not advance to hyperparameter tuning
until the data memo explains class imbalance, domain shift, and the first leakage hypothesis."*
It is the artifact that says the EDA is finished, and the three things it must explain are exactly
the three this corpus is most at risk from.

**Output.** `docs/EDA/data_memo.md`, versioned, regenerated on every corpus change.

---

## The standing rule

🔴 **Re-run X1 (shortcut audit) and X3 (adversarial validation) after *every* corpus change.**
They are cheap, they take minutes against a metadata table that already exists, and they catch most
of the failure modes in [data/09](../data/09-risks-and-checks.md) before those cost weeks.
