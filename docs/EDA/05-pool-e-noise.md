# 05 — Pool E: Non-Musical Sound

**11.5 GiB.** `musan` (10.3 — `noise/`, `speech/`, `music/` partitions) and `rirs-noises`
(1.2 — `pointsource_noises/`, `real_rirs_isotropic_noises/`, `simulated_rirs/`), plus
CompSpoof's `env_sources/*/bonafide/` (EnvSDD, VGGSound ambient). Target ~10 h — the smallest pool,
and the one that has already cost the most.

Pool E carries cell 9 and the noise component of composed samples. It asserts
`label_voice_present = 0, label_music_present = 0`. It is **robustness material only** — but it is
also where this project's worst split leak came from, so its EDA is a forensics pass as much as a
census.

---

## E0 — Decisions taken 2026-09-12, and what they cost

E1 and E2 were run over the whole sources rather than the 90-row smoke corpus, and both RIRS
sources are now `blocked:` in `configs/eda.yaml` — blocked rather than deleted, so the finding
keeps its evidence and cannot be quietly rediscovered.

| source | measured | decision |
|---|---|---|
| `rirs-pointsource` | 843 files, **843 of 843 sha256-identical to `musan-noise`**, 843 distinct digests on each side, **0.00 h** MUSAN does not already hold | dropped — it was never a second source |
| `rirs-isotropic` | 417 files, median **1.365 s**, **314 of 417 below the 4 s floor**, channels 1/2/8/16/30 (38 mono), 0.89 h | out of pool E — impulse responses, material for the reverb transform (E2). ⚠️ **Partly wrong — see E0d** |

Both earlier figures came from the smoke corpus and understated the case: the leak was recorded as
88 files (it is 843) and the isotropic census as 79 of 90 rows (it is 314 of 417).

**What it cost.** Pool E consolidates to **930 rows / 6.23 h from one source**, `musan-noise`, and
`gates.min_groups_per_role` is 6. The noise role therefore has **no honest validation**:
`build_folds` cannot rotate on a single source, and G-EDA3 names `musan-noise` at 2 path-derived
groups. This is E4's question, answered, and the answer is that the noise role **is** the binding
constraint on fold count.

The drops did not cause it — they revealed it. One of the two was a byte-identical copy of MUSAN
and the other was not noise, so pool E has had one independent source all along. The remedy already
in the plan is CompSpoof's `env_sources/<corpus>/bonafide/` halves (E4), which moved from wave 2 to
blocking and **landed the same day** — see E0b.

**What it bought.** G-EDA5 went from 843 cross-source duplicate groups to **0** — it now passes.
The single remaining duplicate group is unrelated: 5 MLAAD files under `fake/lb/VITS2-Claude/`
that are byte-identical to each other, from five different utterances.

---

## E0d — 🔴 E0's second decision was half wrong: that directory holds two kinds of file

Found by re-checking E0 rather than by a new measurement, which is the point of re-checking it.
`real_rirs_isotropic_noises/` is **not** one kind of material. Its filenames carry the distinction
and its directory structure does not:

| kind | pattern | n | median | min | below 4 s | channels | hours |
|---|---|--:|--:|--:|--:|---|--:|
| impulse responses | `*_rir_*`, `air_*binaural*` | **325** | 1.25 s | 0.10 s | **314** | 1/2/8/16/30 | 0.14 |
| isotropic noise | `*_noise_*` | **92** | 30.00 s | 10.00 s | **0** | 8 (90), 30 (2) | 0.76 |

**Every one of the 314 sub-floor files is an impulse response, and not one noise recording is
below the floor.** The 1.365 s median and the 314 count that E0 gave as its reason are statistics
of the *mixture*; quoted against the whole source they describe a property that only half of it
has. Blocking the RIR half stays correct — an IR is a convolution kernel and belongs to the reverb
transform. Blocking the other 92 was not argued for, and they are real ambient recordings from
**10 distinct rooms** (`largeroom1/2`, `mediumroom1/2`, `smallroom1/2`, `cirline_ofc`, and
`simroom1/2/3`), 0.76 h, all 30 s, **none duplicated anywhere else in the corpus**.

A signal check confirms the two halves behave differently, using the S tier on the blocked files
directly — fraction of energy in the first 100 ms, which is ~1 for a kernel and ~0 for a
stationary recording:

| set | frac in first 100 ms (median) | peak position (median) |
|---|--:|--:|
| `rirs-isotropic` `*_noise_*` sample | 0.003 | 0.573 |
| `musan-noise` | 0.000 | 0.358 |
| CompSpoof EnvSDD bonafide | 0.024 | 0.451 |

The noise half looks like MUSAN, not like a kernel.

**Acted on the same day.** `SourceSpec.name_glob` was added — narrow on purpose, and paired with a
`name_glob_reason` for the same reason `exclude` is — and the source is now two:
`rirs-isotropic-rir` (blocked, `*_rir_*` and `air_*`) and `rirs-isotropic-noise` (pool E,
`*_noise_*`). No file can belong to both, and a test asserts that rather than trusting the globs.

Pool E is now **14,194 rows across three runnable sources and four independent groups**:
`musan-noise`, `rirs-isotropic-noise` (RVB2014), and CompSpoof's AudioCapsEnv, EnvSDD and
VGGSoundEnv. The floor is 6, so it is still short — but short by two rather than three, and the
missing groups are now a fetch question rather than a bookkeeping one.

---

## E0b — CompSpoof lands, and pool E is a pool again

Fetched 2026-09-12 with `fetch_from_s3.py compspoof-v2 --only '*_source.tar.gz' --only '*_label.csv'`
— **21.8 GB of the 111.8 GB** in the store, sha256-verified, because the rest is speech and
mixtures we hold better sources for. Registered as `compspoof-env-bonafide`, pool E.

**13,172 files, 14.64 h.** Pool E is now **14,102 rows / 20.9 h across two sources**, past the ~10 h
target for the first time. `enumerate_source` returns exactly 13,172 of the 122,303 wav files under
`ESDD2` — the other 109,131 are excluded, and the exclusion is the decision:

> `exclude: [spoofed, speech_sources, mixed_audio, original_audio]`

⚠️ `spoofed` is the one that matters. It is **generated environmental sound** — 12,273 of it
AudioLDM — which is neither pool E (real) nor pool D (fake **music**), and has no cell of its own.
Without the exclusion, 12,273 AudioLDM clips would carry the label *real noise*.

### What the census says about it

| corpus | n | sample rate | channels |
|---|---|---|---|
| EnvSDD | 6,696 | 16 kHz | mono |
| VGGSoundEnv | 4,202 | 44.1 kHz | stereo |
| AudioCapsEnv | 2,274 | 44.1 kHz (one file 22.05 kHz) | 1,578 stereo / 696 mono |

🔴 **Three findings, each of which changes something downstream.**

**1. Every file is exactly 4.00 s.** Not a median — 13,172 of 13,172, one distinct value. Pool E's
new source is a constant, and `duration_s == 4.00` separates it from `musan-noise` (median 11.13 s)
perfectly. Test clips are 4–60 s, so a 4.00 s noise bed cannot cover one without tiling, and tiling
is a render-time decision that must be identical on train and test (R2).

**2. Sample rate and channel count are a corpus fingerprint — on the native plane only.** EnvSDD is
the only 16 kHz source and VGGSoundEnv the only all-stereo one, so `(orig_sr, orig_channels)`
recovers the corpus almost exactly. The 16 kHz chain resample collapses it. This is the first place
in the corpus where the two measurement planes of R1 differ in what they leak, which makes pool E
the cheapest place to *test* that the chain plane is measuring what we think it is.

**3. `eval_source` and `test_source` are splits of the same corpora, and they overlap.** E1 found
**32 duplicate groups over 112 rows** inside the source, **18 of them spanning the two splits**.
They are not two independent halves, and the true independent-group count for pool E is **3**
(AudioCapsEnv, EnvSDD, VGGSoundEnv) plus MUSAN's two — not the 6 that path depth will report by
counting `eval_source/...` and `test_source/...` separately. That is the LJSpeech caveat in G-EDA3,
one pool over, and here it errs optimistic.

No file in the new source is byte-identical to anything in `musan-noise`.

---

## E0c — The S tier over pool E, and what it settled

14,102 files, both planes, **0 decode failures, 21 minutes** (32 threads; `user + sys` is 41
minutes, so the GIL holds the speedup to ~2× and the S tier is I/O- and decode-bound, not
parallel-bound). Pool E is measured in full, so these are population statements, not samples.

🔴 **Pool E answered the project's open R1 question, and the premise holds.** The 16 kHz chain
destroys the native bandwidth fingerprint: 15,246 Hz and 15,289 Hz of median effective bandwidth
for the two 44.1 kHz corpora become 8,000 Hz, and what looks like a residual is content rather than
format. The resampler does leave a skirt signature — AUC **0.940** on *identical* audio, **0.611**
once content varies — so it is a train/test domain difference worth one cheap transform, not a
shortcut. The numbers, the controlled experiment and the three checks that overturned this
section's own first conclusion are in
[00 §4c](00-harness.md#4c---r1s-premise-measured-on-pool-e-it-holds); they belong there because
they are a property of the chain, not of pool E.

What is specific to pool E:

| | median `rms_dbfs` | median `crest_factor_db` | silence ratio |
|---|---|---|---|
| AudioCapsEnv | −22.8 | 16.5 | 0.000 |
| VGGSoundEnv | −23.6 | 16.7 | 0.000 |
| EnvSDD | **−44.8** | 12.9 | 0.025 |
| musan-noise | −19.8 | 19.8 | 0.010 |

⚠️ **EnvSDD sits 20 dB below every other noise source, and the chain does not touch it** — it is
natively 16 kHz, so both planes are identical. A noise bed's level is exactly what an SNR-controlled
mix has to normalize away; without that normalization, "quiet noise bed" *is* the corpus label, and
E4's independent groups become distinguishable by a single scalar. The mix should set component
level from a measured target, not from the file as published.

---

## E0e — E3 answered: 58.4% of pool E can supply a 4 s window

With the split landed, pool E is **14,194 rows / 21.7 h** and the S tier covers all of it. E3's
question — *the real size of pool E, after the sampler floor* — has an answer:

| corpus | n | with a ≥4 s span | % | median longest span |
|---|--:|--:|--:|--:|
| AudioCapsEnv | 2,274 | 1,663 | 73.1 | 4.00 s |
| VGGSoundEnv | 4,202 | 2,888 | 68.7 | 4.00 s |
| `rirs-isotropic-noise` | 92 | 62 | 67.4 | **30.00 s** |
| musan-noise | 930 | 563 | 60.5 | 6.53 s |
| EnvSDD | 6,696 | 3,110 | **46.4** | **2.65 s** |
| **pool E** | **14,194** | **8,286** | **58.4** | — |

**8,286 files and 14.45 h survive the floor, not 14,194 and 21.7.** The newly admitted RVB2014
recordings have the longest usable spans in the pool by a factor of five.

⚠️ **EnvSDD's 46.4% is the level finding wearing a different hat, and the two must be read
together.** Its median `rms_dbfs` is −44.8, and `SILENCE_DBFS` is −50: the corpus's *average* level
sits 5 dB above the threshold that defines silence, so an absolute criterion scores nearly half of
it as unusable. That is a real property of the files as published — the sampler does not renormalize
before looking — but it is not a statement that the audio is empty. It is the same argument for
setting component level from a measured target rather than from the file, one step further on: get
the level wrong and even the usability census is wrong.

---

## E1 — 🔴 Cross-source duplicate sweep — Tier S, and it must run over **all five pools**

**Compute.** Two passes over every file in the corpus, not just pool E:

1. **Byte identity** — sha256, grouped. This is what caught the RIRS/MUSAN overlap.
2. **Content fingerprint** — the re-encoding-robust hash from [00 §4](00-harness.md) (quantized mel
   mean), grouped with a near-match tolerance. Byte identity cannot see the same recording at a
   different bitrate.

Emit `_shared/duplicates.parquet` and populate `dup_group` for every matched set.

**Why it is meaningful.** It has already happened once, silently, and it invalidated every score
measured before it was found. `RIRS_NOISES/pointsource_noises/` is MUSAN's `noise/free-sound/` set
**redistributed** — 88 byte-identical files under the same basenames, different `file_id`,
different `source_name`, so `folds.grouping_atoms`' union-find could not link them and the same
recording sat in TRAIN on the fold where its twin was in VAL.
`validate_manifest` never looks at `sha256`. ⚠️ *"The cascade is the lesson"*: linking them
correctly collapsed pool E from 4 groups to 2 and `build_folds` immediately refused — revealing
that the earlier fold table had been feasible **only because of the leak**.

**🔴 The specific prediction worth testing first.** MUSAN's music partition ships as
`music/fma/`, `music/jamendo/` and `music/hd/` — **it is FMA and Jamendo material**, and we are
separately acquiring FMA (`fma_small`) and MTG-Jamendo (`raw_30s`). That is the same redistribution
pattern that produced the RIRS leak, on the **0.27-weight** music head instead of on noise, and
between pools that both feed pool C. The smoke corpus already uses `musan-music-fma` and
`musan-music-jamendo` as pool-C sources. **Check this before building any fold table.**
🔷 This is our inference from the directory naming, not a confirmed overlap — but the cost of
checking is one hash sweep and the cost of not checking is a corpus rebuild.

**What we discover.** Every duplicate group in the corpus, and whether MUSAN's music partition must
be dropped, deduplicated against FMA/Jamendo, or linked by `dup_group`.

**What it changes.** `dup_group` in the manifest; possibly a source dropped outright.
**Gate.** **G-EDA5** — zero shared sha256 and zero near-duplicate pairs across TRAIN/VAL in any
fold. Verified on the built folds, not asserted.

---

## 🔴 E1b — E1 pass 2: the method fell short, and following it found a real leak

Run 2026-09-14 over the chain-plane `ltas` vectors of all 58,883 usable S-tier rows
(`eda.analyze.signal`, blocked cosine, ~2 min).

### The method is weaker than E1 asked for, and the number says so

E1 pass 1 compares sha256. Pass 2 was meant to be the **content** fingerprint that catches a file
re-encoded or republished at another rate. Measured, the long-term average spectrum is not that:

| quantile of each file's *best* match | cosine |
|---|--:|
| median | **0.965** |
| p90 | 0.993 |
| p99 | 0.998 |

A median file already matches *some* other file at 0.965. At a 0.999 threshold the sweep returns
**868 pairs, none cross-source** — and decoding the top hits shows why that number cannot be read as
868 duplicates: `bus-helsinki-20-789-a_0` against `bus-vienna-38-1134-a_0` differ by up to **0.61 in
amplitude** and agree spectrally because buses sound like buses.

⚠️ So it is a **timbre** fingerprint, not an identity one. It shortlists candidates; it does not
confirm duplicates. A real pass 2 needs a landmark/chromaprint-style fingerprint or waveform
cross-correlation over the shortlist. `G-EDA5` stays honest about this: it passes on byte identity
and says so in its own detail line.

### What following the shortlist found

The hits pointed at CompSpoof, and the publisher's own filenames settled it exactly — no threshold
involved.

🔴 **CompSpoof is segmented, and its two splits share parent recordings.** One parent becomes many
4 s files, and over the full 13,172-file census:

* **292 parent recordings appear in both `eval_source` and `test_source`, across 1,060 files** —
  8.1% of the source, all under `EnvSDD`;
* **17 stems are byte-identical across the split**, e.g. `a001_11` filed under `TUTSED2017Dev` on
  one side and `TUTSED2016Dev` on the other, same sha256.

Our grouping key was the path's own answer — `<split>/env_sources/<Dataset>`, six tidy groups that
meet the floor. That key puts segment 0 of a Helsinki bus in train and segment 1 in validation.

**Fixed**: `compspoof-env-bonafide` now groups by the **parent recording**, one rule per subtree,
each read off the files rather than assumed —

| subtree | stem | parent |
|---|---|---|
| `AudioCapsEnv` | `Y-4B1PkgXOMI_80_seg000` | the YouTube id |
| `VGGSoundEnv` | `-3z5mFRgbxc_000030.mp4_chunk1` | the YouTube id |
| `TUTASC2019Dev` | `airport-barcelona-0-12-a_0` | the clip |
| `TUTSED*` | `a001_33` | the recording |
| `UrbanSound8K` | `100263-2-0-137` | the Freesound id |

**6 groups → 10,710**, and all 292 straddling parents are now exactly one group each — verified by
re-measurement, not asserted. The split is deliberately *not* in the key: it is what the path offers
and it is the wrong atom.

⚠️ Pool E is in `full_pools`, so there is no draw to invalidate and no S-tier number changed.

---

## E2 — Content-kind classification: noise recordings vs impulse responses ⚠️ Tier S

**Compute.** Per file: duration, and the shape of the energy-decay envelope — an impulse response
is an initial transient followed by monotone exponential decay; a noise recording is stationary.
Classify each pool-E row as `noise_recording` / `impulse_response` / `ambient`.

**Why it is meaningful.** *"Two of RIRS's three subdirectories are not noise at all"*
([data/12](../data/12-acquisition-status.md)). `simulated_rirs` was never taken;
`real_rirs_isotropic_noises` was, and is predominantly **real impulse responses** — measured median
duration 2.00 s, with **79 of 90 rows below the 4 s sampler floor**. It was dropped after the fact.
An IR is a **convolution kernel**, not an audio sample: it belongs in the augmentation registry as
a reverb transform (B12 — ☆ reverb via IR convolution at p=0.2 with a dry/wet mix), never as a
pool-E draw.

**What we discover.** Which rows are kernels. Two uses, not one: exclude them from pool E, **and
route them into the reverb transform**, which is currently unimplemented and for which we already
hold the material.

---

## E3 — Duration census against the sampler floor ⚠️ Tier S

**Compute.** Duration histogram per subdirectory, and the count above `duration_range[0]` = 4.0 s.
Report the **floor-corrected** file count per grouping atom.

**Why it is meaningful.** The `rirs_isotropic` failure mode in one sentence: 79 of 90 rows fell
below the floor, so `sampler.py` filtered them out and *"the group contributed nothing while still
counting toward the file tally."* A pool-E group that looks adequate in a file count and is empty
after the floor is exactly what makes `build_folds` refuse for a reason nobody can see.

**What we discover.** The real size of pool E — the number that decides the achievable fold count
on the noise role.

---

## E4 — Independent-group inventory ⚠️ Tier S

**Compute.** Group pool E by *true* origin after E1's dedup: MUSAN `free-sound`, MUSAN
`sound-bible`, CompSpoof `env_sources/EnvSDD/bonafide`, CompSpoof `env_sources/VGGSoundEnv/bonafide`,
and whatever survives from RIRS. Count them. Run `build_folds` feasibility on the noise role.

**Why it is meaningful.** *"Pool E's independent sources are scarcer than they look"* — MUSAN
supplies only two, and `build_folds` refuses a rotation where a fold's TRAIN or VAL side has no real
noise to draw from. The two CompSpoof bonafide sets are what made the smoke corpus's 2-fold table
possible at all.

**What we discover.** Whether the noise role is the binding constraint on the fold count, which is
a different question from whether pools A–D are.
⚠️ **Worth knowing before the next corpus is built**: an apparent new noise source may be a
redistribution of one already held. E1 is how you find out; this is why it matters.

---

## E5 — Cell-9 viability and the `PRESENT=0` question ⚠️ Tier B

**Compute.** Over pool E: `vad_speech_ratio` (Silero) and PANNs `Music`/`Speech` scores. Flag any
file where either fires — a "noise" file containing speech or music contradicts cell 9's
`PRESENT=0, PRESENT=0`.

**Why it is meaningful.** MUSAN's `noise/` partition contains real-world recordings, and real-world
recordings contain distant speech and background music. A cell-9 sample with audible speech asserts
`VOICE_PRESENT = 0` while carrying voice — and #417333 A3 confirms `PRESENT=1` at **any** duration,
which makes the assertion wrong at any level of audibility, not just at a loud one.

**⚠️ And cell 9 itself is not settled.** [data/12](../data/12-acquisition-status.md) records that
A3 answered only the duration half: *"whether files exist with both `PRESENT=0`, and whether
AI-generated environmental sound is `FILE_FAKE`, remain unanswered."* So this measurement tells us
how much cell-9 material we have that is *defensibly* cell 9 — which bounds how much weight to put
on a cell whose definition is still open. The cell mix in `configs/run_default.yaml` currently gives
cell 9 a **0.115** share, the third largest.

**What it changes.** Reassignment (F-A1) for files with evidenced components, and the cell-9 share
in the sampler's `cell_mix` if defensible material is scarcer than the mix assumes.
⚠️ Changing `cell_mix` re-triggers the C1/C3 soundness check in `SamplerConfig.__post_init__` —
a mix that violates either fails at load, by design.

---

## E5b — ✅ Answered 2026-09-16: 11,901 files are defensibly cell 9, and 2,113 are not

Silero VAD over all 14,194 decoded pool-E rows, chain plane
(`eda.analyze.screens.cell9_viability`). Cell 9 is `(0, 0, None, None)`.

| source | n | ≥ 4 s | unmeasured | **voice-evidenced** | **viable** | share |
|---|--:|--:|--:|--:|--:|--:|
| `compspoof-env-bonafide` | 13,172 | 13,172 | 0 | **2,006** | 11,166 | 84.8% |
| `musan-noise` | 930 | 737 | 1 | 107 | 643 | 69.1% |
| `rirs-isotropic-noise` | 92 | 92 | 0 | 0 | 92 | 100% |
| **total** | **14,194** | **14,001** | **1** | **2,113** | **11,901** | **83.8%** |

**11,901 files clear the sampler's 4 s floor with no speech evidence at all**, which comfortably
supports the **0.115** cell-9 share `configs/run_default.yaml` currently gives — the third largest.
The cell is viable on supply; what E5 said was unsettled was its *definition*, and this measurement
does not settle that.

🔴 **2,113 files assert `VOICE_PRESENT = 0` while carrying speech evidence.** The threshold is
`vad_speech_ratio_50 > 0`, i.e. **any** evidence at all, and that is deliberate rather than
conservative: [#417333 A3](../competition/05-talkboard-qa.md) confirms `PRESENT = 1` at *any*
duration, so choosing a tolerance would mean choosing a level of audibility the rules say does not
exist. These are reassignment candidates (F-A1) and they are the same work list as **G-EDA6** —
CompSpoof's environmental bonafide half carries 15.2% of it, which is what a real-world
environmental corpus contains.

⚠️ **Every number above is an upper bound, and the reason is recorded in the function itself.**
Silero is a *speech* VAD: it cannot evidence music, so the `music_present = 0` half of cell 9 goes
unchecked here. [09 §7](09-next-steps.md) records why PANNs was declined and
[RESULTS §5.1](RESULTS_FOR_ANALYSIS.md) records the misreading that followed from forgetting this
once already. A file of café ambience with background music is counted viable above and is not.

⚠️ One file's VAD did not run. It is reported as `unmeasured` and is **not** counted viable — `na`
is not `pass`, which is the same tri-state the gates use.
