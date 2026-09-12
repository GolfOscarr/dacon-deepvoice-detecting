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
