# 10 — Preprocessing, Filtering & Salvage

How raw pool files become training-ready material, and **where I stop and ask you**.

---

## 0. Two governing principles

### 🔴 P1 — Filter for *label-evidence sufficiency*, not for cleanliness

The instinct "filter out audio with excessive noise" is right in most ML tasks and **wrong here**.
Our test set explicitly contains **전화채널 audio** — narrowband, codec-degraded, noisy by
construction — plus MP3 at unknown bitrates. Noise is a **property of the target domain, not a
defect**. Discarding noisy training audio makes the model worse on precisely the samples that are
hardest.

The legitimate version of your concern is different and sharper:

> Drop a file when the **label is no longer supported by evidence** — e.g. a "voice-only" file
> where the voice is inaudible under noise, so the model is asked to learn `VOICE_FAKE` from
> something that carries no voice.

So the filter question is never *"is this clean?"* but *"can a human still tell that the labelled
component is present, and could its real/fake status plausibly be judged?"*

### 🔴 P2 — Transformations must be train/test symmetric; filters are train-only

| Operation type | Applied to test set? | Risk |
|---|---|---|
| **Preprocessing** (resample, downmix, DC removal, loudness) | **Yes — identical code path** | Asymmetry ⇒ the model learns our pipeline, not the artifact |
| **Filtering** (drop / quarantine a file) | **No — impossible.** We must predict all 1,200 files | Over-filtering ⇒ training distribution drifts away from test |
| **Salvage** (use only part of a file) | No — training-only | Splice artifacts becoming a label cue |

We cannot filter the test set. Therefore **every filter is a distribution-shift decision**, and
that is exactly why they need review gates (§5).

---

## 1. Two-layer architecture

```
LAYER 0 — IMMUTABLE ORIGINALS
   pools/<pool>/<source>/<file>            never modified, never deleted
   └── this is what ships to DACON

LAYER 1 — CURATION (offline, once per corpus version)   ← human gates live here
   sidecar metadata only:
     verdict.parquet     keep / quarantine / drop  + reason + threshold version
     validity_mask.npz   per-file boolean mask at 50 ms resolution
     quality.parquet     per-segment scores (energy dB, SNR proxy, clipping, VAD, tags)

LAYER 2 — RUNTIME (on the fly, seeded)
   load original → apply validity mask → sample crop → compose → augment → normalize
```

🔴 **Nothing in Layer 1 rewrites audio.** Every curation decision is a *sidecar annotation*, so any
threshold can be revisited by regenerating metadata — no re-download, no lost data, and the corpus
we ship to DACON stays byte-identical to what we obtained.

---

## 2. Preprocessing catalog (P) — symmetric, train **and** test

Tier legend in [07](07-eda-plan.md#tier-legend-used-in-05-06-07).

| # | Tier | Method | Purpose | Output / parameters | Gate |
|---|---|---|---|---|---|
| **P-S1** | S | **Robust decode** — any of MP3/WAV/FLAC, corrupt-frame tolerance, force float32, never assume extension | The eval server hands us mixed containers; a decode crash burns one of 3 daily submissions ([competition/02](../competition/02-submission.md)) | `load_audio()` with per-file try/except and a fallback path | — |
| **P-S2** | S | **Resample to 16 kHz**, one fixed resampler | Test-chain parity ([06 A-S1](06-augmentation-spec.md)) | Resampler + params from `signal_chain.yaml` | **G1** |
| **P-S3** | S | **Channel policy** — downmix / dual-channel / mid-side | Test set has mono *and* stereo per sample. Mono-duplicated "stereo" may itself be a cue — or a trap ([07 E-B4](07-eda-plan.md)) | One documented policy applied everywhere | **G6** |
| **P-S4** | S | **DC offset removal** | Harmless, symmetric, prevents a trivial corpus-identity cue | — | — |
| **P-A1** | A | ⚠️ **Loudness normalization** (peak or LUFS) | Removes level as a corpus-identity shortcut — **but** may also remove a genuine cue (generators have characteristic loudness). Only defensible if the organizers normalized too | Decide from `signal_chain.yaml`; ablate both ways | **G6** |
| **P-A2** | A | ⚠️ **Silence trimming** | Silence statistics are a **known ASVspoof shortcut** ([survey/01](../survey/01-sota-speech.md)). Trimming kills the shortcut and the cue together | Prefer *not* trimming; inoculate with `A-A11` silence edits instead. Ablate | **G6** |
| **P-B1** | B | **Bandwidth detection & tagging** (non-destructive) | Detects narrowband/telephone provenance and records it — used for stratified validation, never for modification | `effective_bandwidth_hz` column | — |
| **P-B2** | B | **Pre-emphasis / rumble high-pass** | Mild, symmetric, standard | Fixed cutoff, documented | — |
| **P-C1** | C | ➕ **Denoise-residual as an extra channel** — feed `[raw, denoised, raw−denoised]` | The residual is *what a denoiser thinks is not speech/music* — a plausible carrier of generation artifacts. **Additive, not destructive** | 3-channel input experiment | — |

### Tier X — rejected preprocessing

| Method | Why not |
|---|---|
| **Denoising / enhancement as a replacement** | Neural denoisers are themselves generative — they can erase the artifact we detect, or manufacture one. The rules also make denoised-real still **REAL**, so we must model that audio, not clean it away. Keep as **P-C1** (extra channel) only |
| **Source separation as preprocessing** | Naive separate-then-detect fails: 38% FPR (vocals) / 94.7% (accompaniment), because artifacts spread across all separated stems ([survey/03](../survey/03-sota-singing-mixed.md)). Use separation for **per-band SNR features** instead |
| **Per-file spectral subtraction / adaptive gating** | Destroys the low-level detail spoof detectors rely on |
| **Anything applied to train but not test** | Guarantees a pipeline shortcut (P2) |

---

## 3. Filtering catalog (F) — training-only

| # | Tier | Filter | What it catches | Action | Gate |
|---|---|---|---|---|---|
| **F-S1** | S | **License / redistribution verdict** | Data we cannot ship ⇒ cannot train on ([01](01-rules-check.md)) | **Drop before download** | **G2** |
| **F-S2** | S | **Corruption** — decode failure, zero length, all-silent, NaN/inf, DC-only | Broken files | Drop; log | — |
| **F-S3** | S | **Duplicates & near-duplicates** — hash + audio fingerprint | Same source in train and val ⇒ inflated CV | Drop from the later split | — |
| **F-S4** | S | 🔴 **Generation-failure detection** (self-generated audio only) | TTS/TTM failure modes: silence, looping/babble, truncation, wrong duration, wrong language, prompt not spoken | Drop or regenerate | **G3** |
| **F-A1** | A | **Pool-membership verification** — Silero VAD for voice, PANNs tagging for music | Pool C "instrumental" files that contain vocals; Pool A files that are actually music | **Reassign, don't drop** — it becomes a mixed-cell sample | **G7** |
| **F-A2** | A | 🔴 **Label-evidence sufficiency** (the real version of "too noisy") | Labelled component present but not perceptible — component SNR below a floor for the whole file | Salvage first (§4); drop only if nothing survives | **G3, G4** |
| **F-A3** | A | **Usable-duration bound** | After salvage, no contiguous valid region ≥ 4 s (the test minimum) | Drop, or concatenate (V-B2) | **G3** |
| **F-B1** | B | **Statistics-T low-information filter** — `T = std+var+rms+pwr`, quantile cut | Low-information chunks ☆ `[BirdCLEF 2024, 1st]` | Segment-level flag, not file drop | **G3** |
| **F-B2** | B | **Per-class / per-speaker / per-artist capping** | Head-class dominance; ★ `[BirdCLEF 2024, 3rd]` capped at 500 per class, keeping most recent | Downsample, keep in pool | — |
| **F-B3** | B | **Embedding-space outlier detection** | Files unlike anything else in their pool — often mislabelled or mis-sourced | Flag for manual listen | **G3** |

### Tier X — rejected filters

| Filter | Why not |
|---|---|
| **"Too noisy" by absolute SNR** | Noise is the target domain (P1). The test set contains telephone audio by design |
| **"Low quality" by bitrate / bandwidth** | Same reason — and bandwidth correlates with the telephone subset we most need |
| **Dropping clipped audio** | Clipping occurs in real phone recordings; inoculate instead |
| **Any filter applied *asymmetrically across labels*** | Instantly creates the shortcut the whole plan is built to avoid ([06](06-augmentation-spec.md#-the-governing-rule)) |

---

## 4. Salvage (V) — how we rescue partially-bad files

Your instinct is right and important: with a pool this constrained, **file-level dropping is
wasteful**. But I'd change the mechanism.

### 🔴 Recommended: validity masks, not cut-and-concatenate

| | Cut & physically concatenate | **Validity mask + region-limited cropping** |
|---|---|---|
| Splice artifacts | Introduced into every salvaged file ⚠️ becomes a label cue if salvage rates differ by class | **None** — we never join anything |
| Reversibility | Destructive; threshold change ⇒ redo | Regenerate a small `.npz`, seconds |
| Shipping to DACON | Must ship modified audio + explain it | Ship pristine originals |
| Variety | Fixed | Every epoch samples a different valid crop |

**Mechanism**: score every 50 ms frame, derive a boolean validity mask per file, and at training
time sample crops **only from contiguous valid regions**. A file that is 60% unusable still
contributes its good 40%, with zero splicing.

```
V-A1  segment scoring     → energy dB, SNR proxy, clipping ratio, spectral flatness,
                            VAD probability, tag confidence   (50 ms hop)
V-A2  mask derivation     → threshold + morphological close/open, min-run length
V-A3  region inventory    → list of contiguous valid spans ≥ 4 s per file
V-A4  crop sampling       → runtime sampler draws only from valid spans
```

| # | Tier | Method | Purpose | Output |
|---|---|---|---|---|
| **V-A1** | A | Per-segment quality scoring | The evidence base for every mask decision | `quality.parquet`, 50 ms resolution |
| **V-A2** | A | Mask derivation with hysteresis + min-run length | Avoids chattering masks that fragment a file into unusable slivers | `validity_mask.npz` |
| **V-A3** | A | Region inventory | Tells us how much of each pool actually survives, per source | `usable_duration.csv` — feeds **G4** |
| **V-B1** | B | Region-limited crop sampler | Runtime consumption of the masks | Sampler code |
| **V-B2** | B | ⚠️ **Fallback concatenation** — only when no single valid span ≥ 4 s | Rescues files that would otherwise be lost | **Must be applied to REAL and FAKE at equal rates**, with sigmoid crossfade (`A-A4`), and tagged `spliced=True` so we can ablate it out |

⚠️ **The salvage trap**: if noisy files cluster in one class (e.g. our self-generated fakes are
clean while public reals are noisy), then salvage rate correlates with label and the mask itself
becomes a shortcut. **Report salvage rate per cell and per pool — it must be comparable.** This is
gate **G4**.

---

## 5. 🔴 Human review gates

Design rule for when I stop and ask you: **a gate exists wherever a decision is (a) irreversible
or expensive to undo, (b) threshold-dependent with no ground truth, (c) affects a large share of
the corpus, or (d) trades our training distribution against the test distribution.**

| Gate | Trigger | What I bring you | Decision needed | Default if you say "you decide" |
|---|---|---|---|---|
| **G1** | Signal-chain spec inferred from the 3 dummy files | Measured evidence per file (headers, LTAS, LUFS, rolloff) and my inferred `signal_chain.yaml`, with confidence per field | Confirm or correct. **Everything downstream depends on this** | Adopt the inference, but widen randomization on low-confidence fields rather than pinning |
| **G2** | Any dataset's license verdict is ⚠️ or contested | Licence text excerpt, the specific clause, my reading, and the cost of excluding it | Legal risk call — **yours, not mine** | Exclude. Cheaper than disqualification |
| **G3** | **Any threshold, before it is applied at scale** | A **review pack** (§6): distribution histogram with candidate cut points, % of corpus affected at each, and 15–20 listenable boundary examples with scores | Approve a cut point or move it | Use the pre-registered quantile, apply to ≤10% of the pool, and re-gate before full application |
| **G4** | A filter or salvage step would **drop >5% of any pool**, or salvage rates differ by >10 points across cells | Before/after distribution comparison, per-pool and per-cell loss table, and what the shortcut audit says | Accept the loss, loosen the threshold, or salvage differently | Loosen until under 5% and flag for revisit |
| **G5** | Any operation that would **modify or delete stored audio** | What changes, why, and why a sidecar can't do it | Explicit approval | **Refuse** — use a sidecar annotation instead |
| **G6** | A **symmetric preprocessing** choice (channel policy, loudness normalization, silence trimming) | The tradeoff, what the dummy files suggest, and an ablation plan | Pick a policy | Choose the least destructive option and ablate later |
| **G7** | **Pool/cell reassignment** driven by automatic detection (e.g. VAD finds vocals in "instrumental") | Count reassigned, rate per source, and 10 examples to listen to | Confirm the detector is right | Reassign but tag `auto_reassigned=True` so it can be excluded |
| **G8** | **Corpus freeze** at the end of Phase 3 | Full corpus card: cell counts, hours, salvage rates, shortcut-audit AUC, generator coverage, license ledger completeness | Approve the freeze | Do not freeze without approval |

### How review packs reach you

For **G1, G3, G4 and G7** the useful artifact is something you can *listen to*. I'd propose
publishing each review pack as an Artifact — an HTML page with the histogram, the candidate cut
points, and the boundary examples as embedded playable audio (16 kHz mono clips as data URIs; ~20
clips fits comfortably). You click through, listen, and tell me where the line goes. Say the word
and I'll build the generator when we reach the first threshold.

---

## 6. 🔴 Threshold calibration protocol

This is the direct answer to *"if we set the noise threshold to 10, that might be too low."*
**No threshold is ever hardcoded from intuition.** Every one follows this protocol:

```
1. MEASURE      compute the score distribution over a random sample of the pool
                (never assume a scale — dB of what, referenced to what?)
2. EXPRESS      state the threshold as a QUANTILE of our own data, not an absolute
                number, so it stays meaningful as the corpus changes
3. PACK         build a review pack: histogram + candidate cut points + % affected
                at each + 15-20 boundary clips (just below / at / just above)
4. GATE         you listen and set the line                              ← G3
5. RECORD       threshold_version, value, quantile, rationale, date, approver
                → provenance ledger
6. APPLY        regenerate sidecar metadata only; originals untouched
7. VERIFY       re-run shortcut audit (E-S2) + salvage-rate parity (G4)
8. REVISIT      re-validate after any corpus change
```

**Why quantiles, not absolutes**: "noise threshold = 10" has no meaning until we know the unit and
the reference. A quantile ("drop the bottom 2% by component SNR") is self-calibrating, survives
corpus changes, and makes the *cost* explicit — you always know what fraction you are giving up.

**Pre-registered starting quantiles** (to be confirmed at G3, not applied blind):

| Threshold | Starting proposal | Rationale |
|---|---|---|
| Segment validity (component SNR) | bottom **2%** of segments | Deliberately conservative — P1 says keep noisy audio |
| Statistics-T low information | bottom **5%** of chunks ☆ `[BirdCLEF 2024, 1st]` used 0.8 quantile — far more aggressive; we start gentler | Their task had abundant data; ours does not |
| Silero VAD voice presence | **0.4** (community-adjusted from the 0.5 default) ★ `[BC25 notebook]` | Verification pass for Pool C |
| Minimum contiguous valid span | **4.0 s** | The test-set minimum duration |
| Max drop per pool before escalation | **5%** | Triggers **G4** |

---

## 7. Audit trail

Every curation decision writes a row linking **file → operation → threshold version → verdict →
reason → approver → date**. Three reasons this is non-negotiable:

1. **Reversibility** — any threshold can be revisited by regenerating sidecars.
2. **DACON's 출처 명시 의무** and the 학습데이터 구성 보고서 (20 pts) — we must be able to state
   exactly how the training corpus was constructed.
3. **Debugging** — when a cell's EER is bad, the first question is always "what did we throw away
   from that cell?"

Extends the ledger schema in [08](08-build-plan.md#provenance-ledger) with:
`curation_verdict · verdict_reason · threshold_version · usable_seconds · salvage_ratio ·
spliced · auto_reassigned · approver · approved_date`.

---

## 8. Failure modes this design is guarding against

| # | Failure | Guard |
|---|---|---|
| 1 | Filtering out the noisy audio that *is* the test distribution | P1; Tier X filters; **G4** |
| 2 | Preprocessing applied to train but not test | P2; single shared code path; **G6** |
| 3 | Denoising erases the artifact we detect | Denoise rejected as preprocessing; kept only as an extra channel (**P-C1**) |
| 4 | Salvage rate correlates with label ⇒ the mask becomes a shortcut | Per-cell salvage parity report; **G4**; shortcut audit (**E-S2**) |
| 5 | Splice artifacts become a "fake" cue | Masks instead of concatenation; fallback splicing applied to both classes and tagged |
| 6 | A bad threshold silently destroys 30% of a pool | Quantile expression + review packs + **G3/G4** |
| 7 | Irreversible curation forces a re-download | Layer 0 immutability; sidecars only; **G5** |
| 8 | Generated-audio failures (looping, silence) train the model on garbage | **F-S4**, gated at **G3** |
| 9 | We cannot explain the corpus to the judges | Audit trail §7 |

---

## 9. Execution order

| Step | Methods | Gate |
|---|---|---|
| 1 | E-S1 dummy forensics → `signal_chain.yaml` | **G1** |
| 2 | F-S1 license verdicts | **G2** |
| 3 | P-S1…P-S4 preprocessing core; G6 policy decisions | **G6** |
| 4 | F-S2, F-S3 corruption + duplicates | — |
| 5 | F-S4 generation-failure detection (as generation starts) | **G3** |
| 6 | V-A1 segment scoring over the whole corpus | — |
| 7 | V-A2/V-A3 mask derivation + region inventory | **G3, G4** |
| 8 | F-A1 pool-membership verification | **G7** |
| 9 | F-A2/F-A3 label-evidence sufficiency + duration bound | **G3, G4** |
| 10 | E-S2 shortcut audit + salvage parity check | — |
| 11 | Tier B filters as evidence warrants | **G3** |
| 12 | Corpus freeze | **G8** |
