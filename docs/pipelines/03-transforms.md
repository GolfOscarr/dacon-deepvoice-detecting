# 03 — The Three Registries

Where a processing idea gets plugged in, and which of three contracts it must satisfy.

🔴 **The registry a transform belongs to is decided by one question**: *does it also run at test
time?* [data/10 P2](../data/10-preprocessing-and-filtering.md#-p2--transformations-must-be-traintest-symmetric-filters-are-train-only)
settles it, and getting it wrong is silent in both directions.

| Registry | Runs at test time | Sees labels | Sees RNG | Lives in |
|---|:--:|:--:|:--:|---|
| **Preprocess** | ✅ **identical code path** | ❌ | ❌ | shared module, vendored into `submit.zip` |
| **Augment** | ❌ | ❌ **structurally** | ✅ | training only |
| **Filter** | ❌ (impossible — all 1,200 files must be predicted) | ✅ | ❌ | offline, sidecar only |

---

## 1. Preprocess — symmetric, and shipped

```python
Preprocess = Callable[[Tensor, int, Tensor | None], Tensor]   # (wav, sample_rate, lengths) -> wav
```

Deterministic, no RNG, no labels, **per file**. Applied in the same order by the training pipeline
and by `submit.zip`, from one module with one call site.

Already implemented and on the inference path:

| Step | Where | Config |
|---|---|---|
| Channel policy | [`models.audio.prepare_waveform`](../../models/AGENTS.md) | `AudioConfig.channels` — `downmix` \| `left` \| `mid_side` |
| Band restriction | [`models.audio.bandpass`](../../models/AGENTS.md) | `AudioConfig.band_hz` |

Still to build, per [data/10 §2](../data/10-preprocessing-and-filtering.md#2-preprocessing-catalog-p--symmetric-train-and-test):
robust decode (**P-S1**), resample to 16 kHz with one fixed resampler (**P-S2**), and whatever
**G6** settles for loudness and silence policy.

### 🔴 The rule-2.4 contract every preprocess step must satisfy

> A file's output must not depend on what else is in the batch.

This is not theoretical here. `bandpass` ran its rFFT over the *padded* batch tensor, so one sample
of padding moved the filtered waveform by up to 1.0 on unit-variance audio — while its docstring
asserted the opposite. `align_time` interpolated over padded frame counts. `frame_max` was
padding-sensitive in the *submitted* probability while the test asserted on `clip_logits`
([`PROGRESS.md`](../../PROGRESS.md)). Three violations, none caught by a green suite.

Consequences for the registry:

- `lengths` is a **required** argument, not an optional one. Any step that transforms along time
  operates on each row's valid prefix.
- ❌ No batch statistics. No per-batch normalization, no percentile over the batch, no shared
  frequency grid derived from the padded width.
- Every registered step is covered by the batch-invariance test in [05](05-invariants.md) — solo
  versus inside a batch of longer, louder files, bitwise identical over the valid prefix.

### Plugging in an EDA finding

The two shapes most likely to come out of EDA land in **different registries**, and the difference
is not cosmetic:

| Finding | Registry | Why |
|---|---|---|
| "Restrict to 0–4 kHz — the telephone slice lives there and D9 measured up to 25% relative EER reduction under codec conditions" | **Preprocess** | It changes what the model is shown at test time too. `AudioConfig.band_hz` already exists; this is a config change, not new code ([09 B7](../architecture/09-open-questions.md)) |
| "Emphasize 2–4 kHz, where the artifact energy concentrates" | **Preprocess** | Same reasoning. A fixed, content-independent emphasis curve is part of the shipped signal path |
| "Jitter gain ±6 dB" | **Augment** | Label-independent variation the test set does not receive ([A-A7](../data/06-augmentation-spec.md)) |
| "Drop files where the labelled component is inaudible" | **Filter** | Train-only, and a distribution-shift decision requiring **G4** |

⚠️ **A content-*dependent* emphasis is still preprocess.** "Amplify wherever energy is low" is
deterministic given one file, so it ships — but it must then be measured under
[E-S2](../data/07-eda-plan.md), because a step that reacts to content can correlate with label
through the content even though it never reads the label.

---

## 2. Augment — label independence enforced by signature

```python
Augment = Callable[[Tensor, Generator], Tensor]    # (wav, rng) -> wav
```

🔴 **The augment function does not receive the labels.** Not "must not read them" — cannot. The
governing rule of [data/06](../data/06-augmentation-spec.md#-the-governing-rule) is
`P(T | L) = P(T)`, and a signature is a stronger guarantee than a code review.

The RNG is drawn from `hash(sample_id, epoch, global_seed)` ([A-S2](../data/06-augmentation-spec.md)),
never from anything label-derived, which is also what makes rendering byte-reproducible.

### Ordering

Per the [pipeline order](../data/06-augmentation-spec.md#pipeline-order-per-training-sample), and
note where the two ends sit:

```
step 3  structural composition       overlap (gain ratio) | sequential (crossfade)
step 4  augment registry             RawBoost, gain, noise, time shift, SpecAugment, RIR, silence
step 5  TEST-CHAIN NORMALIZATION     always last -- A-S1, A-S3, A-S4
```

⚠️ **Step 5 is not in the augment registry.** It is drawn into `SampleSpec.normalize` and applied
by the renderer, because it is the one stage that models *what the organizers did to the test set*.
Its parameters come from **G1** `signal_chain.yaml`, which does not exist yet — so the stage can be
written now and cannot be parameterized until the dummy-file forensics run
([E-S1](../data/07-eda-plan.md)).

🔴 **MixUp is not here either.** [A-A1](../data/06-augmentation-spec.md) and A-A2 change which
components are present and whether they are generated, so they are **component draws in step 2**,
not signal transforms — otherwise they would violate the invariant that labels come from steps 1–2
only.

### Calibration

⚠️ The measured winning recipe is **milder than instinct suggests**: gain ±6 dB, noise SNR
10–30 dB, time shift ±0.5 s — *"the heavy lifting is done by MixUp, not by signal mangling"*
([kaggle/06 §5](../kaggle/06-notebook-code.md)). Per-source `aug_strength`
([A-B7](../data/06-augmentation-spec.md)) scales this: heavier on clean read speech, lighter on
already-degraded telephony.

---

## 3. Filter — offline, sidecar, never audio

```python
Filter = Callable[[ManifestRow, QualityRow], Verdict]   # -> keep | quarantine | drop
```

🔴 **Nothing in this registry rewrites or deletes audio** ([data/10 §1](../data/10-preprocessing-and-filtering.md#1-two-layer-architecture)).
Verdicts are sidecar annotations, so any threshold can be revisited by regenerating metadata, and
the corpus shipped to DACON stays byte-identical to what we obtained. **G5** exists to refuse any
proposal to the contrary.

The filter criterion is **label-evidence sufficiency, not cleanliness**
([data/10 P1](../data/10-preprocessing-and-filtering.md#-p1--filter-for-label-evidence-sufficiency-not-for-cleanliness)).
Our test set contains 전화채널 audio — narrowband, codec-degraded, noisy *by construction*. Noise is
a property of the target domain, not a defect. The legitimate question is never *"is this clean?"*
but *"is the labelled component still discernible enough for its real/fake status to be judgeable?"*

Because we cannot filter the test set, **every filter is a distribution-shift decision**, which is
why **G3** gates any threshold before it is applied at scale and **G4** fires when a step would
drop >5% of a pool.

---

## 4. 🔴 Time invariance — the second structural rule

> **Steps 4–5 are time-invariant. Every time-warping decision lives in the draw (steps 0–3),
> where the spec records it.**

This is the same move as §2's signature rule, applied to a different failure. Labels are kept out
of augments by giving them nowhere to enter; audio is kept on the timeline by making a warp
unrepresentable in the stages that run after the draw.

**Why it has to be structural.** `frame_intervals` are intervals in absolute seconds on the
timeline the sampler drew, computed from `ComponentDraw.target_start_s`
([01 §4](01-sample-contract.md#4-renderedsample)) before any file is opened. Anything that moves
audio *after* that point desynchronizes the frame labels from the waveform — silently, with a green
suite, in exactly the way `align_time` did. ⚠️ The alternative design — let a transform warp time
and report the true placement back — was rejected because it destroys **spec-as-ledger**: the
shortcut audit ([05 I1b](05-invariants.md)) would become a frequency table over placements the
model never saw, and the eval-set freeze would stop being exact.

### The four ways to move audio

| # | Class | Examples | Recognisable by |
|---|---|---|---|
| **1** | **Rigid shift** | A-A8 time shift, an encoder's delay, leading silence | one lag, the same everywhere |
| **2** | **Rate change** | A-C2 time stretch, an uncompensated resample | head and tail lag differ |
| **3** | **Non-monotonic edit** | A-A11 internal trimming, splicing, packet-loss concealment | head and tail lag differ; length may not |
| **4** | 🔴 **Group delay** | **A-A10 RIR convolution**, any non-linear-phase filter | one lag — *and nobody looks for it* |

⚠️ **Class 4 is the one that is never on anybody's list**, and it is why this is measured rather
than reviewed. `models.audio.bandpass` is safe only because it is zero-phase (a brick wall in the
rFFT domain), and `scipy.signal.resample_poly` only because it is linear phase and compensates its
own filter delay. Neither of those is a decision anyone recorded — they are **load-bearing
accidents**, of the same kind that has produced four shipped defects here. A-A10 would introduce a
shift equal to its direct-path offset and nothing downstream would notice.

### How it is enforced

Registration **probes** the step: it is run on fixed-seed broadband noise, and the output is
correlated back against the input in two windows.

- A **length change** is refused — length is the timeline.
- **Head and tail lag disagreeing** is refused: the time base was warped, not moved, and no single
  number can carry that into `frame_intervals`. 🔴 Two windows, not one — a single lag catches
  classes 1 and 4 but is blind to 2 and 3, since a 1% stretch and an internal excision both leave
  the head exactly where it was.
- A **lag that does not match the step's declared `group_delay`** is refused. The declaration is
  *checked*, never believed: declare 0 and shift by 137 and it fails; declare 137 and shift by 0
  and it fails too.
- An **augment may not declare a delay at all**. A delay it wanted is a draw.

⚠️ The probe uses noise, not a chirp, and that detail is load-bearing: a chirp's head and tail hold
different frequencies, so a filter with frequency-dependent phase (`pre_emphasis`, a 2-tap
differencer) reads as head and tail disagreeing and a step that moves nothing gets refused.

⚠️ The probe runs with **default** parameters, so it cannot see a warp that only a drawn parameter
turns on. `augment_chain` therefore re-checks the length on the real audio, on every sample.

### Where the warps went instead

| Method | Now lives in | As |
|---|---|---|
| **A-A8** time shift | the draw | `SamplerConfig.silence_lead_s` → `ComponentDraw.target_start_s` |
| **A-A11** silence edits | the draw | `silence_lead_s` / `silence_tail_s`; components occupy `duration_s − lead − tail` |
| **A-S3** codec delay | step 5, cancelled | the encoder's gapless header, with the decoded length asserted ([data/06 A-S3](../data/06-augmentation-spec.md)) |
| **A-A4** crossfade | step 3, without moving anything | complementary sigmoid tapers at the joint, because an overlapping crossfade would move a component off the drawn timeline |
| **A-C2** time stretch | not implemented | it needs a rate field on `ComponentDraw`; tier C at p≈0.1, not worth the field yet |
| **P-S2** resample | `render.load_audio`, once per file | it changes the sample count, and a registry whose steps may change `lengths` cannot be composed |

🔴 **The codec delay is the one warp that happens in a stage the contract covers**, and it escapes
because step 5 is *conceptually* identity in time: the test chain re-encodes the audio, it does not
move it. So the delay is an implementation artifact to **cancel and assert**, not a warp to declare
— which is why `_codec_roundtrip` checks the decoded frame count instead of trusting anything.
