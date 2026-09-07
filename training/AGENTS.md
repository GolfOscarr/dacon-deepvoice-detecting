# Using the training pipeline

Everything that turns the component pools into a batch, and the audit that says
the batch is not a trap.

**The one rule: sampling is separate from rendering.** `sample_spec()` is pure
and touches no audio; `render()` does all the I/O. Almost every guarantee below
follows from that split — the shortcut audit is a frequency table over specs,
the eval set is frozen as *specs* rather than a seed, and reproducibility is
`render(spec) == render(spec)`.

Why each choice was made: [`docs/pipelines/`](../docs/pipelines/README.md) and
[`docs/training/`](../docs/training/README.md). This file is how to *use* it.

| Module | Use it for |
|---|---|
| `training.spec` | `SampleSpec`, `ComponentDraw`, the nine cells, the RNG key |
| `training.manifest` | Loading and validating the manifest. Rejects the two row kinds' confusions |
| `training.sampler` | Drawing specs. C1 / C3 / DOSS live here |
| `training.audit` | **I1–I9** and **I21** over a drawn stream. No corpus, no model, no GPU |
| `training.folds` | Building `folds.parquet`, and **VG1 A1–A7 / A10** over it |
| `training.synthetic` | A manifest with the real corpus's pathologies, and the audio it names |
| `training.registries` | The three transform contracts. **I14** lives here |
| `training.render` | `SampleSpec` → audio. All the I/O, and `render(spec) == render(spec)` |
| `training.collate` | `list[RenderedSample]` → batch, and duration bucketing. **I15** lives here |
| `training.dataset` | The torch `Dataset`: specs → render → collate, plus the frozen eval set |
| `training.loop` | S1→S3, resume, EMA, soup, validation, the **VG gates** and the leak tripwires |

---

## 🔴 A component row has no cell

The single most likely misreading. `cell` is a property of a **composition**,
and compositions are drawn at runtime. A component file from pool A is voice —
it is not "cell 1" until the sampler has decided not to give it any music.

| Row kind | Has | Used how |
|---|---|---|
| `component` | `pool` ∈ A–E, `cell` null | drawn and composed |
| `whole_file` | `cell` ∈ 1–9, `pool` null | used as-is |

Validation rejects both confusions, so a manifest that muddles them fails loudly
rather than producing a silently wrong `cell` distribution.

---

## The whole path, once

```python
from training.audit import run_audit
from training.sampler import Sampler, SamplerConfig
from training.synthetic import synthetic_manifest

manifest = synthetic_manifest(n_per_pool=100, n_whole_file=100, seed=0)
sampler = Sampler(manifest, SamplerConfig(), slice_="train")

spec = sampler.sample_spec(sample_id=0, epoch=0, seed=0)
assert spec.file_fake in (0, 1)          # derived from spec.cell, never stored

report = run_audit(sampler, n=4000, manifest=manifest)
assert report.ok, str(report)
```

⚠️ Two lines are easy to skip and both are wrong to skip. `manifest=` is what
lets the audit check split safety and the both-sides rule at all — without it
those two invariants report `skipped`. And `report.ok` must be *asserted*:
`run_audit` returns a report, it does not raise.

---

## Labels come from the cell, and nowhere else

`SampleSpec` has no label fields. They are properties derived from `cell`:

```python
spec.voice_present, spec.music_present, spec.voice_fake, spec.music_fake
spec.file_fake        # computed by metrics.dacon.file_fake_label
spec.stratum          # "voice-only" | "music-only" | "mixed" | "neither"
assert spec.file_fake == spec.labels["file_fake"]
```

🔴 That is deliberate. If they were fields a transform could write to them, and
"labels come from steps 1–2 only" would be a convention rather than a fact.

⚠️ `file_fake` is OR over **present** components, which is what makes cells 1–4
well defined. Pass `0`, not `None`, for an absent component's fake label if you
ever call `file_fake_label` yourself — `None` survives only incidentally.

---

## The one knob: `f8`

```python
from training.sampler import SamplerConfig

SamplerConfig(f8=0.0)   # label-conditional  <- primary
SamplerConfig(f8=1.0)   # strict
```

`f8` is the composed fraction of cell 8, and it fixes every other one:

| `f8` | `f5` | genuine whole-file audio | policy |
|---|---|---|---|
| **0.00** | **0.725** | **13.8%** | conditional — AI songs and natural songs usable |
| 1.00 | 1.000 | 0% | strict — every mixed file composed |

`f6 = f7 = 1` always: those cells hold one real and one fake component, so they
cannot be scraped. `SampleSpec` refuses to build a whole-file cell 6 or 7.

⚠️ **Falling back to strict is a value change, not a second code path.**

---

## Running the audit before you trust anything

```python
from training.audit import run_audit

report = run_audit(sampler, n=20_000, manifest=manifest)
print(report)          # one line per invariant, with the measured quantity
report.raise_for_status()
```

🔴 The audit measures a **drawn stream**, never the config it came from. The
constraint arithmetic can be right while the sampler implementing it is wrong,
and only the stream catches that.

⚠️ Two failures worth recognising, because both were real:

- `I8_C1_positive_rates` out at **0.820** on `v_pres`/`m_pres` — the cell mix
  over-weights cells 6/7, which all have *both* components present.
- `I2b_mixedness_balance` — the mix passes C1 but "is a mixed file" predicts
  FAKE. C1 alone does not catch it.

---

## Building `folds.parquet`

**Sealed PROBE, rotating TRAIN/VAL.** A family is either sealed into PROBE —
never TRAIN, never VAL, in any fold — or it rotates: VAL in exactly one fold,
TRAIN in all the others. That is docs/validation/01 §3's music option 1 read
literally ("5 TRAIN / 2 VAL rotating / 1 sealed PROBE"), and it is why the family
floors stay at ≥20 voice / ≥8 music.

```python
from training.folds import FoldConfig, apply_folds, build_folds, check_split_integrity
from training.synthetic import synthetic_manifest

corpus = synthetic_manifest(n_per_pool=240, n_whole_file=200,
                            n_families=24, n_sources=8)
plan = build_folds(corpus, FoldConfig(n_folds=5))

print(plan)                       # slice counts, VAL rows per fold, then caveats
report = check_split_integrity(plan.frame, run_scheme_version="synthetic-v1")
report.raise_for_status()         # VG1 A1-A7 and A10

fold0 = apply_folds(corpus, plan.frame, fold=0)      # the manifest for one fold
assert set(fold0["slice"]) <= {"train", "val", "shadow", "probe"}
```

🔴 `folds.parquet` holds `slice ∈ {train_val, shadow, probe}` and a `fold`, not a
TRAIN/VAL label. **`apply_folds(..., fold=k)` is what resolves it** — `train_val`
becomes `val` for the families assigned to fold `k` and `train` for every other
one. Calling `Sampler` on the raw fold table would train on the validation set.

⚠️ `plan.caveats` is the part people skip. It is where the music-head variance
warning lives, with the *realized* families-per-validation-fold rather than a
mean — at 8 music families a 5-fold leaves one sealed and the rest rotating, so
some fold validates on exactly one family
([`docs/validation/01 §3`](../docs/validation/01-split-scheme.md#-the-music-head-cannot-support-the-planned-split)).
`plan.write()` drops them in `folds.caveats.txt` beside the parquet so they
travel with the table.

### Four things it refuses to do

`build_folds` raises `FoldInfeasible` rather than quietly returning a worse
split. Each of these is a real corpus property, not a bug in the caller:

| Refusal | Why |
|---|---|
| fewer rotating families than `n_folds` | a fold that validates on no family of its own validates nothing |
| nothing sealed into PROBE | the sealed slice is the only check on "VAL became a training set". Pass `allow_no_probe=True` to accept the blind spot deliberately |
| a fold side the sampler cannot draw from | no real voice components ⇒ cells 1/5/6 cannot be composed, and `Sampler` only finds out at draw time |
| a `pair_id` that fuses the whole corpus | see below |

🔴 **Under rotation the fold count is bounded by the number of real source
corpora per role, not only by the family count.** Every one of the `k` VAL sides
needs its own real voice, real music and noise source, and PROBE needs one more —
so `k = 5` wants ≥6 of each. That is why the snippet above passes `n_sources=8`.

```python
import pytest

from training.folds import FoldConfig, FoldInfeasible, build_folds
from training.synthetic import synthetic_manifest

thin = synthetic_manifest(n_per_pool=240, n_whole_file=200, n_families=24)
with pytest.raises(FoldInfeasible, match="real source corpora"):
    build_folds(thin, FoldConfig(n_folds=5))          # only 3 corpora per role

plan = build_folds(thin, FoldConfig(n_folds=2))       # 2 is feasible
```

### 🔴 The five grouping keys are one constraint, not five

`artifact_family`, `source_name`, `speaker_ref_id`, `pair_id` and `dup_group` are
all equivalence constraints, so `grouping_atoms` takes their **transitive
closure**. A `pair_id` binding one LibriTTS utterance to its HiFi-GAN twin binds
the *whole LibriTTS corpus* to the *whole HiFi-GAN family*.

That is not a quirk of the implementation, it is the real constraint: a corpus
whose twins were generated before the family partition was frozen has no
family-disjoint split at all, and `build_folds` says so instead of dropping a
key. It is also why `training.synthetic` twins only the (corpus, vocoder)
combinations in `_TWIN_COMBINATIONS` — the earlier random pairing fused 1,088 of
1,200 rows into one inseparable group.

VG1 A1–A5 are all the same statement over different keys: **the value resolves to
exactly one `(slice, fold)` cell.** Two cells means two roles in the same fold —
TRAIN and VAL at once, or sealed and rotating at once.

### Where A8 and A9 are

Not here. They are statements about **compositions**, and a component row has no
cell, so they run at eval-set materialization instead
([`docs/validation/04 §VG1`](../docs/validation/04-audit-gates.md#vg1--split-integrity)):

```python
from training.audit import run_audit
from training.folds import FoldConfig, apply_folds, build_folds
from training.sampler import Sampler
from training.synthetic import synthetic_manifest

corpus = synthetic_manifest(n_per_pool=240, n_whole_file=200,
                            n_families=24, n_sources=8)
fold0 = apply_folds(corpus, build_folds(corpus, FoldConfig()).frame, fold=0)

report = run_audit(Sampler(fold0, slice_="val"), n=1500, manifest=corpus,
                   eval_floors=True)
print(report.results["I7_eval_size_floors"])     # VG1 A8/A9, measured
```

⚠️ `eval_floors=True` is not the default and must not become one. A **training**
stream is not required to meet the 1,200-per-class floor, so without the flag the
check reports `SKIPPED`, never `PASS`.

---

## The three registries, and why they have different signatures

```python
import numpy as np
import torch

from training.registries import (AUGMENT, FILTER, PREPROCESS, RegistryError,
                                 augment_chain, preprocess_chain)

# Augment: (wav, rng) -> wav. Training only, and there is no third argument.
noisy = augment_chain((("gaussian_noise", {"snr_db": 20.0}),
                       ("gain_jitter", {"db": -3.0})))(
    torch.randn(2, 16_000), np.random.default_rng(0))

# Preprocess: (wav, sample_rate, lengths) -> wav. Shipped, and `lengths` is
# required -- each row is transformed over its own valid prefix.
batch = torch.randn(4, 32_000)
lengths = torch.tensor([16_000, 32_000, 24_000, 5_000])
clean = preprocess_chain((("dc_offset", {}), ("pre_emphasis", {})))(
    batch, 16_000, lengths)
assert clean.shape == batch.shape

# Filter: (manifest_row, quality_row) -> Verdict. Offline, sidecar, no audio.
verdict = FILTER.build("corruption")({"file_id": "A0001"}, {"decode_ok": False})
assert verdict.action == "drop" and not verdict.keeps
```

🔴 **An augment cannot be given a label.** Not by convention — the registry
refuses to accept the function at all:

```python
from training.registries import Registry, LABEL_PARAM_NAMES, RegistryError

reg = Registry("augment", ("wav", "rng"), LABEL_PARAM_NAMES)
try:
    @reg.register("peeks")
    def peeks(wav, rng, *, file_fake=0):
        return wav * (2.0 if file_fake else 1.0)
except RegistryError as exc:
    assert "forbidden" in str(exc)
else:                                  # pragma: no cover
    raise AssertionError("the registry accepted a label-reading augment")
```

`P(T | L) = P(T)` ([docs/data/06](../docs/data/06-augmentation-spec.md#-the-governing-rule))
is then a property of the type, and the same mechanism gives **G5** for free: a
filter is handed two metadata rows, so "nothing here rewrites audio" is not a
promise anybody has to keep.

⚠️ Which registry a step belongs to is decided by **one** question: *does it also
run at test time?* Getting it wrong is silent in both directions
([docs/data/10 P2](../docs/data/10-preprocessing-and-filtering.md)).

### What is deliberately *not* registered

| Missing | Why |
|---|---|
| **A-A8 time shift**, **A-A11 silence edits** | They move audio along the timeline while `frame_intervals` stay put. Now drawn instead: `SamplerConfig.silence_lead_s` / `silence_tail_s` → `ComponentDraw.target_start_s`. See the contract below |
| **MixUp (A-A1/A-A2)** | Changes which components are present and whether they are generated, so it is a **step-2 component draw**, not a signal transform |
| **Resample (P-S2)** | Changes the sample count, and a registry whose steps may change `lengths` cannot be composed. It runs once per file in `render.load_audio`, with the resampler injected |
| **Test-chain normalization (A-S1/S3/S4)** | Not variety but a model of *what the organizers did to the test set* — drawn into `SampleSpec.normalize` and applied by the renderer, always last |
| **P-A1 loudness, P-A2 silence trimming** | Gate **G6**. `label_evidence` likewise refuses to run without `threshold_version` (**G3**): a default threshold is the hardcoded intuition [docs/data/10 §6](../docs/data/10-preprocessing-and-filtering.md) exists to forbid, and it would be invisible in a green suite |

### 🔴 The second structural rule: steps 4–5 do not move audio

> **Steps 4–5 are time-invariant. Every time-warping decision lives in the draw
> (steps 0–3), where the spec records it.**

`frame_intervals` are intervals *on the drawn timeline*. Anything that moves audio after the draw
desynchronizes the frame labels from the waveform — silently, which is how `align_time` shipped.
So registration **measures** it rather than asking you to promise it:

```python
from training.registries import PREPROCESS, RegistryError

try:                                   # a step that shifts by 37 samples
    @PREPROCESS.register("late")
    def late(wav, sample_rate, lengths):
        import torch
        return torch.roll(wav, 37, dims=-1)
except RegistryError as exc:
    assert "declares group_delay=0" in str(exc)
else:                                  # pragma: no cover
    raise AssertionError("the registry accepted an undeclared time shift")

assert PREPROCESS.group_delay_of("dc_offset") == 0
```

The step is run on fixed-seed broadband noise and correlated back against the input **in two
windows**: a length change, a head/tail disagreement (a rate change or an internal edit), or a lag
that does not match the declared `group_delay` all refuse registration. An augment may not declare
a delay at all — a delay it wanted is a draw.

⚠️ Declare the *measured* delay, never a guessed one: `group_delay=137` on a step that shifts by 0
fails too. And 🔴 **class 4 is the one nobody looks for** — `models.audio.bandpass` is safe only
because it is zero-phase and `resample_poly` only because it is linear phase; A-A10 RIR convolution
would shift by its direct-path offset. Those were load-bearing accidents until this measured them.

The four classes and where each warp went instead:
[`docs/pipelines/03 §4`](../docs/pipelines/03-transforms.md).

---

## Rendering

```python
import tempfile
from pathlib import Path

import torch

from training.render import ManifestIndex, RenderConfig, render
from training.sampler import Sampler, SamplerConfig
from training.synthetic import synthetic_manifest, write_synthetic_corpus

root = Path(tempfile.mkdtemp())
corpus = synthetic_manifest(n_per_pool=2, n_whole_file=2, seed=0,
                            duration_range=(6.0, 8.0))
write_synthetic_corpus(corpus, root, seed=0)       # the audio the manifest names

index = ManifestIndex.from_frame(corpus)           # build once, render many
cfg = RenderConfig(root=root)
spec = Sampler(corpus, SamplerConfig(duration_range=(4.0, 6.0))).sample_spec(0)

sample = render(spec, index, cfg)
assert sample.wav.dim() == 2 and sample.sample_rate == 16_000   # (C, S), 16 kHz
assert torch.equal(render(spec, index, cfg).wav, sample.wav)    # I10, bitwise
```

⚠️ **`sample.wav` is `(C, S)` as decoded, not mono.** The channel policy is
`AudioConfig.channels`, applied by `models.audio.prepare_waveform` at the *same
call site* in training and inference. Downmixing in the dataset would fork that
policy into two places, disable the A-B3 channel augmentations and make the
`mid_side` leak test impossible to run.

🔴 **`frame_intervals` are absolute seconds, never a frame grid.** Frame rate
belongs to the frontend; the collator does not know it and must not guess.

```text
sample.frame_intervals   {"voice": ((0.0, 4.7, 0),), "music": (...), "file": (...)}
sample.targets           the five keys of losses.TARGET_FOR_COLUMN
```

Whole-file rows carry empty tuples — there is no composition to describe.
⚠️ The default loss does not consume frame targets (`clip_weight` is committed
at 1.0), but they are produced anyway: **T1** cannot run without them and
retrofitting means re-rendering the corpus.

### The stage order, and the one that must stay last

```
decode      P-S1 robust decode (extension never trusted) + P-S2 resample to 16 kHz
compose     step 3 -- overlap (gain ratio) | sequential (sigmoid taper at the joint)
augment     step 4 -- AUGMENT registry, called as fn(wav, rng)
normalize   step 5 -- TEST CHAIN, always last
```

⚠️ **The preprocess registry is not run by `render`**, deliberately. A preprocess
step is train/test *symmetric*, so its call site is the model boundary — next to
`prepare_waveform` and `bandpass`, in the training loop and in `submit.zip`
alike. A rendered sample is the analogue of a *raw test file*.

🔴 **A codec round-trip must not move the audio.** LAME's 1105-sample (69 ms)
delay is cancelled by the encoder's gapless header, which ffmpeg can only write
when it can seek back over its output — i.e. to a file, never to a pipe. Encoding
to a pipe silently shifts every MP3 sample by 69 ms while `frame_intervals` stay
put, which is the `align_time` defect somewhere the existing tests do not look.
`render` checks the decoded length rather than trusting it.

---

## Determinism

```python
from training.spec import spec_rng
rng = spec_rng(sample_id, epoch, seed)          # blake2b, not hash()
```

🔴 Python's `hash()` is salted per process, so a `hash()`-keyed stream is
reproducible *within* a run and different across runs — the exact opposite of
what A-S2 asks for. An epoch is a fixed count of drawn specs; without that,
`sample_id` is undefined and reproducibility is nominal.

---

## Collating a batch

```python
import tempfile
from pathlib import Path

import torch

from models.audio import prepare_waveform
from models.config import AudioConfig
from training.collate import BATCH_KEYS, collate
from training.render import ManifestIndex, RenderConfig, render
from training.sampler import Sampler, SamplerConfig
from training.synthetic import synthetic_manifest, write_synthetic_corpus

root = Path(tempfile.mkdtemp())
corpus = synthetic_manifest(n_per_pool=2, n_whole_file=2, seed=0,
                            duration_range=(6.0, 8.0))
write_synthetic_corpus(corpus, root, seed=0)
index, cfg = ManifestIndex.from_frame(corpus), RenderConfig(root=root)
sampler = Sampler(corpus, SamplerConfig(duration_range=(4.0, 6.0)))

batch = collate([render(spec, index, cfg) for spec in sampler.epoch_specs(4)])
assert set(batch) == set(BATCH_KEYS)
assert batch["wav"].dim() == 3                       # (B, C_max, S_max), not mono
wav = prepare_waveform(batch["wav"], AudioConfig())  # -> (B, S), the model's input
```

⚠️ **`batch["wav"]` is `(B, C_max, S_max)`.** The channel policy is applied by
`prepare_waveform`, at the same call site as inference — the collator does not
downmix, does not bandpass, does not run the preprocess chain and does not cast
to the training precision.

🔴 **Rows with fewer channels than the batch are promoted by *repeating their own
channels*, never by zero-filling.** That is the rule-2.4 surface of the layout:
`C_max` belongs to the other rows, so the promotion must be invisible at the
model boundary — and under `downmix` a zero-filled mono row comes back at **half
amplitude**, i.e. its score would depend on what shared its batch. Cyclic repeat
is bitwise the identity for every policy in `models.config.CHANNEL_POLICIES`, and
`tests/test_collate.py` enumerates that tuple rather than restating it, so a new
policy that breaks the property fails the suite instead of shipping.

⚠️ `collate(..., pad_value=...)` exists for the tests, not for training. §2 of
[`docs/pipelines/04`](../docs/pipelines/04-collation.md) says the padding value
must not matter, and the way to know that is to collate twice and compare the
**submitted probability** — which
`tests/test_collate.py::test_the_pad_value_cannot_reach_the_submitted_probability`
does, at `atol=0.0`.

🔴 That was not true until M4 measured it. `frontends.frames_for` rounds *up*, so
a row whose length is not a multiple of the frontend hop has a last valid frame
that is **part padding**, and that frame is masked *in* — so the pad content was
reaching the score through it, by 1.35e-3 on rendered audio. It went unseen
because every padding test in the suite used `lengths = SR * 4`, an exact
multiple of the 320-sample hop; the pipeline draws `U(4, 60)` s and produces
arbitrary lengths, so in production the partial frame is the normal case.
`Frontend.forward` now zeroes past each row's `lengths` before encoding, which is
a **no-op on the shipped path** (both sides pad with zeros) and makes the
guarantee structural rather than incidental. `tests/test_model.py::ragged`
carries the fixture convention forward.

### Duration bucketing

```python
from training.collate import bucket_batches, padding_fraction, spec_durations
from training.sampler import Sampler, SamplerConfig
from training.synthetic import synthetic_manifest

specs = list(Sampler(synthetic_manifest(n_per_pool=60, n_whole_file=60, seed=0),
                     SamplerConfig()).epoch_specs(512))
d = spec_durations(specs)                       # from the SPEC -- no audio decoded

loose = bucket_batches(d, 32, n_buckets=1, seed=0)
tight = bucket_batches(d, 32, n_buckets=4, seed=0)
assert padding_fraction(d, tight) < padding_fraction(d, loose)
```

✅ Bucketing is free to choose: `LossConfig.ranking_weight` is committed at 0 and
stage S4 is dropped, so **no loss term is sensitive to batch composition**.

⚠️ Two caveats survive. Bucketing reshapes the per-batch cell mix (duration and
cell are not independent — sequential compositions run long), so a bucketed plan
must still meet **C2**: audit it with `audit_specs(specs, batch_size=...)` and
read `I9_C2_present_count_floor`. And the duration-vs-score check on the REAL
class must be measured on **unbucketed** batches — which is why `eval_batches`
refuses to bucket and `training_batches` refuses a frozen list.

---

## The dataset, and the two modes

```python
import tempfile
from pathlib import Path

import pytest

from training.dataset import (SpecDataset, eval_batches, frozen_eval_specs,
                              training_batches)
from training.render import ManifestIndex, RenderConfig
from training.sampler import Sampler, SamplerConfig
from training.synthetic import synthetic_manifest, write_synthetic_corpus

root = Path(tempfile.mkdtemp())
corpus = synthetic_manifest(n_per_pool=2, n_whole_file=2, seed=0,
                            duration_range=(6.0, 8.0))
write_synthetic_corpus(corpus, root, seed=0)
index, cfg = ManifestIndex.from_frame(corpus), RenderConfig(root=root)
sampler = Sampler(corpus, SamplerConfig(duration_range=(4.0, 6.0)))

train = SpecDataset.from_sampler(sampler, 8, index, cfg)   # redrawn per epoch
train.set_epoch(1)
plan = training_batches(train, 4, n_buckets=2, seed=0)

held = SpecDataset.frozen(frozen_eval_specs(sampler, 6, seed=0), index, cfg,
                          slice_="train")
with pytest.raises(RuntimeError):        # the eval set is a value, not a seed
    held.set_epoch(1)
assert [i for b in eval_batches(held, 4) for i in b] == list(range(6))
```

🔴 **The evaluation set is frozen as *specs*, not as a seed.** A seed reproduces
only against the same sampler, the same manifest and the same code, and all
three change during a competition. `set_epoch` on a frozen dataset raises — the
alternative is a validation curve whose rows quietly change underneath it, which
no downstream assertion can see.

### 🔴 A composed sample has no fold, so the fold is resolved first

`fold_manifest(manifest, folds, fold=k)` — a thin, required-argument wrapper on
`apply_folds` — resolves TRAIN/VAL **on the manifest**, before a single spec is
drawn. The sampler then has nothing out-of-fold to draw from, so containment is
structural rather than checked afterwards.

That is not a stylistic preference. A composed sample draws two or three
components and nothing binds them to one family, so "the fold of a composed
sample" is undefined; any rule that picked one (the first component's fold, the
voice component's) would be an accident of draw order dressed up as a policy.

```python
from training.dataset import SpecDataset, fold_manifest
from training.folds import FoldConfig, build_folds
from training.render import ManifestIndex
from training.sampler import Sampler, SamplerConfig
from training.synthetic import synthetic_manifest

corpus = synthetic_manifest(n_per_pool=240, n_whole_file=200,
                            n_families=24, n_sources=8)
folds = build_folds(corpus, FoldConfig(n_folds=5)).frame
train = fold_manifest(corpus, folds, fold=2)          # `fold=` is required

sampler = Sampler(train, SamplerConfig(), slice_="train")
ds = SpecDataset.from_sampler(sampler, 500, ManifestIndex.from_frame(train))
report = ds.audit(train)                              # forwards slice_ and fold
assert report.results["I5_split_safety"][0], report.results["I5_split_safety"][1]
```

⚠️ `ds.audit(train)` is the whole containment story: **I5** already re-derives the
allowed `file_id` set from the manifest, so the dataset wires the arguments
rather than rebuilding the check ([`docs/pipelines/05 §5`](../docs/pipelines/05-invariants.md)).
Called without a manifest, I5 reports `SKIPPED` — never a pass.

---

## The training loop

Everything above the dataset: when a sample is shown to the model, what the model
is allowed to learn from it at that point, and **whether the number that comes
out is allowed to be quoted**.

```python
import dataclasses, tempfile
from pathlib import Path

from models.config import load_model_config, load_train_config
from models.model import DeepVoiceNet
from training.dataset import SpecDataset
from training.loop import LoopConfig, train_stage
from training.render import ManifestIndex, RenderConfig
from training.sampler import Sampler, SamplerConfig
from training.synthetic import synthetic_manifest, write_synthetic_corpus

root = Path(tempfile.mkdtemp())
corpus = synthetic_manifest(n_per_pool=2, n_whole_file=2, seed=0,
                            duration_range=(6.0, 8.0))
write_synthetic_corpus(corpus, root, seed=0)
index, rcfg = ManifestIndex.from_frame(corpus), RenderConfig(root=root)

model = DeepVoiceNet(load_model_config("configs/a_stub.yaml"))
train = SpecDataset.from_sampler(
    Sampler(corpus, SamplerConfig(duration_range=(4.0, 5.0))), 2, index, rcfg)
train_cfg = dataclasses.replace(load_train_config("configs/train_joint.yaml"),
                                epochs=1, batch_size=2, stage="joint")

result = train_stage(model, train, train_cfg=train_cfg,
                     loop_cfg=LoopConfig(out_dir=root / "run", n_buckets=1))
assert result.steps >= 1 and result.checkpoints          # resumable from any of them
```

⚠️ `train_stage` takes a **training** `SpecDataset` and refuses a frozen one. It
calls `set_epoch` once per pass, and a frozen eval set refuses `set_epoch` on
purpose — the two refusals are the same rule seen from both ends.

---

### 🔴 Resume restores the *draw*, not only the weights

```python
from training.loop import SamplerState

state = SamplerState(pass_index=3, epoch_seed=0, n_specs=4096,
                     batch_seed=3, batch_index=57)
```

The corpus is **drawn**. A checkpoint that restores model, optimizer and RNG but
not *where in the draw it was* resumes onto a different sample stream, reports it
under the same `exp_id`, and nothing shows it — not an exception, not the loss
curve, not a green suite.

`Sampler.epoch_specs` is keyed on epoch-local `i`, `epoch` and `seed`, and
`bucket_batches` on `(durations, seed)`, so there is **no hidden generator to
serialise** and the whole sampler state is those five numbers. That is the payoff
of the stateless sampler, and also why it is so easy to forget.

| Restored | Dropping it costs |
|---|---|
| `sampler` | a **different corpus**, silently |
| `rng` | dropout draws (`dropout_in=0.25`, `dropout_out=0.5`) diverge on step 1 |
| `optimizer` | AdamW's moments restart |
| `scaler` | the fp16 loss scale replays its warm-up |
| `ema` | the averaged weights restart from the resume point |

`tests/test_loop.py::test_a_resumed_run_reproduces_an_uninterrupted_run_bitwise`
asserts the resumed weights are **bitwise** identical, and the four tests after it
drop one piece each and prove the assertion goes red.

⚠️ Resuming against a dataset that draws a different `n`, or at a different seed,
**raises**. So does resuming into a different stage: the optimizer was built over
a different parameter set.

---

### The three stages, and the one that is gone

```python
from models.config import load_model_config
from training.loop import STAGES, stage_plan

cfg = load_model_config("configs/a_stub.yaml")
assert STAGES == ("independent", "joint", "codec_aware")

print(stage_plan("independent", cfg))     # 5 groups of 1 branch, frontends frozen
print(stage_plan("codec_aware", cfg))     # 1 group of 5, 4-way codec

try:
    stage_plan("rank_polish", cfg)        # S4
except NotImplementedError as exc:
    assert "DROPPED" in str(exc)
else:                                     # pragma: no cover
    raise AssertionError("S4 ran")
```

| Stage | Branches | Frontends | Codec | Grade |
|---|---|---|---|---|
| **S1** `independent` | one at a time | **frozen** | 1-way | ☆ both component papers do it |
| **S2** `joint` | all together | per config | 1-way | ⚠️ EER delta 0.5 pts — *below* our resolution |
| **S3** `codec_aware` | all together | per config | **4-way** | ★ **the best-evidenced stage in the recipe** |
| ~~S4 `rank_polish`~~ | — | — | — | ❌ refuted; `stage_plan` raises |

🔴 **S2 is kept by argument, S3 by measurement.** The famous "largest single gain"
(ACC 69.40 → 85.12) is a *thresholded* result; the EER deltas from the same
ablation are 3.59 → 3.12 and 8.72 → 7.86, below our ≈1 pt local resolution. S3's
evidence is ★ ArtifactNet P2→P3: hard-negative FPR **98.7% → 8.0%**, cross-codec
drift **−83%** ([`docs/training/04`](../docs/training/04-schedule.md)). Budget
accordingly — if time runs out, S3 is the rung to keep.

🔴 S1 **overrides** `FrontendConfig.freeze` rather than reading it. Reading the
field would make S1 and S2 identical on both shipped stubs (they already say
`freeze: true`), so an S1-vs-S2 comparison would measure nothing.

⚠️ **The override is announced, not silent.** If a config says `freeze: false`,
S1 still freezes — and `StagePlan.caveats` names the frontends whose field is not
being honoured, `print(plan)` shows it, `StageResult.caveats` carries it, and
`aggregate_folds(..., caveats=result.caveats)` puts it in the ledger row.
Reported rather than raised, because the field *is* honoured in S2 and S3: this
is a divergence to announce for one stage, not a config the schedule cannot run.
A silent divergence between a config field and actual behaviour is the defect;
the override itself is the schedule.

⚠️ "Each branch alone" is enforced by the **parameter set**, not by zeroing loss
terms: AdamW's weight decay and momentum move a branch nobody is training this
pass, so a loss-only restriction would make "alone" a claim rather than a fact.

---

### 🔴 S3's codec expansion is a cross product, never a draw

```python
from training.loop import CODEC_VARIANTS, codec_variant_specs
from training.sampler import Sampler, SamplerConfig
from training.synthetic import synthetic_manifest

corpus = synthetic_manifest(n_per_pool=20, n_whole_file=20, seed=0)
specs = list(Sampler(corpus, SamplerConfig()).epoch_specs(50))
expanded = codec_variant_specs(specs, CODEC_VARIANTS)

assert len(expanded) == 4 * len(specs)
assert sorted(s.cell for s in expanded) == sorted(c for s in specs
                                                  for c in [s.cell] * 4)
```

Every variant is paired with **every** spec, so `P(normalize | label) = P(normalize)`
is structural rather than a property the sampler has to remember. A per-spec draw
is the known leak: `container = mp3 if fake else wav` scored **AUC 1.000** on the
I1b metadata probe while every other invariant stayed green
([`docs/pipelines/05 §1`](../docs/pipelines/05-invariants.md)).

The four legs, and why these four:

| Variant | Why |
|---|---|
| `{}` | as decoded |
| `mp3 @ 64 kbps` | ⚠️ 64 kbps cost MusicDET **+37 EER points** |
| `flac` | lossless, but a different container |
| `telephone 8 kHz + µ-law` | ★ ASVspoof 5's hardest condition is codec-10 — 8 kHz, low bitrate — and that is our telephone slice |

⚠️ Short of the full A-S3 menu (AAC / OPUS / AMR-NB / GSM) for `render`'s reason:
each needs its encoder delay verified the way MP3's is, and an uncancelled delay
moves the audio while `frame_intervals` stay put.

🔴 **Do not audit the expanded stream with I1b.** The expansion emits four
near-duplicates of every spec and I1b's probe is *cross-validated*, so a row's
twins land in the other folds and the classifier memorises. Measured: expanding
four ways with **no codec at all** trips I1b at AUC **1.0000**. Audit the stream
the expansion is built from — `tests/test_loop.py::test_i1b_must_not_be_run_on_the_expanded_stream`
pins this so nobody "improves" it back.

---

### EMA and the soup

```python
from models.config import load_model_config
from models.model import DeepVoiceNet
from training.loop import EMA

model = DeepVoiceNet(load_model_config("configs/a_stub.yaml"))
ema = EMA(model, decay=0.999)
ema.update(model)
model.load_state_dict(ema.state_dict_for(model), strict=True)   # strict, always
```

🔴 The EMA is **bias-corrected**, like Adam's moments. A raw EMA is initialised at
the starting weights, so at decay 0.999 it is still 63% initialisation after 1,000
steps — on a short schedule the "EMA weights" would mostly be the random init and
the run would report a number for a model it never trained. With the correction,
the EMA after one update is *exactly* the current weights, which is an identity a
test can assert.

```python
from training.loop import checkpoint_soup
```

`checkpoint_soup([a, b, c])` is ★ free ensembling at zero inference cost, across
**epochs and seeds** — hence a list of files rather than an in-run buffer, since
the across-seed soup comes from separate runs. It **refuses** a mismatched key
set, a shape mismatch or a differing config rather than averaging what it can:
the average of two architectures is not a model, and a partial average is a
`load_state_dict` failure deferred onto whoever ships it. Integer buffers are
carried, not averaged.

---

### Validating a fold

```python
from training.loop import evaluate
```

`evaluate(model, frozen_dataset)` → a `ValidationReport` carrying the `MetricSet`
**and** `per_cell` **and** `per_generator`. The breakdowns are fields, not
something a caller may forget to ask for: a good pooled EER routinely hides a
collapsed cell, and cells 6/7 are the entire reason the competition has two fake
heads.

🔴 **Nothing here computes a metric.** `metrics.dacon.dacon_score`,
`metrics.breakdown` and `metrics.aggregate.fold_mean` do, and
`tests/test_loop.py` proves it by breaking `roc_curve` and watching `evaluate`
fail. Three details of the official estimator are load-bearing and a
reimplementation that "cleans up" any of them disagrees with the leaderboard.

⚠️ `evaluate` refuses a redrawable dataset, and `predict` uses `eval_batches` —
in order, unbucketed, nothing dropped.

🔴 **`LoopConfig.eval_precision` defaults to `fp32` and should stay there**, even
though inference ships fp16. bf16 carries 8 mantissa bits, and squashing a bf16
logit *ties files together*. Measured, and 🔴 the effect is size-dependent:

| n | bf16 unique | fp16 | fp32 | VG5 gate is `> 0.5 n` |
|---|---|---|---|---|
| 400 (a convenient fixture) | 241 ✅ | 370 ✅ | 400 ✅ | passes |
| **1,200 (the VAL floor)** | **399 ❌** | 996 ✅ | 1,200 ✅ | **fails** |

A bf16 evaluation passes VG5 on any fixture small enough to be convenient and
fails at the size we actually validate on.

---

### 🔴 Aggregation: the mean of per-fold metrics

```python
from training.loop import aggregate_folds
```

Each fold is scored by a **different model**, their score scales differ, and EER
is computed on the merged ranking — so concatenating raw OOF scores measured
**0.1705 against a true 0.100**
([`docs/validation/02 §4`](../docs/validation/02-metric-harness.md#4-how-we-aggregate)).
`aggregate_folds` delegates to `metrics.aggregate.fold_mean` and nothing in
`training.loop` concatenates a score column.

⚠️ **A single-fold run is first-class**, not a degraded mode — the 5-fold sweep is
often unaffordable and Replay speed is fold 0 by definition. What it does *not*
give is `Score_sd`, which is P4's input and the ★ E5 tiebreaker; `fold_mean`
returns **0.0** there, which reads as "perfectly stable". So `RunReport` carries
a caveat saying that 0.0 is an absence rather than a measurement, and
`sd_is_a_measurement` is `False`.

---

### 🔴 Tripwires: numbers too good to be true

```python
from metrics.dacon import MetricSet, roll_up
from training.loop import leak_tripwires

def metrics(eer_file, eer_voice, eer_music, auc=0.9):
    ads, cps, score = roll_up(eer_file, eer_voice, eer_music, auc, auc)
    return MetricSet(eer_file, eer_voice, eer_music, auc, auc, ads, cps, score,
                     2000, 2000, 2000)

fired = leak_tripwires(metrics(0.10, 0.09, 0.005), "generator_disjoint")
assert not fired.ok                                   # music EER 0.5% -- suspect a leak

blind = leak_tripwires(metrics(0.10, 0.0, 0.0), "generator_overlapping")
assert "L1_music_unseen_generator" in blind.skipped   # SKIP, never PASS
assert not blind.results["L3_perfect_separation"][0]  # the row that *does* apply
```

| Head | Suspicious if | Because |
|---|---|---|
| music fake, unseen generator | **< 3% EER** | published cross-generator is **46.4%** |
| voice fake, unseen generator | **< 1% EER** | ASVspoof 5's best is ~4% |
| any head | perfect separation on a random split | re-split by generator |

⚠️ It takes the `MetricSet` the **official harness already produced**, not a
prediction frame — re-deriving these EERs would put a second EER implementation
in the one repo that forbids them, and the tripwire could then disagree with the
number it guards.

🔴 **"Unseen generator" is measured, not declared.**

```python
from training.loop import measured_split_kind
```

`measured_split_kind(train_specs, val_specs, index)` compares the *realised*
generator families of the two drawn streams. A `split_kind=` argument would be
switched off by the same mistake the tripwires exist to catch — the caller who
believes the split is family-disjoint is exactly the caller whose 0.5% music EER
needs explaining. It is the house pattern of `training.registries`, which
*measures* time invariance rather than trusting a declaration field.

⚠️ A stream with no generated component returns `"undecidable"`, and L1/L2 then
SKIP. "Disjoint from nothing" is vacuously true and would arm the tripwires on a
split they cannot speak about.

---

### The gates, and the one that is missing

```python
from training.loop import run_gates, validate_fold
```

`validate_fold(model, eval_dataset, train_specs=...)` is the composed form: it
scores the fold, runs the gates and runs the tripwires, and returns one
`FoldResult`. 🔴 `train_specs` is a **required** argument, and that is why the
function exists — the tripwires need to know whether VAL is generator-disjoint
from TRAIN, that question is *measured*, and measuring it needs both streams. So
a `FoldResult` cannot be produced without the tripwires having run, or having
said by name why they could not.

| Gate | Wired to |
|---|---|
| **VG1** A1–A7, A10 | `training.folds.check_split_integrity` |
| **VG1** A8/A9 | `training.audit.audit_specs(..., eval_floors=True)` |
| **VG1** A1–A6 at draw time | the same audit's **I5** — needs `slice_` and `fold` |
| **VG2** | the same audit's **I1b** — E-S2 at spec level |
| **VG3** adversarial validation | ⚠️ **SKIP — not implemented** |
| **VG4** | `metrics.breakdown.t3_gap` |
| **VG5** B1–B5 | `training.loop.output_sanity` |
| **VG5** B2a | the same, qualifying B2 below the 1,200-row resolution floor |
| **VG6** | the `probe_openings.log` line count, refused at 4 |

🔴 **B2 is a ratio, so it is n-relative** — a bf16 column gives 241 unique of 400
and passes, then 399 of 1,200 and fails. **B2a** reports SKIP below 1,200 rows so
a small-n green B2 can never read as "ranking resolution confirmed". It does not
switch B2 off: a constant or saturated column is visible at any n.

⚠️ `slice_` must name the slice the eval specs were **drawn from**, not the role
they are being used in. `validate_fold` forwards `eval_dataset.slice_` and
`eval_dataset.fold` so I5 runs rather than SKIPs — and it immediately caught a
frozen set built from a `slice_="train"` sampler while `SpecDataset.frozen`'s
default labelled it `"val"`. Without a manifest I5 reports SKIP, never a pass.

⚠️ **VG3 is the honest gap, and it is deliberately deferred.** It needs a
TRAIN-vs-VAL classifier over the VG2 metadata features. It reports SKIP rather
than shipping green, because a stub would let a run claim a gate it never ran —
and a low VG3 AUC is only weak evidence of absence anyway. It is not implemented
now because there is no corpus: run against synthetic specs it would tell us
about `training/synthetic.py`, and the feature matrix it needs
(`training.audit._feature_frame`) is private — duplicating it would let VG2 and
VG3 silently disagree about what "metadata" means.

🔴 **If you are building the corpus, this is your exposure.** Two things are
unguarded until VG3 exists, and both are corpus-side:

- **Label-independent TRAIN/VAL domain drift.** VG2 asks whether metadata
  predicts the *label*; VG3 asks whether it predicts the *slice*. Nothing else
  asks the second question, so VAL families skewing to a different bitrate,
  duration or bandwidth regime move the fold score with every other gate green.
- **A corpus edit can silently break ledger comparability.** A composition change
  is supposed to re-trigger VG2 **and** VG3 and reset comparability with earlier
  ledger rows ([`docs/validation/02 §5`](../docs/validation/02-metric-harness.md#5-what-the-metrics-are-and-are-not-invariant-to));
  we can detect only the VG2 half of that re-trigger.

⚠️ The direction lost is the decisive one: a **high** VG3 AUC is decisive
evidence of a problem, a low one only weak evidence of its absence. What makes
the deferral safe is that the neighbours cover most of the same ground — VG1
A1–A7 give structural family/source/speaker disjointness, the L1/L2 tripwires
catch the *symptom* of a leak, and VG4's T3 gap is the matched control for the
corpus-identity risk VG3 was cited for. The residual is drift that never surfaces
as a too-good EER.

🔴 A SKIP is not a pass and is not a failure. `RunReport.quotable` is `False` only
for a *red* gate; skips are listed by name in `skipped_gates()` and printed by
`__str__`, so the shortfall lands in the ledger rather than in a docstring. If a
SKIP blocked, nothing would ever be quotable and the distinction would stop being
read.

⚠️ **No experiment is quotable without a VG1–VG6 pass recorded alongside it**
([`docs/validation/04`](../docs/validation/04-audit-gates.md)). `run_gates` +
`aggregate_folds` produce that record; `RunReport.as_ledger_row()` is the subset
of [`docs/validation/03 §5`](../docs/validation/03-decision-protocol.md#5-experiment-ledger)'s
schema this module can fill. The rest — `hypothesis`, `git_sha`, `parent_exp_id` —
is the experiment runner's, and 🔴 `hypothesis` is written **before** the run.

---

### What is deliberately not here

| Missing | Why |
|---|---|
| A `DataLoader` with workers | Worker processes would put the draw behind a second, per-worker RNG and the bitwise-resume guarantee would become a claim about `torch.utils.data`'s seeding. There is no corpus yet, so nothing is waiting on throughput. When there is: `DataLoader(dataset, batch_sampler=plan, collate_fn=collate)` **plus** a new resume test, in that order |
| Distributed training | Same reason, one level up. `_masked_mean` already keeps the graph connected so DDP does not deadlock on unused params |
| Teacher wiring / distillation | `TrainConfig.teachers` is validated (frozen, known frontend) but no teacher is loadable — `build_frontend` raises for anything but `stub` while the SSLAM / EAT / W2V-BERT licences are unverified. `multitask_loss` takes `teacher_emb` when there is one |
| An LR schedule | Rung 4 of [`architecture/08 §4b`](../docs/architecture/08-training-recipe.md)'s ladder is `optimizer`, and ★ the Kaggle ordering puts it **last**: *"common mistake: over-searching schedules before solving data shift and imbalance"*. A constant LR is the honest default until T1 has run |
| Running an experiment | **T1** (`clip_weight` 1.0 vs 0.5, Medium speed, never Replay) is the first one, and there is no corpus. This module is the instrument |
