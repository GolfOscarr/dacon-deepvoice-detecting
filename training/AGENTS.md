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
| `training.audit` | **I1–I9** over a drawn stream. No corpus, no model, no GPU |
| `training.folds` | Building `folds.parquet`, and **VG1 A1–A7 / A10** over it |
| `training.synthetic` | A manifest with the real corpus's pathologies |

⬜ `registries` · `render` · `collate` · `dataset` · `loop` — milestones M3–M5.

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

```python
from training.folds import FoldConfig, apply_folds, build_folds, check_split_integrity
from training.synthetic import synthetic_manifest

corpus = synthetic_manifest(n_per_pool=200, n_whole_file=200, n_families=24)
plan = build_folds(corpus, FoldConfig(n_folds=5))

print(plan)                       # slice counts, then one line per caveat
report = check_split_integrity(plan.frame, run_scheme_version="synthetic-v1")
report.raise_for_status()         # VG1 A1-A7 and A10

split = apply_folds(corpus, plan.frame)     # the manifest the pipeline reads
```

⚠️ `plan.caveats` is the part people skip. It is where the music-head variance
warning lives — at 5 music families a 5-fold puts **one family in each
validation fold** and PROBE cannot be carved out at all
([`docs/validation/01 §3`](../docs/validation/01-split-scheme.md#-the-music-head-cannot-support-the-planned-split)).
`plan.write()` drops them in `folds.caveats.txt` beside the parquet so they
travel with the table.

### Three things it refuses to do

`build_folds` raises `FoldInfeasible` rather than quietly returning a worse
split. Each of these is a real corpus property, not a bug in the caller:

| Refusal | Why |
|---|---|
| too few VAL families for `n_folds` | a validation fold with no family of its own validates nothing |
| a slice with no PROBE families | the sealed slice is the only check on "VAL became a training set". Pass `allow_no_probe=True` to accept the blind spot deliberately |
| a slice the sampler cannot draw from | no real voice components ⇒ cells 1/5/6 cannot be composed, and `Sampler` only finds out at draw time |

```python
import pytest

from training.folds import FoldConfig, FoldInfeasible, build_folds
from training.synthetic import synthetic_manifest

narrow = synthetic_manifest(n_per_pool=60, n_whole_file=60)   # 8 families/pool
with pytest.raises(FoldInfeasible, match="cannot fill 5"):
    build_folds(narrow, FoldConfig(n_folds=5))

plan = build_folds(narrow, FoldConfig(n_folds=2))             # 2 is feasible
assert any("per validation fold" in c for c in plan.caveats)
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

### Where A8 and A9 are

Not here. They are statements about **compositions**, and a component row has no
cell, so they run at eval-set materialization instead
([`docs/validation/04 §VG1`](../docs/validation/04-audit-gates.md#vg1--split-integrity)):

```python
from training.audit import run_audit
from training.folds import FoldConfig, apply_folds, build_folds
from training.sampler import Sampler
from training.synthetic import synthetic_manifest

corpus = synthetic_manifest(n_per_pool=200, n_whole_file=200, n_families=24)
split = apply_folds(corpus, build_folds(corpus, FoldConfig()).frame)

report = run_audit(Sampler(split, slice_="val"), n=1500, manifest=corpus,
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
| **A-A8 time shift**, **A-A11 silence edits** | They move audio along the timeline while `frame_intervals` stay put. An augment returns only a waveform, so it cannot tell the renderer the timeline moved. Both belong in the *placement* — jitter `ComponentDraw.target_start_s`, where the spec records it |
| **MixUp (A-A1/A-A2)** | Changes which components are present and whether they are generated, so it is a **step-2 component draw**, not a signal transform |
| **Resample (P-S2)** | Changes the sample count, and a registry whose steps may change `lengths` cannot be composed. It runs once per file in `render.load_audio`, with the resampler injected |
| **Test-chain normalization (A-S1/S3/S4)** | Not variety but a model of *what the organizers did to the test set* — drawn into `SampleSpec.normalize` and applied by the renderer, always last |
| **P-A1 loudness, P-A2 silence trimming** | Gate **G6**. `label_evidence` likewise refuses to run without `threshold_version` (**G3**): a default threshold is the hardcoded intuition [docs/data/10 §6](../docs/data/10-preprocessing-and-filtering.md) exists to forbid, and it would be invisible in a green suite |

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
