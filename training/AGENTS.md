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

## Determinism

```python
from training.spec import spec_rng
rng = spec_rng(sample_id, epoch, seed)          # blake2b, not hash()
```

🔴 Python's `hash()` is salted per process, so a `hash()`-keyed stream is
reproducible *within* a run and different across runs — the exact opposite of
what A-S2 asks for. An epoch is a fixed count of drawn specs; without that,
`sample_id` is undefined and reproducibility is nominal.
