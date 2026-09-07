# Data Pipeline (Phase B→D bridge)

Everything that turns the [component pools](../data/02-label-taxonomy.md) into a batch the
[model](../../models/AGENTS.md) can consume.

| File | Contents |
|---|---|
| [01-sample-contract.md](01-sample-contract.md) | ⭐ **The two-stage split** — `spec → render`, the manifest schema, the `SampleSpec` schema |
| [02-sampler.md](02-sampler.md) | Cell and component sampling, split safety, DOSS capping, the frozen eval set, epoch semantics |
| [03-transforms.md](03-transforms.md) | **The three registries** — preprocess / augment / filter, and how an EDA finding gets plugged in |
| [04-collation.md](04-collation.md) | Batch layout, padding, frame targets, duration bucketing |
| [05-invariants.md](05-invariants.md) | **The assertions** — what is tested rather than asserted in prose |

## Where this sits

[`data/`](../data/README.md) says *what audio we acquire and what may legally be done to it*.
[`validation/`](../validation/README.md) says *how slices are cut and how results are read*.
[`architecture/`](../architecture/README.md) says *what consumes the batch*, and
[`training/`](../training/README.md) says *what the loss does with it* — including three constraints
(**C1**, **C2**, **C3**) that the sampler, not the loss, has to satisfy. This directory is the code
contract joining them, and it is the last design artifact before training code exists.

⚠️ **The corpus does not exist.** Phase B is planned and not executed
([data/08](../data/08-build-plan.md)). Every schema here is written so the pipeline can be built
and fully tested against `stub` frontends and a synthetic manifest *now*, and run unchanged the day
the corpus lands.

## The pipeline in seven lines

1. **Two kinds of manifest row** — *component* files that get composed, *whole* files used as-is.
   `cell` is a property of a composition, so it is null on component rows
   ([validation/01](../validation/01-split-scheme.md#-the-table-is-keyed-on-components-not-on-composed-files)).
2. **Sampling is split from rendering.** `sample_spec()` is pure and touches no audio;
   `render()` does all I/O and DSP. Almost every guarantee below follows from this one split.
3. **Labels come from the spec's component draws and nothing else** — the invariant that makes the
   shortcut audit checkable without decoding a single file.
4. **Three transform registries**, separated by whether the transform runs at test time:
   preprocess (shipped), augment (train-only), filter (offline sidecar).
5. **Whole-file, not tiled.** Training and inference run in the same length regime.
6. **The eval set is a frozen list of specs on disk**, not a seed.
7. Every rule above is a test in [05](05-invariants.md), not a paragraph.

## Non-negotiables

- 🔴 **Rule 2.4**: nothing a file's score depends on may come from another file. Every preprocess
  step is per-file, and batch-invariance is asserted, not assumed — three rule-2.4 violations have
  already shipped and been caught in this repo ([`PROGRESS.md`](../../PROGRESS.md))
- 🔴 **Augment functions never receive the labels.** Structural enforcement of
  `P(T | L) = P(T)` ([data/06](../data/06-augmentation-spec.md#-the-governing-rule)), not a convention
- 🔴 **The composed fraction is equal across cells 5–8** (**C3**), or "composed" predicts FAKE.
  ★ Measured, not inferred: EER above **14%** at zero concatenation boundaries
  ([02 §3](02-sampler.md#3-the-composed-fraction-is-a-constraint-not-a-knob))
- 🔴 **C1 — per-head positive rate ∈ [0.2, 0.8] after masking**, and **C2 — a per-head
  present-count floor per batch.** Both come from [`training/`](../training/README.md) and the loss
  is unsound without them ([02 §4](02-sampler.md#4--constraints-the-objective-imposes--c1-and-c2)).
  ⚠️ A naive 6/7-heavy cell mix violates C1 on both presence heads
- 🔴 **Preprocess code is shared with `submit.zip`**, one module, one call site. A transform that
  runs in training and not at test time teaches the model our pipeline
  ([data/10 P2](../data/10-preprocessing-and-filtering.md#-p2--transformations-must-be-traintest-symmetric-filters-are-train-only))
- 🔴 **Never compute EER in pipeline or training code** — import it ([`metrics/AGENTS.md`](../../metrics/AGENTS.md))
- Target the **server's** Python 3.11.15 / numpy 1.26.4 / pandas 2.0.3 from the first commit

## Status

| | |
|---|---|
| Sample contract | ✅ designed — [01](01-sample-contract.md) |
| Sampler | ✅ designed; ⬜ blocked on `folds.parquet`, which is blocked on the corpus |
| Transform registries | ✅ designed; ⚠️ parameters blocked on **G1** `signal_chain.yaml` |
| Collation | ✅ designed |
| Invariants | ✅ specified (**I1–I21**); ✅ I1–I21 implemented in `training/` (I16 at the model boundary) |
| Any of it in code | ⬜ **none** |
