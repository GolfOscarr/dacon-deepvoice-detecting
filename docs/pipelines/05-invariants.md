# 05 — Invariants

The rules in [01](01-sample-contract.md)–[04](04-collation.md), as assertions.

🔴 **This file exists because of a measured pattern in this repo**: six defects were found in one
review session, *none* by a green test suite, and two shapes recurred — a test asserting an
*adjacent* quantity (`clip_logits` instead of the submitted probability), and a component inherited
from a source recipe that was correct in its regime and silently wrong in ours
([`PROGRESS.md`](../../PROGRESS.md)). Both are worth testing for deliberately rather than trusting
coverage.

---

## 1. Cheap — spec-level, no audio decoded

These run over 10⁵ generated `SampleSpec`s in seconds, because sampling is separated from rendering
([01 §1](01-sample-contract.md#1-the-split-that-everything-else-follows-from)).

| # | Assertion | Guards |
|---|---|---|
| **I1** | For every transform `T` and label `L`: `P(T \| L) = P(T)`, within sampling error | The governing rule ([data/06](../data/06-augmentation-spec.md#-the-governing-rule)). This is [E-S2](../data/07-eda-plan.md) made continuous — its **AUC < 0.60** gate becomes a per-commit check rather than a per-corpus one |
| **I2** | Composedness is label-independent **within each presence stratum**: `f₁=f₂`, `f₃=f₄`, and `f₅ = (p₆f₆+p₇f₇+p₈f₈)/(p₆+p₇+p₈)`; cell 9 excluded | The composition trap ([02 §3](02-sampler.md#-the-implementable-form-stratified-with-f₈-as-the-only-knob)). ⚠️ Assert the **stratified** form: the marginal form passes while `f₁=1, f₂=0` makes "composed" predict REAL among voice-only files |
| **I2b** | 🔴 `P(mixed \| FILE_FAKE=1) = P(mixed \| FILE_FAKE=0)` over **cells 1–8**, and no presence pattern predicts fakeness | **C3 covers mixing, not only concatenation.** Mixedness is set by the cell, so it must be fixed in the cell mix ([02 §4](02-sampler.md#-c3-covers-mixing-not-only-concatenation--and-that-constrains-the-cell-mix)). ⚠️ A mix passing C1 can fail this at 0.769 vs 0.417 |
| **I3** | Every real component file that appears with `file_fake=1` also appears with `file_fake=0` | Content-identity shortcuts ([02 §3](02-sampler.md#one-level-down-components-on-both-sides-of-the-label)) |
| **I4** | `spec.file_fake == file_fake_label(...)` from [`metrics.dacon`](../../metrics/AGENTS.md), over present components | One definition of the label, not two |
| **I5** | No drawn `file_id` lies outside the active slice; family/source/speaker/pair/dup constraints hold | [VG1](../validation/04-audit-gates.md#vg1--split-integrity) A1–A6, at draw time rather than after |
| **I6** | Labels are unchanged when `transforms` and `normalize` are resampled with the components held fixed | That labels come from steps 1–2 only |
| **I7** | `val_specs.parquet` meets ≥1,200 per class per **masked head pool** per fold, and ≥100 per cell per fold | [VG1](../validation/04-audit-gates.md#vg1--split-integrity) A8/A9, which can only be evaluated here — a component row has no cell |
| **I8** | Every head's positive rate, **computed after masking**, lies in **[0.2, 0.8]** | **C1** — removes BCE's imbalance pathology, which is the only reason any ranking/AUC-surrogate term would be worth adding ([02 §4](02-sampler.md#4--constraints-the-objective-imposes--c1-and-c2)). ⚠️ A naive 6/7-heavy cell mix **fails this** at 0.820 on both presence heads |
| **I9** | Every batch meets the per-head **present-count floor** | **C2** — `_masked_mean` scales ~1/√n, so `n=2` batches carry 3.9× the gradient norm of `n=32` ones |

⚠️ **I1, I2b and I3 are the ones that would actually have caught the composition trap**, and neither
needs a model, a corpus, or a GPU. They should exist before the corpus does.

---

## 2. Render-level

| # | Assertion | Guards |
|---|---|---|
| **I10** | `render(spec) == render(spec)` bitwise, across processes | [A-S2](../data/06-augmentation-spec.md) / [R9](../data/09-risks-and-checks.md); required for the 2nd-stage submission to reproduce the Private score |
| **I11** | Two specs differing only in `sample_id` produce different audio | That the RNG key is actually wired, not defaulted |
| **I12** | A rendered sample's duration is in `[AudioConfig.min_seconds, AudioConfig.max_seconds]` | Length-regime parity ([04 §1](04-collation.md#-length-regime-parity)) |
| **I13** | `frame_intervals` lie within `[0, duration_s]` and their union matches the component placement | That frame targets and audio describe the same timeline |

---

## 3. 🔴 Rule 2.4 — the batch-invariance suite

> Nothing a file's score depends on may come from another file.

Three violations have shipped in this repo and been caught: `bandpass` filtering over the padded
batch, `align_time` interpolating over padded frame counts, and `frame_max` being padding-sensitive
in the **submitted probability** while the test asserted on `clip_logits`.

| # | Assertion | Guards |
|---|---|---|
| **I14** | Every registered **preprocess** step: output over a row's valid prefix is bitwise identical solo and inside a batch of longer, louder rows | New preprocess plugins repeating the `bandpass` defect |
| **I15** | Collating the same sample into batches of different composition yields identical `wav[:len]`, `lengths` and `targets` | The collator itself |
| **I16** | Two different pad fillings give the same submitted probability | Already covered for the model; extended to cover pipeline output |

⚠️ **I12 must assert on the quantity that ships.** The `frame_max` violation survived because its
test asserted on an adjacent quantity. Any new test here states which submitted number it protects.

---

## 4. Contract tests against the model

| # | Assertion | Guards |
|---|---|---|
| **I17** | A collated batch is accepted by `DeepVoiceNet.forward(wav, lengths)` after `prepare_waveform`, on both shipped stub configs | Silent drift between pipeline and model |
| **I18** | `batch["targets"]` keys are exactly `set(losses.TARGET_FOR_COLUMN.values())` | A renamed target key failing at hour three of a run |
| **I19** | `multitask_loss` runs on the batch and every branch contributes a finite value | Masking bugs that zero a head |
| **I20** | A batch where **no** row carries a component still produces a finite loss | `_masked_mean`'s zero-denominator path, and DDP not deadlocking on unused params |

---

## 5. What already exists — do not rebuild it

Part of §3 and §4 is already covered at the **model** level. The pipeline versions extend the same
guarantee to pipeline output; they are not replacements, and the existing tests stay.

| Existing test | Covers |
|---|---|
| `tests/test_model.py::test_score_does_not_depend_on_the_batch` | I15, at the model boundary |
| `tests/test_model.py::test_padding_content_cannot_leak_into_a_score` | **I16** — this is the "two pad fillings" guard |
| `tests/test_model.py::test_submitted_probabilities_are_batch_invariant` | I16, on the quantity that ships |
| `tests/test_model.py::test_ranking_over_a_canned_set_is_batch_invariant` | Rule 2.4 on the ordering itself |
| `tests/test_audio.py::test_filter_output_is_independent_of_padding` | **I14**, for `bandpass` only — generalize the pattern to the registry |
| `tests/test_utils.py::test_alignment_is_independent_of_padding_width` | The `align_time` regression |
| `tests/test_losses.py::test_batch_with_no_present_component_is_not_nan` | **I20**, at the loss boundary |

So the genuinely new work is **I1–I13, I15, I17–I19**, plus generalizing I14 from one function to a
registry. 🔴 I1–I9 need no corpus, no model and no GPU — they are frequency tables over generated
specs, and they should exist before the corpus does.

## 6. What is deliberately *not* asserted here

- **Audio quality.** [data/10 P1](../data/10-preprocessing-and-filtering.md#-p1--filter-for-label-evidence-sufficiency-not-for-cleanliness):
  noise is a property of the target domain. There is no "is this clean" test.
- **Score quality.** ⚠️ The leak tripwires — music fake unseen-generator **< 3% EER**, voice
  **< 1%**, or perfect separation on a random split — belong to the training loop's validation
  step, not the pipeline ([architecture/08 §5](../architecture/08-training-recipe.md#5-validation-is-already-built--use-it)).
- **Cell-mix correctness.** The mix is a knob with no ground truth ([02 §8](02-sampler.md#9-one-thing-we-cannot-resolve-locally));
  only its *stability* across a run is assertable, not its rightness.
