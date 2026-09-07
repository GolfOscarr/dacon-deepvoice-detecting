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
| **I1** | For every transform **name** `T` and label `L`: `P(T \| L) = P(T)`, within sampling error. ⚠️ **Names only** — parameter-level leaks are I1b's job | The governing rule ([data/06](../data/06-augmentation-spec.md#-the-governing-rule)). This is [E-S2](../data/07-eda-plan.md) made continuous — its **AUC < 0.60** gate becomes a per-commit check rather than a per-corpus one |
| **I1b** | 🔴 **Metadata-only regression cannot predict `FILE_FAKE`** — cross-validated AUC < **0.60**, over cells 1–8, **pooled *and* per stratum, worst taken**. Features: duration, composedness, structure, component count, gain, crossfade, source offset, **every transform parameter**, and **the whole `normalize` draw** | ★ [E-S2](../data/07-eda-plan.md) promoted to a per-spec check (VG2). **The joint guard**: every balance below can hold individually while a *combination* still separates the labels. ⚠️ Cell 9 excluded — always REAL *and* never composed, an unfixable correlation; including it the strict policy scores 0.6003 and fails spuriously |
| **I2** | Composedness is label-independent **within each presence stratum**: `f₁=f₂`, `f₃=f₄`, and `f₅ = (p₆f₆+p₇f₇+p₈f₈)/(p₆+p₇+p₈)`; cell 9 excluded | The composition trap ([02 §3](02-sampler.md#-the-implementable-form-stratified-with-f₈-as-the-only-knob)). ⚠️ Assert the **stratified** form: the marginal form passes while `f₁=1, f₂=0` makes "composed" predict REAL among voice-only files |
| **I2b** | 🔴 `P(mixed \| FILE_FAKE=1) = P(mixed \| FILE_FAKE=0)` over **cells 1–8**, and no presence pattern predicts fakeness | **C3 covers mixing, not only concatenation.** Mixedness is set by the cell, so it must be fixed in the cell mix ([02 §4](02-sampler.md#-c3-covers-mixing-not-only-concatenation--and-that-constrains-the-cell-mix)). ⚠️ A mix passing C1 can fail this at 0.769 vs 0.417 |
| **I3** | Every real component file that appears with `file_fake=1` also appears with `file_fake=0` | Content-identity shortcuts ([02 §3](02-sampler.md#one-level-down-components-on-both-sides-of-the-label)) |
| **I4** | 🔴 The **pool** each drawn component came from implies the cell's labels — pool B (fake voice) must not back a cell-1 (voice REAL) sample | ⚠️ **Rewritten.** The old version compared `spec.file_fake` to the expression that *defines* it, so a brute force over all nine cells found **0 constructible specs** that could trip it. This cross-checks an independent source |
| **I5** | No drawn `file_id` lies outside the active **slice *and* fold** | [VG1](../validation/04-audit-gates.md#vg1--split-integrity) A1–A6, at draw time rather than after. ⚠️ The first version checked slice only and ignored `fold` entirely |
| ~~**I6**~~ | ❌ **Removed.** "Labels come from the cell" is a property of the **type**, asserted in `tests/test_spec.py` (`SampleSpec` has no label fields). A per-spec loop over derived properties can only ever pass | The `docs/pipelines/05` opening warning, applied to itself |
| **I7** | `val_specs.parquet` meets ≥1,200 per class per **masked head pool** per fold, and ≥100 per cell per fold | [VG1](../validation/04-audit-gates.md#vg1--split-integrity) A8/A9, which can only be evaluated here — a component row has no cell. ⚠️ Implemented behind `eval_floors=True`; on a *training* stream it reports **SKIP**, never PASS. It previously printed PASS while existing nowhere |
| **I7a** | Cells 6/7 are never rendered whole-file | ⚠️ Defense-in-depth only — `SampleSpec` refuses to construct one, so it cannot fail today. It used to occupy I7's key |
| **I10** | 🔴 Realized **generator diversity** in the drawn stream: ≥3 artifact families per fake role, with the effective count reported | `domain_cap` is a weight over what is *present*, so a fold split leaving TRAIN generator-poor reproduces the DOSS failure (6.4k h naive **3.29%** vs 0.2k h balanced **2.77%**) with a green audit. ⚠️ An **absolute** floor: a monoculture slice trivially realizes 100% of its own two families |
| **I8** | Every head's positive rate, **computed after masking**, lies in **[0.2, 0.8]** | **C1** — removes BCE's imbalance pathology, which is the only reason any ranking/AUC-surrogate term would be worth adding ([02 §4](02-sampler.md#4--constraints-the-objective-imposes--c1-and-c2)). ⚠️ A naive 6/7-heavy cell mix **fails this** at 0.820 on both presence heads |
| **I9** | Every batch meets the per-head **present-count floor** | **C2** — `_masked_mean` scales ~1/√n, so `n=2` batches carry 3.9× the gradient norm of `n=32` ones |

🔴 **A review built a stream carrying three leaks that passed this entire section clean.** A
transform *parameter* (`rawboost(strength=0.9 if fake else 0.1)`, AUC 1.000) — I1 counts names, and
the name was balanced. The `normalize` draw (`container = mp3 if fake else wav`, AUC 1.000) — read
by nothing. And a duration shift of opposite sign in mixed vs non-mixed files (AUC 0.897 with the
stratum interaction, 0.499 marginally) — the pooled probe could not see it, while the model gets the
interaction for free because it is trained to predict presence. I1b now covers all three; the
regression tests are in `tests/test_audit.py`.

⚠️ **I1, I1b, I2b and I3 are the ones that would actually have caught the composition trap**, and neither
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
