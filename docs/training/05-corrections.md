# 05 — Corrections This Evidence Requires

✅ **All rows below are now APPLIED.** Each was a claim standing elsewhere in the repo — in code,
in a docstring, or in a design doc — that this directory's evidence contradicts. The table is kept
as the audit trail: it records what the repo used to say and why it changed.

🔴 **Two of them changed on inspection.** `C-2` named the wrong location (no config sets
`clip_weight`; it is a dataclass default) **and**, applied as written, would have silently made the
rule-2.4 `frame_max` guards vacuous — measured: reintroducing the original unmasked-`frame_max`
defect moves the submitted logit by **1.83** at `clip_weight=0.5` and by **0.000** at `1.0`. The
guards were hardened to force the blend on, and `test_clip_only_default_makes_frame_max_guards_vacuous`
now pins that fact. `D-11` was checked against git rather than assumed. They are listed
rather than silently fixed because two of them change **shipped, tested components**, and because
the repo's convention is to record corrections rather than apply them quietly
([architecture/README](../architecture/README.md)).

---

## 1. Code

| # | Location | Standing claim | Correction |
|---|---|---|---|
| ✅ **C-1** | [`models/losses.py`](../../models/losses.py) — `pairwise_ranking_loss` docstring | *"EER is a pure ranking metric, so a pairwise loss optimises it directly — independently arrived at by TFPARN and by the LLM-Detect-AI winner"* | ❌ **TFPARN's own ablation refutes this for EER**: 12.91 → 12.92. Its gains are minDCF/Cllr/actDCF. The function may stay (it is correct code, default weight 0); the justification must not ([03 §1](03-ruled-out.md#1-pairwise-ranking-loss---weight-stays-0)) |
| ✅ **C-2** | `configs/a_stub.yaml`, `configs/b_stub.yaml`, `configs/a_shared_trunk.yaml`, `configs/b_three_branch.yaml` | `clip_weight: 0.5` on all five branches, chosen by symmetry | ⚠️ Default to **1.0** pending **T1**. Our config is the union of the two configurations that measured *worse than utterance-only* ([02 §3](02-the-loss.md#3--clipweight--10--the-one-change-worth-engineer-days)) |
| ✅ **C-3** | `configs/train_joint.yaml` and `LossConfig.weights` default | `voice/music/file/v_pres/m_pres = 1.0` each, inherited from PC-Mix | Set **metric-proportional** .45/.27/.18/.05/.05 ([02 §4](02-the-loss.md#4-per-head-weights--metric-proportional-by-argument)) |
| ✅ **C-4** | [`models/losses.py`](../../models/losses.py) — `_masked_mean` docstring | *"Normalising by `mask.sum()` … would make a head's effective learning rate move with how many present-component files happened to land in the batch"* | ⚠️ **Mechanism is loose.** Measured: *both* schemes move with `n_present`, in opposite directions (~1/√n vs ~√n; 16× spread at n=2). What subset-normalization actually fixes is the **loss scale**, and the conclusion survives at the epoch level. The choice is still right for us, for a reason the docstring does not give — under batch-normalization the cell mix would silently reweight the heads ([02 §4](02-the-loss.md#-the-weight-you-set-is-not-the-weight-in-effect)) |

🔴 **C-2 is the only one that changes behaviour rather than prose**, and it is a config value, not
code. No model code changes are proposed here. If **T1** shows the frame term carries signal, the
MulBS split in [02 §3](02-the-loss.md) *would* touch `models/heads.py` — that is a separate decision
and is not requested by this file.

---

## 2. Documentation

| # | Location | Standing claim | Correction |
|---|---|---|---|
| ✅ **D-1** | [`architecture/08 §1`](../architecture/08-training-recipe.md#1-the-staged-schedule) | S2 joint stage is *"the largest single gain"*, citing PC-Mix ACC 69.40 → 85.12 and CompSpoof F1 0.668 → 0.908 | ⚠️ Both are **thresholded metrics**. The EER deltas from the same PC-Mix ablation are **3.59 → 3.12** and **8.72 → 7.86** — below our floor. Keep the stage, re-grade the claim ([04 §2](04-schedule.md#2--s2s-evidence-is-a-thresholded-metric-result--keep-the-stage-drop-the-claim)) |
| ✅ **D-2** | [`architecture/08 §1`](../architecture/08-training-recipe.md#1-the-staged-schedule) | S4 ranking polish, ☆ ~1 bps | 🔴 **Delete the stage** ([04 §4](04-schedule.md#4--s4-is-dropped)) |
| ✅ **D-3** | [`architecture/08 §2`](../architecture/08-training-recipe.md#2-the-loss) | `L_head = 0.5·BCE(clip) + 0.5·BCE(frame_max) + λ_rank · pairwise` | Revise to clip-only with `λ_rank = 0` ([02 §1](02-the-loss.md#1-the-spec)) |
| ✅ **D-4** | [`papers/04 §synthesis`](../papers/04-component-partial.md) | *"Joint training across component branches is the largest single gain in both papers \| PC-Mix +15.72 ACC · CompSpoof 0.668→0.908 F1"* | Annotate: both thresholded; CompSpoof's delta is **inside a separation-based system** whose no-separation baseline is 0.827 |
| ✅ **D-5** | [`papers/09`](../papers/09-training-losses.md) | *"No papers on this axis were selected for deep reading… **Promote to deep read if** we commit to a design decision that depends on this axis"* + *"adopt the loss, not the system"* | ✅ **The promotion criterion has fired and been discharged.** Rewrite the file with the four-axis deep read; reverse *"adopt the loss"* |
| ✅ **D-6** | [`architecture/04 §4`](../architecture/04-heads-and-pooling.md#4-multi-resolution-supervision--an-advantage-we-can-produce-but-should-not-lean-on) | Multi-resolution frame supervision is *"the advantage we get for free"* | ⚠️ Downgrade. Measured **+0.15 in-domain, −0.13 out-of-domain**, and the inventors declined our partial-label case (footnote 11). Its 🔷 note that the frame weight is a *"shortcut-exposure knob"* is **upgraded to ★** ([03 §4](03-ruled-out.md#4-multi-resolution-frame-supervision---downgraded-from-headline-to-optional)) |
| ✅ **D-7** | [`data/02`](../data/02-label-taxonomy.md#-the-mechanism-one-composed-fraction-shared-across-cells) | `f₅ = f₆ = f₇ = f₈` marked 🔷 our inference | ✅ **Upgrade to ★ measured.** PartialSpoof Fig. 5 / §VI-D: EER **above 14% at zero concatenation boundaries**, authors attributing it to overlap-add artifacts |
| ✅ **D-8** | [`PROGRESS.md`](../../PROGRESS.md) | *"Three parallel branches … with joint training across branches (the largest single gain in both component papers)"* | Same re-grading as D-1 |
| ✅ **D-9** | [`architecture/09 B11`](../architecture/09-open-questions.md) | Component loss weights, *"⚠️ probably not resolvable individually"* | ✅ **Confirmed and closed by argument.** A targeted search found **no study** testing loss-weights ∝ metric-weights. Resolution recorded in [02 §4](02-the-loss.md#4-per-head-weights--metric-proportional-by-argument) |
| ✅ **D-10** | [`survey/09`](../survey/09-challenge-playbooks.md) | WaveShield's *"AM-Softmax + focal loss"* listed among the winner's choices | ⚠️ Annotate as **not evidence**: no ablation published, confounded with a 3-member ensemble + LoRA staging + six augmentation families, and scored on **Macro-F1 at a fixed threshold** ([03 §2](03-ruled-out.md#2-focal-loss---provably-zero)) |
| ✅ **D-11** | [`PROGRESS.md`](../../PROGRESS.md) | *"247 tests green on `feat/model-architecture`"* | ⚠️ Pre-existing and unrelated to this work: `main` runs **230**. Flagged for accuracy, not caused here |

---

## 3. Additions rather than corrections

| # | Where | What |
|---|---|---|
| **A-1** | [`pipelines/02 §4`](../pipelines/02-sampler.md#4--constraints-the-objective-imposes--c1-and-c2) | ✅ **APPLIED.** C1 and C2 added, with the per-head `π` formulas, the derived presence-head bound `p3+p4+p9 ≥ 0.2` / `p1+p2+p9 ≥ 0.2`, and a reference cell mix. 🔴 Applying it **found a defect**: the previous "over-weight cells 6 and 7" guidance violates C1 at 0.820 on both presence heads |
| **A-2** | [`pipelines/05`](../pipelines/05-invariants.md) | ✅ **APPLIED** as **I8** (C1) and **I9** (C2); the suite renumbered to I1–I20, and later **I1–I21** when generator diversity was moved off I10 ([`pipelines/05 §1`](../pipelines/05-invariants.md)) |
| **A-3** | [`validation/03`](../validation/03-decision-protocol.md) | ⚠️ Record the **model-selection warning** from ASVspoof 5: the one-class arm produced the *best dev* EER (12.32) and a **+9.24 pt** dev→progress collapse, against softmax's +1.12. We select on VAL; our ±1.7 floor cannot catch a failure of that shape. This is what the sealed PROBE slice is for |
| **A-4** | [`metrics/breakdown.py`](../../metrics/breakdown.py) | Log `w_c / p_c` per head as a standing diagnostic — the weight in effect is not the weight configured ([02 §4](02-the-loss.md#-the-weight-you-set-is-not-the-weight-in-effect)) |

---

## 4. What this evidence does **not** overturn

Recorded so the corrections above are not read as broader than they are.

- **Masked component losses** — unchanged and reinforced ([01 §2.1](01-what-the-metric-demands.md)).
- **The three-branch, no-separation layout** — untouched. PC-Mix's *architecture* claim is separate
  from its joint-training *magnitude* claim, and only the latter is re-graded.
- **S3 codec-aware training** — ★ unchanged, and now the **best-evidenced stage in the recipe**.
- **The non-saturating output map** — unchanged; measured 0.0950 → 0.3017 under saturation.
- **Distillation** — unchanged, and still conditional on [09 B9](../architecture/09-open-questions.md).
- **`pairwise_ranking_loss` as code** — correct, tested, default weight 0. Only its stated
  justification is wrong.
