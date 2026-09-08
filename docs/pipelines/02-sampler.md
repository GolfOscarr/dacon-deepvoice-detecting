# 02 — The Sampler

How a `SampleSpec` is drawn. ★ This is the stage the Kaggle ordering puts *second*, ahead of
architecture — `data split → sampler/loss → architecture → optimizer`, with a named warning against
over-searching architecture before solving data shift
([kaggle/05 E8](../kaggle/05-transferable-playbook.md)).

---

## 1. Two pure levels

```
CellSampler(rng, fold, slice)       -> cell, render_mode
ComponentSampler(rng, role, slice)  -> file_id, offset_s, gain_db, placement
```

Neither touches audio; both read the [manifest](01-sample-contract.md#2-the-manifest) only.

### Split safety binds at level 2

Filter the manifest to the active `slice` **before drawing**, then:

| Key | Constrains |
|---|---|
| `artifact_family` | **fake components only** — real files have no family |
| `source_name`, `speaker_ref_id`, `pair_id`, `dup_group` | both real and fake |

⚠️ A composed sample draws its voice and music components independently. The *combination* is
therefore novel almost every time, including combinations never seen in TRAIN. That is intended —
the axis being held out is the generator, not the pairing.

---

## 2. DOSS capping is a sampler weight, not a corpus edit

🔴 **DOSS measured 0.2k h domain-balanced → 2.77% EER against 6.4k h naive → 3.29%**
([papers/05](../papers/05-generalization.md)). More data is not the lever; domain balance is. But
the paper's version caps files per domain at *dataset construction*, and we compose an unbounded
stream — so the analogue is a sampling weight over `domain_key`:

```
w(file) = min(count(domain(file)), N_c) / count(domain(file))
```

A domain with fewer than `N_c` files is drawn at full weight; a domain with 10× `N_c` is drawn at
1/10 weight. Same balancing effect, **nothing discarded**, and `N_c` becomes a sweepable float
instead of a destructive preprocessing pass.

Start at `N_c = 500`. ⚠️ Capping applies to **fake** domains; the real side is governed by the
BR/AG balance finding instead — generator-diverse 0.74 AUC · speaker-diverse 0.69 ·
**balanced 0.82** ([papers/05](../papers/05-generalization.md)) — so real-source diversity is
scaled *alongside* generator count, not after it.

---

## 3. The composed fraction is a constraint, not a knob

Let `f_c` be the fraction of cell-`c` samples drawn from component rows rather than whole-file rows.

> 🔴 `f₅ = f₆ = f₇ = f₈`

Cells 6 and 7 can **only** be composed — they cannot be scraped, which is the entire reason two
fake heads exist. If cell 5 is only natural songs, "composed" predicts FAKE perfectly, local CV
looks superb, and the leaderboard does not move
([data/02](../data/02-label-taxonomy.md#-the-mechanism-one-composed-fraction-shared-across-cells)).

### 🔴 The implementable form: stratified, with `f₈` as the only knob

⚠️ Pairwise equality `f₅=f₆=f₇=f₈` is *sufficient but too strong*: with `f₆=f₇=1` forced it drives
`f₅=f₈=1`, banning natural songs and AI songs outright. ⚠️ And the naive weakening — marginal
`P(composed|FAKE) = P(composed|REAL)` — is *too weak*: it can hold while composedness predicts the
label inside a stratum (`f₁=1, f₂=0` balances marginally yet "composed" ⇒ REAL among voice-only
files). **Both verified numerically.**

The form that is neither:

```python
f[6] = f[7] = 1.0                                     # forced: cells 6/7 cannot be scraped
f[1] = f[2] = a                                       # voice-only: any common rate
f[3] = f[4] = b                                       # music-only: any common rate
f[5] = (p[6]*f[6] + p[7]*f[7] + p[8]*f[8]) / (p[6] + p[7] + p[8])
#      cell 9 excluded: a file with no components cannot be fake
```

**`f₈` selects the policy.** Under the [reference mix](#the-reference-cell-mix):

| `f₈` | required `f₅` | genuine whole-file mass | policy |
|---|---|---|---|
| **0.00** | **0.725** | **13.8%** | ⭐ **conditional — primary** |
| 0.25 | 0.793 | 10.3% | |
| 0.50 | 0.862 | 6.9% | |
| 1.00 | 1.000 | 0% | **strict** — every mixed file composed |

🔴 **Ship `f₈ = 0`**: all AI songs used as-is, ~27.5% of cell 5 as natural songs. **`f₈ = 1`
recovers the strict policy from the same formula** — a value change, not a second code path — and is
the move if the metric degrades. ✅ Excluding cell 9, the stratified constraint also gives marginal
balance, verified to within **0.0014** for any `a`, `b`.

🔴 **This is now measured, not inferred.** ★ PartialSpoof
([arXiv:2204.05177](https://arxiv.org/abs/2204.05177)) §VI-D / Fig. 5 breaks EER down by number of
concatenation boundaries: it degrades sharply as boundaries fall and is **worst — above 14% — at
zero boundaries**, the authors attributing it to overlap-add artifacts around the joins. ⚠️ **Worse
for us than for them**: their eval set is also composed, whereas a genuine AI song or a real phone
call with hold music in the DACON test set carries no splice artifact at all. Full reading in
[training/03 §4](../training/03-ruled-out.md#4-multi-resolution-frame-supervision---downgraded-from-headline-to-optional).

⚠️ **The consequence for Phase B**: natural-song volume usable in cells 5 and 8 is bounded by how
much 6/7 material exists, not by how much we can acquire. Surplus natural songs are simply not
sampled.

### One level down: components on both sides of the label

🔷 Every component file should appear on both sides of `FILE_FAKE` wherever its type permits — a
real voice file supports cells 1 and 5 (`FILE_FAKE=0`) **and** cell 6 (`FILE_FAKE=1`); a real
instrumental supports 3, 5 and 7. Fake components are asymmetric by construction and cannot. This
catches content-identity shortcuts, which the transform-level rule does not, and it is assertable
over the spec stream ([05](05-invariants.md)).

---

## 4. 🔴 Constraints the objective imposes — C1 and C2

These come from [`training/`](../training/README.md) and are **properties of the sampler**. The loss
is unsound without them, and both are checkable over the spec stream with no audio.

### C1 — per-head positive rate ∈ [0.2, 0.8], measured *after masking*

Under BCE, a negative sample's gradient scales with the predicted probability, so at low positive
rates the loss barely trains on negatives. That pathology is the **only** reason an AUC-surrogate or
pairwise ranking term would be worth adding — and every such term measured at or below zero for us
([training/01 §2.4](../training/01-what-the-metric-demands.md#24--we-control-the-imbalance-so-we-remove-the-pathology-instead-of-patching-it)).
🔴 **We compose the corpus, so we decline the imbalance rather than buying a loss term to survive it.**

Given cell probabilities `p₁…p₉` and the [taxonomy](../data/02-label-taxonomy.md#the-8-cells):

```
π_file    = p2 + p4 + p6 + p7 + p8
π_voice   = (p2 + p7 + p8) / (p1 + p2 + p5 + p6 + p7 + p8)      # masked to voice-present
π_music   = (p4 + p6 + p8) / (p3 + p4 + p5 + p6 + p7 + p8)      # masked to music-present
π_v_pres  = p1 + p2 + p5 + p6 + p7 + p8
π_m_pres  = p3 + p4 + p5 + p6 + p7 + p8
```

⚠️ **The two presence heads are the binding constraint, and they bind against advice given
elsewhere in this file.** Since `π_v_pres ≤ 0.8` and `π_m_pres ≤ 0.8` reduce to:

```
p3 + p4 + p9  ≥  0.2        # voice-presence negatives
p1 + p2 + p9  ≥  0.2        # music-presence negatives
```

…the **single-component cells and cell 9 are the only source of presence negatives**, and cells
6/7/8 all have *both* components present. 🔴 **Naively over-weighting cells 6 and 7 violates C1.**
Measured over candidate mixes:

| Cell mix | π_file | π_voice | π_music | π_v_pres | π_m_pres | C1 |
|---|---|---|---|---|---|---|
| Uniform 1/9 | 0.556 | 0.500 | 0.500 | 0.667 | 0.667 | ✅ |
| **Reference (below)** | 0.600 | 0.486 | 0.486 | 0.720 | 0.720 | ✅ |
| Naive 6/7-heavy | 0.720 | 0.512 | 0.512 | **0.820** | **0.820** | ❌ both presence heads |

### 🔴 C3 covers *mixing*, not only concatenation — and that constrains the cell mix

[data/02](../data/02-label-taxonomy.md#-the-composition-trap) lists **mixing** alongside
concatenation, crossfade and gain as a structural transform that *"must appear on both sides of
every label."* **Mixedness is determined by the cell**, so unlike the other transforms it cannot be
fixed in the transform sampler — it has to be fixed in the **cell mix itself**:

```
P(mixed | FILE_FAKE=1)  =  P(mixed | FILE_FAKE=0)          over cells 1-8
```

⚠️ **Cell 9 is excluded, deliberately.** A file with no components present cannot be fake, so
`P(FAKE | cell 9) = 0` by definition. That is a true property of the label space — it holds at test
time too — not a corpus artifact, and including cell 9 lets a mix appear balanced for the wrong
reason.

⚠️ **An earlier draft of this section failed this check.** A mix of
`.10 .10 .10 .10 .12 .15 .15 .10 .08` satisfies C1 but gives
**P(mixed | FAKE) = 0.667 vs P(mixed | REAL) = 0.375**, so *"is a mixed file"* predicted FAKE at
**0.769 vs 0.500**. That is the composition trap wearing a different hat, and C1 alone does not
catch it.

⚠️ Those four figures are computed **over cells 1–8**, per the definition above. An earlier draft
quoted 0.300 and 0.417, which are the cell-9-*inclusive* values — inconsistent with the exclusion
stated two paragraphs earlier, and caught by the mutation test in `tests/test_audit.py`. The verdict
is unchanged: the gap is 0.292 rather than 0.367, and still a large violation.

### The reference cell mix

🔷 Verified against C1 **and** C3. Over-weights cells 6/7 above their uniform 1/9 share, keeps a
real cell-9 allocation, and leaves every head inside its bound:

| cell | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| p | .060 | .130 | .060 | .135 | .155 | **.125** | **.125** | .095 | **.115** |

| Check | Value | |
|---|---|---|
| `π_file` | 0.610 | ✅ |
| `π_voice` · `π_music` | 0.507 · 0.511 | ✅ |
| `π_v_pres` · `π_m_pres` | 0.690 · 0.695 | ✅ |
| **C3** — `P(mixed\|FAKE)` vs `P(mixed\|REAL)` | 0.566 vs 0.564 | ✅ gap 0.002 |
| `P(FAKE \| voice-only / music-only / mixed)` | 0.684 · 0.692 · 0.690 | ✅ **flat — presence predicts nothing** |

🔴 **The last row is the point.** No presence pattern carries information about fakeness, so the
model cannot reach `FILE_FAKE` through the presence heads.

⚠️ **The structure of the solution is itself a finding.** Balancing mixedness forces **large
single-component fake cells** (2 and 4, ~.13 each) to offset the mixed fake cells (6/7/8), and a
**large cell 5** (.155) as the mixed-real counterweight. Over-weighting 6/7 further is feasible but
consumes presence-head headroom — at `p₆=p₇=.18`, `π_v_pres` reaches 0.779 against the 0.8 bound and
cells 1/3 are squeezed to ~.02. **Do not raise 6/7 without re-running the audit.**

⚠️ This **resolves the open cell-9 question** §7 previously left unanswered. `p₉ = .115`, not a
token share: cell 9 is the release valve for both presence heads under C1.

### C2 — floor on per-head present-count per batch

`_masked_mean` normalises by `mask.sum()`, so a head's gradient norm scales as ~1/√n in the number
of present samples. Measured on our own loss:

| `n_present` (of 32) | ‖grad‖ subset-norm | vs `n=32` |
|---:|---:|---:|
| 2 | 0.3746 | **3.9×** |
| 8 | 0.1894 | 2.0× |
| 32 | 0.0958 | 1.0× |

⚠️ Rare-component batches therefore produce large, high-variance steps. Enforce a minimum
present-count per head per batch (or accept the variance knowingly). ★ This is the concrete form of
E9's *"tune loss, sampler and augmentation as a package."*

---

## 5. The eval sampler is the same code, run once

```
val_specs.parquet  <- sample_spec() with a fixed seed, materialized, committed to the run
```

🔴 **A frozen list of specs, not a frozen seed** — see
[01 §1](01-sample-contract.md#1-the-split-that-everything-else-follows-from). The materialization
step is also where the size floors get checked, because it is the first point they are knowable:

| Assertion | Source |
|---|---|
| ≥1,200 per class **per head pool** per fold | [validation/01 §4](../validation/01-split-scheme.md#4-size-floors) — 95% power on a 1-pt EER gap |
| ≥100 samples per cell per fold | [VG1 A9](../validation/04-audit-gates.md#vg1--split-integrity) |

⚠️ The masked pools are what make the first one bite: the Voice pool is only cells 1,2,5,6,7,8 and
the Music pool only 3,4,5,6,7,8, so sizing by total file count under-fills both. 🔴 **Both
assertions are properties of the eval spec list, not of `folds.parquet`** — a component row has no
cell, so they cannot be evaluated at fold-construction time. [VG1](../validation/04-audit-gates.md#vg1--split-integrity)
A8/A9 therefore run here, against `val_specs.parquet`, while A1–A7 and A10 run against
`folds.parquet`.

SHADOW is drawn the same way with its condition overrides forced on
([validation/01 §2](../validation/01-split-scheme.md#shadow-is-a-condition-axis-not-a-generator-axis)),
and for `shadow_kind='a'` rows the spec is a **copy of the paired VAL spec** with only `normalize`
changed — that is what makes the VAL→SHADOW drop attributable to the channel and nothing else.

---

## 6. Epoch semantics

With unbounded composition there is no natural epoch, but `TrainConfig.epochs` and any LR schedule
need one, and [A-S2](../data/06-augmentation-spec.md) keys the RNG on
`hash(sample_id, epoch, global_seed)`.

> **An epoch is a fixed count of drawn specs**, `steps_per_epoch × batch_size`, with `sample_id`
> the index within it.

Without this, `sample_id` is undefined and reproducibility is nominal rather than real.

---

## 7. The knobs, and what each rests on

| Knob | Start | Evidence |
|---|---|---|
| Cell mix | the **[reference mix](#4--constraints-the-objective-imposes--c1-and-c2)** | Over-weights 6/7 (they are the reason two fake heads exist and cannot be scraped) **subject to C1 and C3** — the naive version of this advice violates the presence-head bound *and* makes mixedness predict fakeness |
| `f` (composed fraction) | — | 🔴 Not a knob. Equal across 5–8 |
| `N_c` domain cap | 500 | DOSS ([papers/05](../papers/05-generalization.md)); cheap to sweep since it is a weight |
| Gain ratio | skewed quiet over ~[−15,+15] dB | ★ G2Net: low-SNR → high-SNR generalizes, not the reverse ([A-A3](../data/06-augmentation-spec.md)) |
| Cell 9 share | **0.115** — ✅ resolved by C1 | It is the *only* cell with both presence labels 0, so it is the release valve for `π_v_pres` and `π_m_pres`. "Small, robustness only" is too small: starving it breaks C1 ([§4](#4--constraints-the-objective-imposes--c1-and-c2)) |
| Folds per experiment | **1** | 5-fold on every experiment is not affordable in the ~10 remaining modelling days ([architecture/08 §4b](../architecture/08-training-recipe.md#4b--the-budget-nobody-costed-engineer-days)). Fold-mean is reserved for promotion decisions ([validation/03](../validation/03-decision-protocol.md)) |

## 8. What is blocked, and what is not

**Not blocked.** The sampler and the fold builder can both be written now against the manifest
schema and a synthetic corpus. VG1 should *fail loudly* when a head has too few artifact families
to support the requested fold count, and emit the music-head variance caveat into the ledger rather
than relying on anyone remembering it.

**Blocked.** 🔴 Trustworthy music-head numbers, on the ≥8-music-family floor
([validation/01](../validation/01-split-scheme.md#-the-music-head-cannot-support-the-planned-split)):
at 5 families a 5-fold puts one family per validation fold and PROBE cannot be carved out at all,
on the highest-weighted component head (0.27).

## 9. One thing we cannot resolve locally

⚠️ **The test set's cell composition is unknown, and it moves 0.45 of the score.** File EER pools
all 1,200 files into one ranking, and EER is not invariant to cell composition within a class.
Two consequences worth stating rather than burying:

- The **training** cell mix is chosen for what the model learns. We cannot match a distribution we
  cannot observe.
- The **eval** cell mix is an *assumption about the test set*, frozen into `val_specs.parquet`. It
  should be recorded as an assumption with a rationale, and is a candidate for sensitivity
  analysis — score a fixed model against two or three frozen eval compositions and check whether
  candidate *ordering* changes. If ordering is stable, the assumption is cheap; if not, it is a
  headline risk.

🔷 We do not believe the [4-submission LB decomposition](../validation/05-lb-probe-plan.md) recovers
composition cheaply, so this is a documented unknown rather than a probe.
