# 02 — Label Taxonomy: 4 Pools → 8 Cells

## The key simplification

We do **not** collect eight kinds of dataset. We collect **four component pools** and *compose*
everything else. Composition gives us exact labels for free.

```
POOL A: real voice        ┐
POOL B: fake voice        ├─→ compose ─→ 8 labeled cells + sequential variants
POOL C: real instrumental │
POOL D: fake instrumental ┘
POOL E: non-musical sound (noise/environment)  ← robustness only
```

"Voice" includes **spoken and sung** (the rules classify vocals as voice).
"Instrumental" = 반주·악기음 with no vocals (the rules' definition of 음악).

## The 8 cells

Per competition rules: `FILE_FAKE = VOICE_FAKE OR MUSIC_FAKE` over **present** components.
Component labels on absent components are ignored by the metric (masked EER).

| # | Cell | V_present | M_present | V_fake | M_fake | FILE_fake | Built from |
|---|---|:--:|:--:|:--:|:--:|:--:|---|
| 1 | Voice-only REAL | 1 | 0 | 0 | – | **0** | A |
| 2 | Voice-only FAKE | 1 | 0 | 1 | – | **1** | B |
| 3 | Instrumental-only REAL | 0 | 1 | – | 0 | **0** | C |
| 4 | Instrumental-only FAKE | 0 | 1 | – | 1 | **1** | D |
| 5 | Mixed R/R | 1 | 1 | 0 | 0 | **0** | natural songs, **and** A+C composed |
| 6 | Mixed **R/F** | 1 | 1 | 0 | 1 | **1** | A+D composed ⭐ |
| 7 | Mixed **F/R** | 1 | 1 | 1 | 0 | **1** | B+C composed ⭐ |
| 8 | Mixed F/F | 1 | 1 | 1 | 1 | **1** | AI song generators, **and** B+D composed |
| 9 | Neither (noise only) | 0 | 0 | – | – | 0 | E — small, robustness only |

⭐ **Cells 6 and 7 are the reason this competition has two separate fake heads.** They cannot be
scraped and are unlikely to exist in any public dataset. If they are missing from training, the
two fake heads collapse into a single entangled "fakeness" axis and both EERs suffer.

## 🔴 The composition trap

If cell 5 (mixed, both real) consists only of **natural songs** while cells 6/7/8 are all
**our artificial mixes**, then "artificially mixed" perfectly predicts FAKE. The model learns
our mixing pipeline, not AI artifacts. Local CV looks superb; the leaderboard does not move.

**Rule**: cell 5 must contain artificial A+C mixes in similar proportion to how cells 6/7 are
built, using the identical mixing code path. Same for cell 8 — include both natural AI songs and
artificial B+D mixes.

Corollary: **every structural transform (mixing, concatenation, crossfade, gain) must appear on
both sides of every label.** See [06](06-augmentation-spec.md#-the-governing-rule).

### 🔴 The mechanism: one composed fraction, shared across cells

The rule above named no mechanism, so it could not be checked. It has one, and it is a **sampler
parameter, not an acquisition target**:

> Let `f_c` be the fraction of cell-`c` samples rendered from *composed* component rows rather than
> from whole-file rows ([validation/01 §3](../validation/01-split-scheme.md#-the-table-is-keyed-on-components-not-on-composed-files)).
> **`f_c` must be equal across every cell produced on both sides of `FILE_FAKE`.**

Cells 5 and 8 can be rendered either way. Cells 6 and 7 can **only** be composed — that is the
whole reason they cannot be scraped, and the whole reason two fake heads exist. So if `f₅ = 0`
(cell 5 is only natural songs) while `f₆ = f₇ = 1`, "composed" predicts FAKE perfectly. Holding
`f₅ = f₆ = f₇ = f₈` closes it, and it is assertable over the spec stream without decoding audio.

★ **This is now measured, not inferred.** PartialSpoof
([arXiv:2204.05177](https://arxiv.org/abs/2204.05177)) §VI-D / Fig. 5 breaks EER down by number of
concatenation boundaries and finds it **worst — above 14% — at zero boundaries**, attributing it to
overlap-add artifacts around the joins. ⚠️ **Worse for us than for them**: their eval set is also
composed, whereas a genuine AI song or a real phone call with hold music in the DACON test set
carries no splice artifact at all.

🔴 **And the rule covers *mixing*, not only concatenation.** Mixedness is set by the cell, so unlike
the other structural transforms it cannot be fixed in the transform sampler — it must be fixed in
the **cell mix**: `P(mixed | FILE_FAKE=1) = P(mixed | FILE_FAKE=0)` over cells 1–8 (cell 9 excluded,
since a file with no components cannot be fake). A verified reference mix is in
[pipelines/02 §4](../pipelines/02-sampler.md#-c3-covers-mixing-not-only-concatenation--and-that-constrains-the-cell-mix).

### ✅ Decided: label-conditional is primary, strict is the fallback — and they are one knob

Cells 6 and 7 are producible **only** by composition, so `f₆ = f₇ = 1` in practice. Pairwise
equality would then force `f₅ = f₈ = 1` — **no natural song usable as a cell-5 sample, no AI song as
cell-8**. That is safe but discards both pools, and full AI songs (Suno/Udio) are a real slice of the
target distribution.

⚠️ **The marginal form is not enough.** `P(composed | FAKE) = P(composed | REAL)` can be satisfied
while composedness still predicts the label *inside* a presence stratum — e.g. `f₁=1, f₂=0` balances
marginally while "composed" predicts REAL perfectly among voice-only files. **Verified numerically.**

🔴 **The correct form is stratified** — composedness must be label-independent *within each presence
stratum*:

```
voice-only :  f₁ = f₂                       (any common value)
music-only :  f₃ = f₄                       (any common value)
mixed      :  f₅ = (p₆f₆ + p₇f₇ + p₈f₈) / (p₆ + p₇ + p₈)      with f₆ = f₇ = 1
```

⚠️ **Cell 9 is excluded**, for the same reason it is excluded from the mixedness balance: a file with
no components cannot be fake, so the "neither" stratum can never be label-balanced. Excluding it,
the stratified constraint also delivers **marginal** balance — verified to within 0.0014 for any
common single-cell rate.

**`f₈` is the single knob, and the two policies are its endpoints:**

| `f₈` | required `f₅` | natural in cell 5 | AI songs (cell 8) | genuine whole-file mass | policy |
|---|---|---|---|---|---|
| **0.00** | **0.725** | 0.0427 | 0.0950 | **13.8%** | ⭐ **conditional — primary** |
| 0.25 | 0.793 | 0.0320 | 0.0713 | 10.3% | |
| 0.50 | 0.862 | 0.0213 | 0.0475 | 6.9% | |
| 1.00 | 1.000 | 0 | 0 | **0%** | **strict — fallback** |

(masses are corpus fractions under the [reference cell mix](../pipelines/02-sampler.md#the-reference-cell-mix))

🔴 **Decision: run `f₈ = 0` as primary** — all AI songs used as-is and ~27.5% of cell 5 as natural
songs, giving 13.8% genuine whole-file audio. **Fall back to `f₈ = 1` (strict) if the metric
degrades**, which is a one-value change, not a second code path.

### The same rule one level down: components on both sides

🔷 The transform-level rule has a component-level analogue that is cheaper to enforce and catches
a different shortcut — content identity rather than pipeline identity:

> **Every component file should appear on both sides of `FILE_FAKE` wherever its type permits.**

A real voice file supports cells 1 and 5 (`FILE_FAKE=0`) *and* cell 6 (`FILE_FAKE=1`); a real
instrumental supports 3, 5 and 7. Fake components are asymmetric by construction and cannot — a
fake voice file is only ever in a fake file. Enforcing this where it is possible stops the model
from keying on *which* real component it is looking at, and it is a sampler-side invariant, not a
data-acquisition one.

## Mixing gain is a first-class variable

Published finding ([survey 03](../survey/03-sota-singing-mixed.md)): detection accuracy tracks
**stem energy**. Vocals at low relative level are the hard case (65–80% TPR vs 97–98% for
accompaniment). The hybrid-stems paper swept **−12 dB to +12 dB**.

🔴 **Updated after the Kaggle review**: do **not** sample uniformly. ★ `[G2Net 2021, 3rd]` found
models generalize **low-SNR → high-SNR but not the reverse**, and injected at
`SNR ~ max(N(3.6,1), 1)` for a 2–8 bps gain. So **skew the voice/music gain ratio toward the quiet
end** across roughly [−15, +15] dB, and record it per sample so we can report accuracy-vs-SNR
curves ([07 E-B6](07-eda-plan.md), also 2nd-stage 결과 해석 material). See
[06 A-A3](06-augmentation-spec.md).

## Sequential variants (the 순차적 case)

Any cell can be rendered as a **temporal sequence** rather than an overlap:

- `V | M`, `M | V`, `V | M | V`, `M | V | M`, up to ~4 segments
- Segment durations sampled so total lands in **4–60 s** (the test range)
- Crossfade 10–200 ms, plus some hard cuts and some silent gaps
- Presence labels = union over segments; fake labels = OR over segments of that component

⚠️ Sequential rendering must be applied to REAL-REAL pairs too (cell 5), at the same rate.

## Edge cases and their correct labels

| Case | V_pres | M_pres | Note |
|---|:--:|:--:|---|
| A cappella / choir | 1 | 0 | vocals are voice; no accompaniment |
| Instrumental / backing track | 0 | 1 | |
| Full song (vocals + backing) | 1 | 1 | the rules' explicit example of 혼합 |
| Rap, whisper, humming, beatboxing | 1 | ? | human vocal production → voice |
| Speech over background music | 1 | 1 | |
| Telephone call with hold music | 1 | 1 | likely sequential |
| Applause, crowd, traffic, rain | 0 | 0 | ⚠️ classic false positive for "music" |
| Speech + non-music noise | 1 | 0 | |
| Instrumental containing chopped vocal samples | 1? | 1 | ⚠️ genuinely ambiguous — see [09](09-risks-and-checks.md) |
| **Denoised / enhanced / volume-normalized real audio** | – | – | 🔴 **stays REAL** per the rules. Must be represented as REAL in training. |
| Vocoder / neural-codec resynthesis of real audio | – | – | **FAKE** (regenerates the component) |

## Prevalence does not need to match the test set

EER and ROC-AUC are both **rank-based and prevalence-insensitive**. We do not know the test
composition and we do not need to. Balance cells for *learning* (roughly equal exposure per
cell, with cells 6/7 deliberately over-represented since they carry the head-decoupling signal),
not to mimic an unknown prior.
