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
both sides of every label.** See [06](06-augmentation-spec.md#the-governing-rule).

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
