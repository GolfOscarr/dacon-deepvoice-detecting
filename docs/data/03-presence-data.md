# 03 — Presence Detector Dataset (Stage 1)

Targets `VOICE_PRESENT_PROB` and `MUSIC_PRESENT_PROB`. Combined metric weight **0.10**,
scored by ROC-AUC. Objective: get this near-saturated cheaply, then stop.

## Core insight: presence labels are free

We never need to *collect* presence-labeled data. We **compose** files from the four pools, so
presence is known by construction:

```
V_present = (any Pool A or B segment was used)
M_present = (any Pool C or D segment was used)
```

This means the presence detector and the fake detectors train on **the same composed corpus** —
one build pipeline, two label groups.

## What actually makes it hard

The label is trivial; robustness is not. Three failure sources:

| Failure | Cause | Mitigation |
|---|---|---|
| **Missed short components** | 3 s of music inside a 60 s file, mean-pooled to nothing | Segment-level scoring + **max/attention pooling**; explicitly generate short-component samples (component occupying 5–20% of duration) |
| **Applause / crowd / rain → "music"** | Broadband rhythmic texture | Pool E negatives with M_present=0, deliberately over-sampled |
| **Low-level components** | Background music at −15 dB under speech | Gain sweep ([02](02-label-taxonomy.md)); log gain per sample and inspect AUC-vs-gain |

## Composition recipe for presence training

| Slice | Share | Content |
|---|---|---|
| Voice only | 20% | Pool A ∪ B, spoken **and sung** |
| Instrumental only | 20% | Pool C ∪ D |
| Mixed, overlapping | 30% | gain ratio ~U(−15, +15) dB |
| Mixed, **sequential** | 20% | 2–4 segments, varied order and gaps |
| Neither / noise-only | 5% | Pool E |
| Hard negatives | 5% | applause, crowd, rain, engine, tonal non-music (alarms, sirens) |

Duration sampled to match the test range **4–60 s**, with a bias toward the short end where
short components are hardest.

## Real vs AI must both appear

🔴 The presence head must fire on **AI-generated** voice and music too. If Pool B (fake voice)
never appears in voice-present training, the head may under-fire on synthetic speech — which is
precisely where the test set is dense. Keep the real/fake ratio within each presence slice near
50/50.

## Sung voice is the trap

A song has V_present=1 **and** M_present=1 (vocals count as voice). If sung vocals are absent
from Pool A, the model learns "melodic → music only" and loses the voice head on every song in
the test set. MUSDB18-HQ vocal stems and Opencpop/M4Singer exist for exactly this.

## Model path

Start with **PANNs CNN14** — preinstalled on the eval server (`panns-inference==0.1.1`), zero
offline-packaging risk, AudioSet mAP 0.439 which is ample for a 0.10-weight ROC-AUC task. Ship a
working presence head in Phase 0. Upgrade to **BEATs / SSLAM** only if LB probing shows presence
is costing points, or if we want the same encoder as the shared trunk / type router
([survey 04](../survey/04-sota-presence.md), [survey 09](../survey/09-challenge-playbooks.md)).

## Acceptance criteria

- ROC-AUC ≥ 0.99 on a **generator-disjoint** and **source-disjoint** validation split, for both heads
- AUC ≥ 0.97 in the worst gain bucket (component at −15 dB)
- AUC ≥ 0.97 for components occupying < 20% of file duration
- Applause/crowd false-positive rate for `MUSIC_PRESENT` < 5% at the operating rank

⚠️ If AUC on a random split is 0.999 but drops sharply on a source-disjoint split, we are
detecting the *corpus*, not the *content*. See [09](09-risks-and-checks.md).
