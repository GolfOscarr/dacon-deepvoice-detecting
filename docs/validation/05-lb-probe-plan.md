# 05 — Leaderboard Decomposition Probe

The leaderboard returns **one scalar** for five heads. Because the score is *linear* in each
head's metric, a small set of submissions recovers all five exactly. This is the only measurement
we can make on the real test distribution, and it is the only legitimate use of the leaderboard
beyond a final score.

## The algebra

Constant columns contribute exactly their neutral value ([02 §4](02-metric-harness.md#an-all-constant-submission-scores-exactly-05000-)),
so holding four heads constant isolates the fifth:

```
Score = 0.5 + w · Δ        where  Δ = (0.5 − EER)   for the three fake heads
                                  Δ = (AUC − 0.5)   for the two presence heads
```

| Head | w | Recovery |
|---|---:|---|
| File | 0.45 | `EER_file  = 0.5 − (S − 0.5)/0.45` |
| Music | 0.27 | `EER_music = 0.5 − (S − 0.5)/0.27` |
| Voice | 0.18 | `EER_voice = 0.5 − (S − 0.5)/0.18` |
| Voice presence | 0.05 | `AUC_vp = 0.5 + (S − 0.5)/0.05` |
| Music presence | 0.05 | `AUC_mp = 0.5 + (S − 0.5)/0.05` |

Verified exact to 1e-12 in `scripts/verify_metric.py` (test T10).

## The plan — 4 marginal submissions

The full-model submission is one we make anyway, so it is not charged to this budget.

| # | Submission | Columns | Yields |
|---|---|---|---|
| **P0** | all-constant `0.5` | — | **Sanity: must return exactly 0.5000** |
| **P1** | model File, constant rest | `FILE_FAKE_PROB` | `EER_file` |
| **P2** | model Music, constant rest | `MUSIC_FAKE_PROB` | `EER_music` |
| **P3** | model Voice, constant rest | `VOICE_FAKE_PROB` | `EER_voice` |
| **F** | full model *(made anyway)* | all 5 | `CPS = (S_full − 0.9·ADS)/0.1` |

Because `ADS` is fully determined by the three recovered EERs, **F gives the presence contribution
for free** — no fourth or fifth probe needed. `CPS = 0.5·(AUC_vp + AUC_mp)`, and since both
presence heads carry the *same* 0.05 weight, the sum is all the decision-relevant information.
Separating them costs one more submission and is only worth it if `CPS` comes back below ~0.97.

**Cost: 4 submissions = 2 days** at 3/day. Verified end-to-end on a synthetic model
(`scripts/verify_metric.py`): recovered `0.132 / 0.181 / 0.294 / CPS 0.967` against ground truth
`0.132 / 0.181 / 0.294 / 0.967`.

This is **minimal** — there are four unknowns (`EER_file`, `EER_voice`, `EER_music`, `CPS`) and
each submission returns exactly one scalar, so four equations are needed and P1–P3 + F supply
exactly four. No cleverer encoding exists.

➕ Side benefit: P1–P3 each run the real model over all 1,200 files, so every probe is also a free
measurement of the **runtime budget** on the actual evaluation hardware — the number
[P5](03-decision-protocol.md#3-the-promotion-rule) needs and that we can only estimate locally.

### P0 is the highest-value single submission in the competition

It costs one slot and validates, at once: the zip structure, the offline packaging, the runtime
budget, the ID/column contract, and our reading of the metric. If it returns anything other than
0.5000, **stop** — something in
[competition/02](../competition/02-submission.md) or
[competition/03](../competition/03-evaluation.md) is misunderstood, and every local number is
built on that misunderstanding. Run it with the trivial model in the `submit.zip` skeleton,
before any real modeling.

## What it can and cannot tell us

**Can**: which head is actually weak on the real distribution; whether the
[16 kHz music premise](../data/07-eda-plan.md) survives contact with the test set; whether our
VAL is optimistic or pessimistic *per head*, which a single aggregate score can never show.

**Cannot**: anything per-cell, per-generator, or per-condition. The test labels stay hidden. It
tells us *where* we are weak, never *why* — that stays a VAL question.

⚠️ The recovery assumes the model produces the **same column** in the probe and in the full
submission. Inference nondeterminism (GPU reduction order, any sampling in the pipeline) breaks
the algebra slightly. Fix seeds, set deterministic kernels where cheap, and treat a recovered
value that disagrees with `F` by more than ~0.002 as a determinism bug rather than a finding.

## The noise floor 🔴

The recovery itself is exact. **The test set is not.** 1,200 files is a sample, and the EER
measured on it is an estimate of the population EER with real error:

| Head | Pool | Files (if the pool is ~50% of the test set) | 95% CI on EER=0.10 |
|---|---|---:|---:|
| File | all 1,200 | 600 / class | **±1.7 pts** |
| Music | music-present only | ~300 / class | **±2.5 pts** |
| Voice | voice-present only | ~300 / class | **±2.5 pts** |

(From the simulation table in [02 §4](02-metric-harness.md#eer-sampling-noise--the-real-precision-limit).
The component pools are smaller than the file pool by exactly the masking, so the two heads we
most want to measure are the two measured worst.)

Two consequences:

1. **Display precision is not the constraint.** Even at 5 decimal places the LB resolves a metric
   to ~1e-4 — four orders of magnitude finer than the sampling noise. Do not reason about rounding.
2. 🔴 **A sub-1-point LB movement is not evidence.** Comparing two models on the same 1,200 files
   is a *paired* comparison and is tighter than the marginal CI, but the population estimate is
   not, and chasing small LB deltas is exactly how a 1,200-sample public set with **Private =
   Public** gets overfit ([competition/03](../competition/03-evaluation.md#3-public-vs-private)).

## Submission budget for the whole competition

3/day × ~22 days ≈ 66 slots. Pre-committed allocation:

| Purpose | Slots |
|---|---|
| P0 sanity + skeleton validation | 2 (one retry) |
| Decomposition round 1 — after the first end-to-end model | 3 |
| Decomposition round 2 — after corpus freeze ([G8](../data/10-preprocessing-and-filtering.md)) | 3 |
| Runtime / packaging validation of real candidates | ~6 |
| Final candidate submissions | ~6 |
| **Reserve** | the rest, deliberately **unspent** |

Leaving slots unused is the intended outcome. ★ E1: OOF decides; the LB is a cross-check. And the
[2nd-stage rubric](../competition/03-evaluation.md#5-second-stage-judging-종합-평가-100-points)
weights the leaderboard at only **30 of 100** points, with the ratio taken over the top-15
finalists — so a compressed LB gap costs far less than a weak
**결과 해석 및 일반화** section, which is written from VAL evidence, not from LB scores.

❌ Two rules bound this, recorded so they aren't reconsidered:

- **[Rule 2.3](../competition/04-rules.md)** — no additional training, tuning or pseudo-labeling
  on the eval set. So probe results may inform *which head to work on*; they may never select or
  label training data.
- **[Rule 2.4](../competition/04-rules.md)** — each test file must be predicted independently.
  Cross-file normalization, batch statistics over the test set, and **rank calibration across the
  cohort are named explicitly as forbidden**. A probe submission is still a submission: its
  constant columns are fine (a constant is not a cohort statistic), but nothing in `script.py`
  may look at more than one file at a time.

⚠️ Rule 2.4 also rules out the obvious fix for score saturation
([02 §4](02-metric-harness.md#score-saturation-is-the-one-output-formatting-risk-that-matters-)).
Tie-freedom in the submission must come from float64 logits, not from rank-normalizing the column.
