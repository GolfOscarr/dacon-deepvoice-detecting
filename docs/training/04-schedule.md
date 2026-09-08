# 04 — The Schedule

Which terms are on when. [02](02-the-loss.md) is the objective; this is the order it is switched on
in, and what each stage is for.

⚠️ This file **revises** [architecture/08 §1](../architecture/08-training-recipe.md#1-the-staged-schedule),
which predates the deep read of this axis. The revisions are re-framings and one deletion; the
stage structure survives.

---

## 1. The stages

| Stage | What trains | Status | Evidence grade |
|---|---|---|---|
| **S1 — independent** | each branch alone, frozen frontends + adapters | keep | ☆ both component papers do it |
| **S2 — joint** | all branches, component losses masked | keep | ⚠️ **re-framed — §2** |
| **S3 — codec-aware** | same, over 4-way codec variants | keep — **strongest stage** | ★ §3 |
| ~~**S4 — ranking polish**~~ | ~~low LR, pairwise loss, ~2 epochs~~ | 🔴 **DROPPED — §4** | ❌ refuted |

---

## 2. ⚠️ S2's evidence is a thresholded-metric result — keep the stage, drop the claim

[architecture/08 §1](../architecture/08-training-recipe.md#1-the-staged-schedule) calls the joint
stage *"the largest single gain"* in both component papers, citing **PC-Mix ACC 69.40 → 85.12** and
**CompSpoof F1 0.668 → 0.908**. Both numbers are recorded correctly in
[papers/04](../papers/04-component-partial.md). **Both are thresholded metrics, and we are scored on
ranking.**

From the *same* PC-Mix ablation (Table VIII), the EER deltas:

| PC-Mix Table VIII, independent → joint | Thresholded | **Ranking (EER)** |
|---|---|---|
| Env utterance | F1 76.41 → 92.67 (**+16.26**) · ACC 69.36 → 92.17 (**+22.81**) | **3.59 → 3.12** (0.47 pts) |
| Speech utterance | — | **8.72 → 7.86** (0.86 pts) |
| 5-class | ACC 69.40 → 85.12 (**+15.72**) | — |

🔴 **On the metric we are scored on, the joint-training gain is below our ≈1 pt local resolution
threshold**, and far below the ±2.5 pt component-head leaderboard noise.
The headline is 15–23 points because ACC and F1 move far more than EER — the same asymmetry this
repo already recorded from the other direction, where a broadcast study saw F1 fall 0.992 → 0.186
while AUC fell only 0.998 → 0.775 ([survey/02](../survey/02-sota-music.md)).

⚠️ **CompSpoof's 0.668 → 0.908 does not describe our regime at all.** Their configurations are:
direct 5-class baseline **0.827**, separation *without* joint learning **0.668**, separation *plus*
joint learning **0.908**. The quoted delta is the effect of adding joint learning **inside a
separation-based system**, and [03 G](../architecture/03-candidates.md) rejects separation on four
grounds. Against the no-separation baseline the whole system gains 0.827 → 0.908, and the
joint-training gain *in our architecture* is not measured by that paper.

**Verdict: keep S2, on different grounds.** It is our architecture anyway — candidate A is already
jointly trained ([03](../architecture/03-candidates.md#-a-is-stage-one-of-b-not-a-fallback)) — the
EER deltas are positive, and the stage is nearly free. But it moves from the
**measure-it** column to the **decide-by-argument** column: we cannot resolve a 0.5-point effect
we cannot resolve locally, and should not spend a fold-sweep pretending otherwise.

---

## 3. ★ S3 is the best-evidenced stage in the whole recipe

Unchanged, and worth stating plainly now that the others have thinned out. ★ ArtifactNet P2→P3:
hard-negative FPR **98.7% → 8.0%**, cross-codec drift **−83%**
([papers/02](../papers/02-music-detection.md)). It is a **training schedule, not an architecture**,
so it is adopted regardless of how the `E-A1` probe resolves.

⚠️ Two independent reasons it matters more for us than the numbers alone suggest:

- Our audio is MP3/WAV/FLAC at 16 kHz with a **telephone-channel slice**. Codec variance is a
  certainty, not a hypothesis.
- ★ The hardest condition in the one genuinely-unseen benchmark we found — ASVspoof 5 — is
  **codec-10: speex, 8 kHz bandwidth, low bitrate** ([03 §3](03-ruled-out.md#3-one-class-and-margin-losses-oc-softmax-am-softmax-samo---declined)).
  That is our telephone slice, named as the worst case by an independent evaluation.

🔷 Given that every objective-side knob measured below our noise floor, **S3 is where the
schedule's remaining value is concentrated.** Budget accordingly.

---

## 4. 🔴 S4 is dropped

[architecture/08 §1](../architecture/08-training-recipe.md#1-the-staged-schedule) scoped S4 as "low
LR, pairwise ranking loss, ~2 epochs", worth ☆ ~1 bps. Both of its justifications fail:

| Justification | Status |
|---|---|
| ★ TFPARN — "combines focal + pairwise, aimed at EER/threshold metrics" | ❌ **Refuted by its own ablation**: pairwise moves EER 12.91 → **12.92**. All its gains are minDCF/Cllr/actDCF ([03 §1](03-ruled-out.md#1-pairwise-ranking-loss---weight-stays-0)) |
| ☆ G2Net F4 — "~1 bps" rank-loss fine-tune | ⚠️ Below our ≈1 pt local resolution threshold. Unresolvable, and unquotable as evidence |

**Dropping S4 frees engineer-days rather than costing them**, which matters against
[architecture/08 §4b](../architecture/08-training-recipe.md#4b--the-budget-nobody-costed-engineer-days)'s
~10 remaining modelling days. Replace it with sampler constraint **C1**
([02 §8](02-the-loss.md#8-what-the-sampler-must-guarantee)) — removing the imbalance pathology costs
no hyperparameters and cannot be washed out by noise.

---

## 5. Warm-up staging — not adopted

★ Zhang et al. 2021 Table 6 reports **UttBW 5.66** against **MulBS 5.90**, which would argue for
pre-training one level then bolting on the other. ❌ **The comparison is contaminated**, from §5.2 of
that paper: the warm-up models used *"the best pre-trained model in the development set"* in the six
rounds, plus hand-pinned seeds `10⁶` and `10⁴` **outside** the standard `10⁰–10⁵` range, while MulBS
did not. Its corroborating signature is SegBW holding the **best dev** utterance EER (2.53) and the
**worst** binary-branch eval (6.07) — a selection artifact.

0.24 points is 14% of our File-head floor regardless. **Not adopted.**

---

## 6. The experiments that are actually worth running

⚠️ Only one objective-side question clears the noise floor. Everything else in
[02](02-the-loss.md) is committed by argument.

| # | Experiment | Question | Speed | Decides |
|---|---|---|---|---|
| **T1** | 🔴 `clip_weight` **1.0 vs 0.5** | Does our frame loss help or hurt? Nobody has measured whether frame supervision improves file-level *ranking* of short fake components | Medium | `clip_weight`, and whether MulBS is worth building at all |
| **T2** | Per-head weights, metric-proportional vs uniform | Verify the argued default does not *hurt* | Medium | ⚠️ not expected to resolve; a guardrail, not a decision |
| **T3** | MulBS split | Only if T1's sign says the frame term carries signal | Medium | ❌ do not run first |
| **T4** | `l_soft` β=0.7 on the `reported` tier | One run, last | Replay→Medium | Expect nothing |

🔴 **T1 is the first training experiment.** It is one config value, it is the correct control, and
we currently ship a `clip_weight` chosen by symmetry against evidence that the value may cost up to
3.63 points. ⚠️ Read its **sign** before its size — the effect in our regime is not the effect in
theirs ([02 §3](02-the-loss.md#-four-limits-on-transferring-that-number--read-before-treating-363-as-ours)).

⚠️ **T1 must not run at Replay speed.** [validation/03 §2](../validation/03-decision-protocol.md)
warns that Replay systematically favours ideas that help early in training, and a loss-structure
change is exactly that shape.

---

## 7. Where the schedule sits in the ladder

Unchanged from [architecture/08 §4b](../architecture/08-training-recipe.md#4b--the-budget-nobody-costed-engineer-days),
with S4 removed:

```
P0-a → P0-b → candidate A → candidate B → S3 codec stage → distillation → extras
```

🔷 One re-ordering worth considering, given this file's findings: **S3 has the strongest evidence of
anything left in the recipe**, and the objective work it was queued behind has largely evaporated.
If engineer-days run short, S3 is a better spend than candidate B, whose marginal claim over A rests
on a single citation about frontend specialization.

⚠️ That is a suggestion, not a decision — the drop order is pre-committed in
[architecture/08](../architecture/08-training-recipe.md) and re-ordering it is a call for the
project owner, not a consequence of this file.
