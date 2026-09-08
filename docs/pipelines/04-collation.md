# 04 — Collation

`list[RenderedSample] -> batch`, and the decisions the batch layout forces.

---

## 1. Whole-file, not tiled

`SegmentationConfig.mode` defaults to `whole_file`, and the pipeline is built for that alone.
[architecture/04 §6](../architecture/04-heads-and-pooling.md#6-temporal-coverage-tiling-not-sampling)
leaves it 🔷 open on modelling grounds; three facts on the code side close it for the pipeline:

| | |
|---|---|
| **The loss has no window axis** | [`models.losses.multitask_loss`](../../models/AGENTS.md) consumes `clip_logits` and `frame_logits` per branch. Tiling needs either per-window labels — which do not exist, since a window of a fake file may itself be real — or `aggregate_windows` threaded into the loss, which imports the measured duration bias into the *training* signal |
| **Duration bias is measured** | Spread over W=1…12 windows: `max` **1.63** · top-k mean **1.18** · quantile **1.08** · `mean` 0.004 ([architecture/04 §6.1](../architecture/04-heads-and-pooling.md#61--cross-window-aggregation-and-the-duration-trap)). Whole-file deletes the mechanism instead of tuning it |
| **It is the native form of an SED head** | Attention pooling already spans the file |

Cost is ~1.6× a non-overlapping tiling by the doc's own FLOP arithmetic. ⚠️ `tiling` stays
reachable by config for the ablation and as the fallback if the runtime measurement
([09 A3](../architecture/09-open-questions.md)) blows the 3.0 s/file budget — but the collator is
not built for it until then.

### 🔴 Length-regime parity

> Training draws its durations from the **test** distribution, `U(4, 60)` s. It does not train on
> cheap short crops and infer on whole files.

⚠️ This is not caution, it is a repeat-offence guard. `tanh` on the attention logits was correct at
the source notebook's T≈250 and turned `clip_logits` into a **mean pool** at our T≈3000 — the exact
failure the SED head exists to prevent — and `PROGRESS.md` records that "our own `whole_file` choice
made it ~12× worse". A train/infer length mismatch is the same bug class, and it would be invisible
to any test that runs both at the same length.

`AudioConfig.min_seconds` / `max_seconds` (4.0 / 60.0) are the authority on the range.

---

## 2. The batch

```python
{
  "wav":        Tensor,   # (B, C, S_max) float32, zero-padded
  "lengths":    Tensor,   # (B,) int64 -- valid samples per row
  "targets":    dict[str, Tensor],   # 5 keys, each (B,) float32
  "frame_intervals": list[dict[str, tuple[tuple[float, float, int], ...]]],
  "specs":      list[SampleSpec],
}
```

| Item | Contract |
|---|---|
| `wav` | **`(B, C, S_max)`, not mono.** [`prepare_waveform`](../../models/AGENTS.md) applies `AudioConfig.channels` downstream, at the same call site as inference |
| `lengths` | Mandatory, never inferred. It is what makes the frame mask correct; without it every padded frame counts as real audio |
| Padding | Zeros. ⚠️ The value must not matter — the "two pad fillings, same score" test is the rule-2.4 guard and stays |
| `targets` | Exactly the five values of [`losses.TARGET_FOR_COLUMN`](../../models/AGENTS.md): `voice_fake`, `music_fake`, `file_fake`, `voice_present`, `music_present` |
| `frame_intervals` | 🔴 **Absolute time, rasterized late.** See [01 §4](01-sample-contract.md#4-renderedsample) |
| `specs` | Carried for the ledger and for the spec-level audits in [05](05-invariants.md) |

Component-loss masking is *not* the collator's job — `multitask_loss` reads `BranchConfig.masked_by`
and takes the mask from `targets["voice_present"]` / `targets["music_present"]` itself.

---

## 3. Duration bucketing, and the interaction it creates

Whole-file batching over a 15× duration span wastes most of the batch on padding unless rows are
bucketed by duration. But:

✅ **This interaction has been resolved away, not solved.** An earlier version of this section
warned that `pairwise_ranking_loss` forms positive/negative pairs *within the batch*, so tight
duration buckets would teach the model to rank only among files of similar length while the
leaderboard ranks all 1,200 files in one pooled ranking — making batch composition part of the
objective.

🔴 **`LossConfig.ranking_weight` is now committed at 0 and stage S4 is dropped**
([training/03 §1](../training/03-ruled-out.md#1-pairwise-ranking-loss---weight-stays-0)): TFPARN's
own ablation moves EER **12.91 → 12.92**, and its gains are entirely in minDCF/Cllr/actDCF, which a
ranking metric cannot read. With no in-batch pairwise term, **no loss term is sensitive to batch
composition**, and bucketing may be chosen purely for padding efficiency.

🔷 **Bucket freely — 3–4 buckets, or as many as throughput wants.** Two caveats survive:

- ⚠️ If the one-off `ranking_weight` ablation in
  [training/06 §4](../training/06-tier-list.md#4--ablations-are-scored--but-the-rejections-are-already-the-artifact)
  is run, it must run **unbucketed or loosely bucketed**, or it will measure the bucketing rather
  than the loss.
- 🔴 Bucketing must not fight **C2** ([02 §4](02-sampler.md#4--constraints-the-objective-imposes--c1-and-c2)):
  a duration bucket is still a batch, and it must meet the per-head present-count floor. ✅ It does,
  and for a structural reason rather than a lucky one — **`sample_spec` draws `duration_s` first,
  from the test distribution `U(4, 60)`, then fits the components into that timeline.** Length
  cannot depend on the cell because the cell is not drawn yet, so a duration bucket cannot act as a
  cell filter. Measured over 2,000 specs in four quantile buckets: voice presence
  0.682 / 0.678 / 0.676 / 0.694, music presence 0.656 / 0.678 / 0.674 / 0.688.
  ⚠️ This corrects an earlier claim here that "sequential compositions run long", which is false
  against the shipped sampler: a sequential draw *reuses* the drawn duration rather than
  concatenating two, so its mean length is 30.85 s against overlap's 30.38 s. Keep the C2 check as
  cheap insurance — and run it on the **plan**, `audit_specs(specs, batches=training_batches(...))`,
  since a `batch_size` alone cuts the stream in draw order and bucketing is what changed that.

⚠️ Bucketing also interacts with the duration-vs-score check that
[architecture/04 §6.1](../architecture/04-heads-and-pooling.md#61--cross-window-aggregation-and-the-duration-trap)
requires on the REAL class — measure that on unbucketed eval batches, not on training batches. The
frozen eval spec list is unbucketed by construction, so this is free.

---

## 4. Precision

`TrainConfig.precision` defaults to `bf16`; inference ships **fp16**
([architecture/07](../architecture/07-runtime-budget.md)). ⚠️ They are not the same numerical
regime, and this repo has already been bitten: GeM overflowed in fp16 above a feature scale of ~40,
producing NaN attention ([`PROGRESS.md`](../../PROGRESS.md)). The pipeline emits **float32** audio
and lets the model cast; it does not pre-cast to the training precision, so the same batch can be
replayed through an fp16 inference path to check the two agree.
