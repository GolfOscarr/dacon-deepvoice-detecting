# Using the metric pipeline

Everything that turns predictions into a number you are allowed to quote.

**The one rule: never compute EER yourself.** Import it. Three details in the official
estimator are load-bearing (`drop_intermediate=False`, FAKE as `pos_label=1`, `argmin` over
`|fpr − fnr|` rather than interpolation), and a reimplementation that "cleans up" any of them
produces a number that silently disagrees with the leaderboard.

Why each rule exists: [`docs/validation/02-metric-harness.md`](../docs/validation/02-metric-harness.md).
This file is how to *use* it.

| Module | Use it for |
|---|---|
| `metrics.dacon` | The official metric. The only place EER is computed |
| `metrics.aggregate` | Combining folds, confidence intervals, comparing two candidates |
| `metrics.breakdown` | Finding *where* a model is weak — per cell, family, fold |
| `metrics.submission` | Writing and validating `submission.csv` |

---

## The frame

Almost everything takes one pandas frame: one row per file, ground truth beside predictions.

| Column | Needed by | Notes |
|---|---|---|
| `voice_present`, `music_present` | all | **Ground truth**, not predictions. These mask the component pools |
| `voice_fake`, `music_fake`, `file_fake` | all | `file_fake` = OR over *present* components — build it with `file_fake_label` |
| the five `*_PROB` columns | all | Names in `PREDICTION_COLUMNS` |
| `cell`, `artifact_family`, `fold` | `breakdown` | Slice keys |
| `file_id`, `pair_id` | `paired_delta_ci`, `t3_gap` | |

0/1 ints, booleans and nullable `Int64` all work; predictions may be float32 or float64.
Row order never matters.

```python
from metrics.dacon import file_fake_label
df["file_fake"] = file_fake_label(df.voice_present, df.music_present,
                                  df.voice_fake, df.music_fake)
```

---

## Recipes

### Score one prediction set

```python
from metrics.dacon import dacon_score

m = dacon_score(df)
print(m.score, m.eer_music, m.n_music)   # Score, one head, and its pool size
m.as_dict()                               # all 8 metrics + the 3 pool sizes
```

`MetricSet` carries `n_file` / `n_voice` / `n_music` deliberately. An EER over 180
music-present files is not the evidence an EER over 2,400 is, and the number alone does not
say which you have.

### Score across folds

```python
from metrics.aggregate import fold_mean
from metrics.dacon import dacon_score

agg = fold_mean([dacon_score(f) for f in fold_frames],
                min_pool=1200, fold_ids=[0, 1, 2, 3, 4])
agg.mean.score, agg.score_sd, agg.excluded_folds
```

`min_pool` drops folds below the size floor and *records* which — averaging in a fold whose
masked pool is too small adds noise, not information.

❌ **Never concatenate raw out-of-fold scores.** Each fold is scored by a different model, EER
is computed on the merged ranking, so pooling measures the score-scale drift between folds.
Measured: **0.1705 against a true 0.100**, where the mean of per-fold EERs gives 0.1040. If you
genuinely need one ranked list for error analysis, use `pooled_oof_scores`, which
rank-normalises per fold first.

### Compare two candidates — the promotion test

```python
from metrics.aggregate import paired_delta_ci
from metrics.dacon import eer

def metric(d):
    return eer(d.file_fake.to_numpy(), d.FILE_FAKE_PROB.to_numpy())

mean, lo, hi = paired_delta_ci(incumbent_df, candidate_df, metric,
                               group_col="artifact_family")
promoted = hi < 0        # EER dropped and the interval excludes 0
```

Both frames must carry the same files in the same order — the function enforces it, because the
pairing is the whole point. At our validation size an unpaired comparison resolves a 1-point
gain 85% of the time; paired, 95%.

⚠️ `group_col` is not decoration. Bootstrapping individual files treats 40 clips from one artist
as 40 independent observations and yields intervals that are far too narrow.

This is criterion **P1** only. A promotion also needs P2–P6 — see
[`03-decision-protocol.md`](../docs/validation/03-decision-protocol.md#3-the-promotion-rule).

### Find where you are weak

```python
from metrics.breakdown import by, breakdown_table, worst_cell_eer, t3_gap

by(df, "cell")                      # one head, one key, sorted worst-first
by(df, "artifact_family", head="music")
breakdown_table(df)                 # every (key, head) pair, for the ledger

worst_cell_eer(df)                  # P2 input
t3_gap(df, head="voice")            # VG4: dict with passes_vg4
```

🔴 **`cell` and `artifact_family` determine the label** — cell 6 is fake by definition, and a
generator emits only fakes. A within-slice EER on either is single-class and undefined. `by()`
detects this and scores those slices against a **shared contrast pool** (the slice plus every
opposite-class row in the same masked pool), recording `contrast="shared"` on each row. Keys
that contain both classes (`fold`, SNR bucket, duration bucket) are scored `"within"` as usual.
Never mix the two kinds in one comparison — check the `contrast` column.

Slices under `min_n` are kept but flagged `thin=True`, and excluded from `worst_cell_eer` /
`worst_family_eer`. Read thin rows with suspicion: exact ties in the ROC grid are a 3.5%
phenomenon at small n and can move EER by a lot, though they are absent at full pool sizes.

### Write a submission

```python
from metrics.submission import read_sample_submission, write_submission, validate_submission

ids, cols = read_sample_submission("data/sample_submission.csv")
write_submission(ids, preds, "output/submission.csv", columns=cols)
validate_submission("output/submission.csv", reference_ids=ids, reference_columns=cols)
```

**Always take `columns` from `read_sample_submission`.** `PREDICTION_COLUMNS` is a documented
default, not ground truth — we have not seen the real `sample_submission.csv`, and the order was
already wrong once.

`write_submission` validates by default and raises `VG5Error` rather than writing a bad file.
`validate_submission` re-reads with `float_precision="round_trip"`; pandas' default parser is not
correctly rounded and perturbs about a third of float64 values.

### The P0 probe

The constant-0.5 submission must score **exactly 0.5000**. It is deliberately degenerate, so it
is the one case where the output gate must be switched off:

```python
import numpy as np
from metrics.submission import read_sample_submission, write_submission

ids, cols = read_sample_submission("data/sample_submission.csv")
write_submission(ids, {c: np.full(len(ids), 0.5) for c in cols},
                 "output/submission.csv", columns=cols, validate=False)
```

Forgetting `validate=False` here means the first planned submission cannot be written at all.

### Decompose the leaderboard

Hold four heads constant, submit, and invert. Weights live in `SCORE_WEIGHTS`.

```python
from metrics.dacon import SCORE_WEIGHTS as W, roll_up

eer_file  = 0.5 - (lb_file_probe  - 0.5) / W["eer_file"]
eer_voice = 0.5 - (lb_voice_probe - 0.5) / W["eer_voice"]
eer_music = 0.5 - (lb_music_probe - 0.5) / W["eer_music"]

ads = roll_up(eer_file, eer_voice, eer_music, 0.5, 0.5)[0]
cps = (lb_full_run - 0.9 * ads) / 0.1          # = 0.5 * (AUC_vp + AUC_mp)
```

Four submissions recover all four quantities; verified exact to ~1e-16 end to end. Full plan and
its noise floor: [`05-lb-probe-plan.md`](../docs/validation/05-lb-probe-plan.md).

---

## Rules, with the consequence of breaking them

| Rule | If you break it |
|---|---|
| ❌ No cross-file statistics in `script.py` | [Rule 2.4](../docs/competition/04-rules.md) violation. Rank normalising a column is explicitly named as forbidden — it is fine locally, never in the submission |
| ❌ Never round or clip predictions on the way out | Saturation reaching the operating point took a measured EER from 0.0950 to **0.3017**, with no error raised |
| ❌ Never pool raw OOF scores across folds | 0.1705 for a true 0.100 |
| ⚠️ Freeze the evaluation composition (`eval_seed`) | EER is invariant to class prevalence but **not** to cell composition within a class: 0.034 → 0.297 with the model unchanged. Two runs with different compositions are not comparable |
| ⚠️ Record `scheme_version` and `corpus_version` with every number | Results across versions are not comparable and must not be averaged |

---

## Errors you will hit

| Error | Means |
|---|---|
| `EmptyPoolError: pool is empty` | A masked pool has no rows — usually a fold with no music-present files. It raises rather than returning 0.5, because 0.5 is a plausible-looking wrong answer that would average into the headline number |
| `EmptyPoolError: single class` | The pool is all-fake or all-real. Undefined, not zero |
| `VG5Error: only N distinct values` | Ranking resolution collapsed. Look for a float32 sigmoid saturating, rounding, or clipping in the export path |
| `VG5Error: constant at 0.5` | Every row is a fallback — the model never ran. This is the failure that scores exactly 0.5000 and looks like a valid submission |
| `ValueError: only N/10000 resamples were scorable` | The pool is too small or too group-imbalanced for a bootstrap interval |
| `ValueError: frames must carry the same files` | `paired_delta_ci` got unaligned frames; the comparison would not be paired |

---

## Verifying

```bash
.venv/bin/python -m pytest                   # known-answer tests
.venv/bin/python scripts/verify_metric.py    # property simulations, printed
```

`scripts/verify_metric.py` imports the shipped `eer` rather than restating it, so it doubles as a
cross-check of this package against the transcribed official definition.

`tests/test_agents_doc.py` executes every Python snippet on this page in an isolated namespace.
If you edit a recipe here, it has to still run — and it has to be self-contained, imports
included.

⚠️ scikit-learn is pinned to the server's **1.8.0** because the metric depends on `roc_curve`'s
tie handling. numpy, pandas and Python are **not** yet matched to the server (3.12 locally against
its 3.11.15); close that before the first real submission.
