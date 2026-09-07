# 06 — Implementation Plan: `metrics/dacon.py`

Brief. The first code in the repo, and a dependency of both the experiment loop and the
`submit.zip` skeleton. Target: **one working day**.

## Why this first

- The [LB sanity probe](05-lb-probe-plan.md#p0-is-the-highest-value-single-submission-in-the-competition)
  needs the submission writer, which lives here.
- Every experiment number is meaningless until the metric is verified against the official
  definition.
- It is the one component whose correctness can be established **completely** with no data —
  known-answer tests only.

## Layout

```
metrics/
├── __init__.py
├── dacon.py          # the official metric — the only place EER is computed
├── aggregate.py      # fold aggregation, group bootstrap, CIs
├── breakdown.py      # Tier-3 diagnostic slices
└── submission.py     # write/validate submission.csv, VG5 assertions
tests/
└── test_metrics.py   # T1-T14 from 02 §8
scripts/
└── verify_metric.py  # already present — the property simulations
```

`dacon.py` imports **only** numpy, pandas and scikit-learn — all preinstalled on the eval server
([competition/02](../competition/02-submission.md#preinstalled-packages-do-not-pin-different-versions)) —
so `submission.py` can be vendored into `submit.zip` unchanged.

## Public surface

```python
# dacon.py — Tier 1
def eer(y_true, y_score) -> float
def auc(y_true, y_score) -> float
def score_frame(df: pd.DataFrame) -> MetricSet      # the 8 Tier-1 numbers
def roll_up(eer_file, eer_voice, eer_music, auc_vp, auc_mp) -> tuple[float,float,float]

# aggregate.py — Tier 2
def fold_mean(per_fold: list[MetricSet]) -> MetricSet          # mean + sd, never pooled raw
def group_bootstrap_ci(df, metric_fn, group_col, n=10_000) -> tuple[float,float]
def paired_delta_ci(df_a, df_b, metric_fn, group_col, n=10_000) -> tuple[float,float,float]

# breakdown.py — Tier 3
def by(df, key) -> pd.DataFrame                     # cell | family | fold | snr | duration
def t3_gap(df) -> float

# submission.py
def write_submission(ids, preds, path) -> None      # float64, no rounding, VG5 asserted
def validate_submission(path, reference_ids) -> None
```

`MetricSet` is a frozen dataclass holding the 8 Tier-1 numbers plus `n` per pool, so a downstream
reader can always see what a number was computed over.

## Non-obvious requirements

These are the reasons this is a day's work and not an hour's — each is a finding from
[02](02-metric-harness.md) that the code has to enforce rather than merely respect.

| # | Requirement | From |
|---|---|---|
| 1 | `roc_curve(..., drop_intermediate=False)`, `argmin`, **no interpolation** | [02 §2](02-metric-harness.md#2-the-official-definitions) |
| 2 | Masked pools use **ground-truth** presence, never predictions | [02 §3](02-metric-harness.md#3-the-masked-pools) |
| 3 | `fold_mean` **must not** concatenate raw OOF scores — mean of per-fold metrics only | [02 §4](02-metric-harness.md#4-how-we-aggregate) |
| 4 | Bootstrap resamples `artifact_family`, and takes the paired difference **inside** each resample | [02 §6](02-metric-harness.md#6-confidence-intervals) |
| 5 | Every returned metric carries the `eval_seed` and `corpus_version` it was computed under | [02 §5](02-metric-harness.md#-not-invariant-to-composition-within-a-class-) |
| 6 | Empty or single-class pool **raises**; never silently returns 0.5 | T13/T14 |
| 7 | `write_submission` emits float64 with no rounding or clipping, and asserts VG5 | [02 §7](02-metric-harness.md#score-saturation-is-the-one-output-formatting-risk-that-matters-) |
| 8 | ❌ No cross-file statistic anywhere in `submission.py` — no rank normalization | [rule 2.4](../competition/04-rules.md) |

Requirement 6 deserves the emphasis: a masked pool goes empty exactly when a fold has no
music-present files, and `EER = 0.5` is a *plausible-looking* wrong answer that would quietly
average into the headline number.

## Order of work

| Step | Deliverable | Done when |
|---|---|---|
| 1 | `dacon.py` + T1–T10, T13, T14 | Tests green; `verify_metric.py` still passes against the imported `eer` |
| 2 | `submission.py` + `validate_submission` | Round-trips DACON's `sample_submission.csv` with the correct ID set |
| 3 | **`submit.zip` skeleton** — constant-0.5 `script.py` | Packaged offline, runs on the 3 dummy files |
| 4 | 🔴 **Submit P0** | Leaderboard returns **exactly 0.5000** |
| 5 | `aggregate.py` + T11, T12 | Fold-rescale test reproduces mean-of-folds 0.1040 vs pooled-raw 0.1705 |
| 6 | `breakdown.py` | Emits the full Tier-3 table from a prediction frame |

Steps 1–4 are the critical path and are worth doing before any modeling. Steps 5–6 are needed
before the first *comparison* between two models, which is later.

## Definition of done

- [ ] All 14 tests green under scikit-learn **1.8.0**, not just 1.9.0 ([02 §9](02-metric-harness.md#9-version-pinning))
- [ ] `scripts/verify_metric.py` imports `metrics.dacon.eer` rather than redefining it
- [ ] `score_frame` reproduces a hand-computed 5-row fixture
- [ ] P0 returned exactly 0.5000 on the real leaderboard
- [ ] No function in `metrics/` reads more than one file's row when called from `script.py`

## Deferred

Not part of this module, recorded so the boundary stays clean: the fold builder
(`folds.parquet`, [01 §3](01-split-scheme.md#3-fold-construction)) and the gate implementations
([04](04-audit-gates.md)) are separate work — they *consume* `metrics/`, and VG2/VG3 need a corpus
that does not exist yet.
