# Validation Design (Phase C)

The local instrument we trust **instead of** the leaderboard.

| File | Contents |
|---|---|
| [01-split-scheme.md](01-split-scheme.md) | Grouping keys, the four slices, fold construction, size floors |
| [02-metric-harness.md](02-metric-harness.md) | ⭐ **The metric register** — every number we track, and how each is measured |
| [03-decision-protocol.md](03-decision-protocol.md) | Three experiment speeds, promotion rule, experiment ledger |
| [04-audit-gates.md](04-audit-gates.md) | **VG1–VG6** — the automated checks that invalidate a run |
| [05-lb-probe-plan.md](05-lb-probe-plan.md) | Per-head decomposition of the leaderboard, and its noise floor |
| [06-implementation-plan.md](06-implementation-plan.md) | `metrics/dacon.py` — layout, requirements, order of work |

## Why this phase comes before modeling

★ `[BirdCLEF playbook 2026]` order of importance: **`data split → sampler/loss → architecture →
optimizer`**. *"Common mistake: over-searching schedules before solving data shift and imbalance."*
Phase C is design work — it needs no corpus and no model, so it runs in parallel with the
[Phase B build](../data/08-build-plan.md) and must finish first.

The specific trap here: **Private = Public** ([competition/03](../competition/03-evaluation.md#3-public-vs-private)).
The leaderboard is a 1,200-file set, queryable 3×/day for 22 days — enough to overfit — and there
is no held-out private split to punish it. The only defence is a local measurement that is
*harder* than the leaderboard.

## The plan in six lines

1. Split by **artifact family**, source and speaker — never at random. A random split reports
   ~0.99 and teaches nothing ([08 splits](../data/08-build-plan.md#splits--generator-disjoint-source-disjoint)).
2. Four slices: **TRAIN · VAL** (tuning) **· SHADOW** (domain shift) **· PROBE** (sealed).
3. Track exactly the numbers in the **[metric register](02-metric-harness.md#1-the-metric-register)** —
   official, decision, diagnostic, guardrail — and reimplement the official metric exactly,
   masked pools included, in `metrics/dacon.py`.
4. Decide with **paired** comparisons on a fixed VAL, not by eyeballing marginal numbers.
5. Six automated gates (**VG1–VG6**) run on every experiment; failing one voids the result.
6. Spend **4 submissions once** to decompose the leaderboard per head — then stop probing.

## Numbers established here

All verified numerically — reproduce with `.venv/bin/python scripts/verify_metric.py`, which
prints every table in [02](02-metric-harness.md) and [05](05-lb-probe-plan.md):

| Fact | Value |
|---|---|
| An all-constant submission scores | **exactly 0.5000** — any value, any column |
| Recovering one head's metric from the LB | exact linear algebra ([05](05-lb-probe-plan.md#the-algebra)) |
| LB display precision as a limit | **not** the limit — 5 dp already gives 1e-4 metric resolution |
| The real limit on LB decomposition | **sampling noise** on 1,200 files: ±1.7 pts EER on the File head, ±2.5 on each component head |
| VAL size to resolve a 1-point EER gap at 95% | **≈1,200 per class per head** (paired, correlated models) |
| EER under score saturation | harmless until saturation reaches the operating point, then catastrophic (0.095 → **0.302**) |
| Pooling raw OOF scores across folds | **invalid** — reported 0.171 against a true 0.100. Use the mean of per-fold metrics |
| EER invariance | ✅ monotone transforms, ✅ class prevalence, ❌ **cell composition within a class** (0.034 → 0.297) |

## Non-negotiables

- Fold assignment generated **once**, frozen to `folds.parquet`, reused by every experiment (★ E4)
- **Generator-disjoint by artifact family**, not by model name ([09 R4](../data/09-risks-and-checks.md))
- The **PROBE slice is sealed** — a hard budget of 3 openings for the whole competition
- Report **per cell and per generator**, never pooled only — a dead cell at 2% of the corpus shifts
  the pooled EER by ~1 pt, inside the noise band, while the cell itself sits at chance
- Cell and family are **label-determining**, so those slices need a
  [shared contrast pool](02-metric-harness.md#-label-determining-keys-need-a-shared-contrast-pool)
- 🔴 **OOF decides whether an idea survives; the public LB is a weak cross-check only** (★ E1)
- No experiment is quotable without a **VG1–VG6 pass** recorded alongside it
