# Processing strategy

How the EDA's findings become the data-processing decisions the pipeline runs on — config values in
`configs/run_*.yaml`, columns in the manifest, and the harness that proves each decision closed the
shortcut it targets.

Inputs are the EDA's artifacts ([EDA/RESULTS_FOR_ANALYSIS](../EDA/RESULTS_FOR_ANALYSIS.md) is the
map, [EDA/data_memo](../EDA/data_memo.md) the conclusions). The catalog of methods this draws on is
[data/10](../data/10-preprocessing-and-filtering.md) (preprocess / filter / salvage, gates G1–G8)
and [data/06](../data/06-augmentation-spec.md) (the augmentation tiers).

| File | Contents |
|---|---|
| [01-analysis-plan.md](01-analysis-plan.md) | 🔴 **The analysis plan** — the stream harness (measure what the model sees, not the raw corpus), seven decision threads, the per-slice strategies, the verification that accepts the result |
| [02-analysis-report.md](02-analysis-report.md) | 🔴 **The analysis report** — the stream harness measured: the draw manufactures the music shortcut (0.995), the crop that removes it, why `f8 = 0` leaks on the component heads, the A6b correction, and the decision table (§13) |
| [03-processing-and-feature-strategy.md](03-processing-and-feature-strategy.md) | 🔴 **The implementation spec** — the decision register (D-1…D-20), the pipeline map with stage ids, one uniform spec block per stage, the manifest and config contracts (`run_v1.yaml` draft), the ordered change list with its tests, the acceptance checklist |
| [04-verification-report.md](04-verification-report.md) | **The verification report** (2026-09-24) — the seven acceptance items of 03 §7 measured on the built `strategy-v1` artifacts, the `f8` sweep, the shipped-audio residue audit, the G-EDA3/4/7 gates and the G4 / G7 tables, and the four issues that need the owner's decision |
| `scripts/strategy/` | `stream_harness.py` (the draw-level audit and policy simulations) · `shipped_residues.py` (the same audit on the shipped sample) · `verify_reproducibility.py` (I10 / I13 / I14 on real specs) · `scalar_threads.py` (pair tests, DC/high-pass, vectors, families, manifest hours, codec) — outputs in `eda/out/_strategy/` |

Confidence marks as in [docs/README](../README.md): 🔷 marks a code reading not yet measured.
