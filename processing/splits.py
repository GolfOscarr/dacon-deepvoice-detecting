"""OFF-5 -- `folds.parquet` over the built manifest (docs/processing/03 §3).

The builder and the checker are ``training.folds`` / ``training.foldcheck``,
called with the ``folds:`` section of ``configs/processing_v1.yaml``. What
this module adds is the outcome as an artifact: the table, the builder's
caveats and the VG1 report written beside the manifest, and the refusal to
write a table that fails VG1.

Measured on the built corpus (step 10): the music head has five composable
fake families (FakeMusicCaps) plus SONICS' two whole-file-only ones, so a
5-fold with a music PROBE is infeasible -- four folds is what the corpus
supports (D-22). Three fixes to ``training.folds._seal_probe`` came out of
the same run: composable groups are sealed before whole-file-only ones, a
seal never leaves a head below ``n_folds`` composable rotating families, and
a family-advancing seal has a row budget (``PROBE_ROW_BUDGET``) so the
LJSpeech pair atom (32 % of the corpus) rotates instead of being sealed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from training.audit import AuditReport
from training.foldcheck import check_split_integrity
from training.folds import FoldConfig, FoldPlan, build_folds

__all__ = ["build_and_check", "write_outputs"]


def build_and_check(manifest: pd.DataFrame, cfg: FoldConfig, shadow_b: list[str] | None = None
                    ) -> tuple[FoldPlan, AuditReport, dict[str, Any]]:
    """Build the folds, run VG1 A1-A7 over them, and summarise. Raises if VG1
    fails: a fold table the checker refuses is not a fold table."""
    plan = build_folds(manifest, cfg, shadow_b=shadow_b)
    report = check_split_integrity(plan.frame)
    if not report.ok:
        raise AssertionError(f"folds fail VG1:\n{report}")
    f = plan.frame
    tv = f[f["slice"] == "train_val"]
    fake = f["artifact_family"].notna()
    by_fold = tv.groupby("fold")
    summary = {
        "n_folds": int(cfg.n_folds),
        "rows": {"train_val": int(len(tv)), "probe": int((f["slice"] == "probe").sum()),
                 "shadow": int((f["slice"] == "shadow").sum())},
        "rows_per_fold": {int(k): int(v) for k, v in by_fold.size().items()},
        "families_per_fold": {int(k): int(v) for k, v in
                              by_fold["artifact_family"].nunique().items()},
        "probe_families": int(f.loc[(f["slice"] == "probe") & fake, "artifact_family"].nunique()),
        "probe_row_share": round(float((f["slice"] == "probe").mean()), 4),
        "caveats": list(plan.caveats),
    }
    # drawable hours per (slice/fold, role, fake) -- the numbers 06 D1/D4 bind on
    comp = manifest[manifest["row_kind"] == "component"].set_index("file_id")
    role_of = {"A": ("voice", "real"), "B": ("voice", "fake"), "C": ("music", "real"),
               "D": ("music", "fake"), "E": ("noise", "real")}
    fk = f.set_index("file_id")
    hours: dict[str, dict[str, float]] = {}
    atoms: dict[str, dict[str, int]] = {}
    for fid, row in comp.iterrows():
        pool = row["pool"]
        if pool not in role_of:
            continue
        where = "probe" if fk.at[fid, "slice"] == "probe" else f"val{int(fk.at[fid, 'fold'])}"
        key = " ".join(role_of[pool][::-1])
        bucket = hours.setdefault(where, {})
        bucket[key] = bucket.get(key, 0.0) + row["duration_s"] / 3600
        if where == "probe":
            atoms.setdefault(key, set()).add(row["source_name"])
    summary["drawable_hours"] = {w: {k: round(v, 1) for k, v in sorted(d.items())}
                                 for w, d in sorted(hours.items())}
    summary["probe_real_atoms"] = {k: len(v) for k, v in sorted(atoms.items())
                                   if k.startswith("real")}
    return plan, report, summary


def write_outputs(out_dir: Path, plan: FoldPlan, report: AuditReport,
                  summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan.frame.to_parquet(out_dir / "folds.parquet", index=False)
    (out_dir / "folds.caveats.txt").write_text(
        "\n".join(plan.caveats) + ("\n" if plan.caveats else ""), encoding="utf-8")
    (out_dir / "folds.vg1.txt").write_text(str(report) + "\n", encoding="utf-8")
    (out_dir / "folds_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
