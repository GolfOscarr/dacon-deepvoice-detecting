"""G-EDA3 / G-EDA4 / G-EDA7, the G4 loss table and the G7 pack over the built
tables (docs/processing/03 §7 items 4 and 5).

    python scripts/verify_gates.py \
        --manifest-dir /data/project/private/dacon-corpus/manifests/strategy-v1

Reads ``manifest.parquet``, ``verdict.parquet`` and ``folds.parquet`` from the
manifest dir and the EDA file table from ``configs/eda.yaml``; writes
``gates_report.json``, ``g_eda7_ledger.csv``, ``g4_table.csv``, ``g7_pack.csv``
and ``g7_examples.csv`` beside them. Exit status 1 on any failing gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eda.config import load_eda_config                            # noqa: E402
from eda.driver import load_files                                  # noqa: E402
from processing.gates import FAIL, run_gates                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--eda-config", default="configs/eda.yaml")
    args = ap.parse_args()
    d = Path(args.manifest_dir)
    manifest = pd.read_parquet(d / "manifest.parquet")
    verdict = pd.read_parquet(d / "verdict.parquet")
    folds = pd.read_parquet(d / "folds.parquet")
    files = load_files(load_eda_config(args.eda_config))
    out = run_gates(manifest, verdict, folds, files)

    pd.set_option("display.width", 200, "display.max_colwidth", 400)
    print(out["gates"].to_string(index=False))
    print(f"\naggregate: {out['aggregate']}")
    print("\n== G-EDA7 ledger (drop rate per filter and label)")
    print(out["g_eda7_ledger"].round(4).to_string(index=False))
    print("\n== G4 loss table (rows and hours dropped per filter and pool / cell)")
    print(out["g4_table"].round(4).to_string(index=False))
    print("\n== G7 pack (reassigned per source)")
    print(out["g7_pack"].round(4).to_string(index=False))

    (d / "gates_report.json").write_text(json.dumps(
        {"gates": out["gates"].to_dict(orient="records"), "aggregate": out["aggregate"]},
        indent=2), encoding="utf-8")
    for name in ("g_eda7_ledger", "g4_table", "g7_pack", "g7_examples"):
        out[name].to_csv(d / f"{name}.csv", index=False)
    return 1 if (out["gates"]["verdict"] == FAIL).any() else 0


if __name__ == "__main__":
    sys.exit(main())
