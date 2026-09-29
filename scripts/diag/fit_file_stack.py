"""Fit processing.file_stack on a run's fold VAL predictions; report leave-one-fold-out.

    $V scripts/diag/fit_file_stack.py --run /data/project/private/dacon-runs/first-v3 \
        --out /data/project/private/dacon-runs/first-v3/diag/file_stack.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metrics.dacon import eer  # noqa: E402
from processing import file_stack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()
    frames = []
    for d in sorted(Path(a.run).glob("fold[0-9]")):
        f = pd.read_parquet(d / "val_predictions.parquet")
        f["fold_name"] = d.name
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for fold, te in df.groupby("fold_name"):
        st = file_stack.fit(df[df.fold_name != fold], C=a.C)
        s = file_stack.apply({c: te[c].to_numpy() for c in te.columns}, st)
        rows.append({"fold": fold, "head": eer(te.file_fake, te.FILE_FAKE_PROB),
                     "stacked": eer(te.file_fake, s)})
    lofo = pd.DataFrame(rows)
    print(lofo.round(4).to_string(index=False))
    print("mean", lofo[["head", "stacked"]].mean().round(4).to_dict())
    st = file_stack.fit(df, C=a.C)
    st["lofo"] = lofo.round(5).to_dict(orient="records")
    st["run"] = str(a.run)
    Path(a.out).write_text(json.dumps(st, indent=1))
    print("wrote", a.out, np.round(st["coef"], 3), round(st["bias"], 3))


if __name__ == "__main__":
    main()
