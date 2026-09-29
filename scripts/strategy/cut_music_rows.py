#!/usr/bin/env python3
"""strategy-v6: cut every long music component row (pools C and D) into 10 s pieces, on
BOTH sides by one rule, so each music corpus's share of drawn tiles follows its HOURS.

    $V scripts/strategy/cut_music_rows.py --in-dir <v6>/_pre --out <v6> [--workers 16]

Why: the music draw has ONE bucket per side and picks every tile's file uniformly among
the rows that can hold it (processing.sampler.bucket_keys / _tiles); draw.domain_weights
only picks the anchor that names the bucket. So a music corpus's tile share is its ROW
COUNT: fakemusiccaps' 10 s rows took ~.85 of fake-music tiles against sonics-sep's 65 s
rows (Suno / Udio), whatever the weights.

Rule (identical for C and D): a row of `duration_s >= MIN_CUT_S` becomes consecutive
PIECE_S pieces; a last remainder >= MIN_LAST_S is kept as a shorter piece, anything less
is dropped, and so is a piece under RMS_MIN_DBFS (a stem's silent stretch would be a
"music present" tile with no music; the Jamendo and ACE-Step 10 s rows apply the same
floor). Shorter rows (fakemusiccaps' 10-10.24 s, the 10 s mtg-jamendo / acestep-inst
rows, cut from 30 s clips by this same rule upstream) are kept as they are.

A piece is a new file: 16 kHz PCM16 WAV cut from the parent's cache16k array (exactly
the audio training reads, channel count kept) at interim/music-cut10/<parent id>~<kk>.wav;
existing interim files are never touched. file_id `cut10:<parent file_id, ':' -> '/'>~kk`
(cache16k/cut10/...). A piece inherits EVERY manifest column of its parent (labels,
atoms, family, domain, licence, orig_sr / orig_channels / container: they describe the
source recording) except file_id, path, sha256, duration_s, stage; and its parent's
fold-table row (slice incl. PROBE, fold, atoms) with only file_id replaced -- no fold
moves. The parents leave the manifest and the fold table. VG1 must pass.

Deterministic: no randomness; the same cache gives byte-identical pieces (checked).
Writes <out>/manifest.parquet, folds.parquet (+ caveats / vg1 / report), cut_map.parquet
(the provenance ledger: piece -> parent file_id, parent path, offset_s within the parent's
16 kHz audio, licence_verdict), cut_report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from processing.cache import cache_path                       # noqa: E402
from processing.splits import write_outputs                   # noqa: E402
from training.foldcheck import check_split_integrity           # noqa: E402
from training.folds import FoldPlan                            # noqa: E402
from training.manifest import validate_manifest                # noqa: E402

CORPUS = Path("/data/project/private/dacon-corpus")
CACHE = CORPUS / "cache16k"
CUT_DIR = "interim/music-cut10"
SR, PIECE_S, MIN_CUT_S, MIN_LAST_S = 16000, 10.0, 15.0, 5.0
RMS_MIN_DBFS = -40.0          # the Jamendo / ACE-Step segments' near-silent floor


def pieces_of(duration_s: float) -> list[tuple[float, float]]:
    """``[(offset_s, length_s), ...]`` of one row; ``[]`` = keep the row as it is."""
    if duration_s < MIN_CUT_S:
        return []
    n = int(duration_s // PIECE_S)
    out = [(k * PIECE_S, PIECE_S) for k in range(n)]
    rest = duration_s - n * PIECE_S
    if rest >= MIN_LAST_S:
        out.append((n * PIECE_S, rest))
    return out


def piece_id(parent: str, k: int) -> str:
    return f"cut10:{parent.replace(':', '/')}~{k:02d}"


def _write_one(parent: str, root: Path, cache_root: Path) -> list[dict]:
    import soundfile as sf
    arr = np.load(cache_path(cache_root, parent), mmap_mode="r")          # (C, n) int16
    total = arr.shape[-1] / SR
    rows = []
    for k, (off, length) in enumerate(pieces_of(total)):
        a, b = int(round(off * SR)), int(round((off + length) * SR))
        x = np.asarray(arr[:, a:b], dtype=np.float64) / 32768.0
        if 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12) < RMS_MIN_DBFS:
            continue                                   # near-silent piece: no music to draw
        fid = piece_id(parent, k)
        rel = f"{CUT_DIR}/{fid.split(':', 1)[1]}.wav"
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            tmp = dst.with_name(dst.name + ".part.wav")
            sf.write(tmp, np.ascontiguousarray(arr[:, a:b].T), SR, subtype="PCM_16")
            os.replace(tmp, dst)
        h = hashlib.sha256(dst.read_bytes()).hexdigest()
        rows.append({"file_id": fid, "parent": parent, "k": k, "path": rel, "offset_s": off,
                     "duration_s": (b - a) / SR, "sha256": h})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", required=True, help="manifest.parquet + folds.parquet to cut")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus-root", default=str(CORPUS))
    ap.add_argument("--cache-root", default=str(CACHE))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    src, out = Path(a.in_dir), Path(a.out)
    if (out / "manifest.parquet").exists() or (out / "folds.parquet").exists():
        raise SystemExit(f"FATAL: {out} already has a manifest / fold table")
    m = pd.read_parquet(src / "manifest.parquet")
    f = pd.read_parquet(src / "folds.parquet")
    music = m["pool"].isin(["C", "D"]) & m["row_kind"].eq("component")
    cut = music & (m["duration_s"] >= MIN_CUT_S)
    parents = m.loc[cut, "file_id"].astype(str).tolist()
    root, cache_root = Path(a.corpus_root), Path(a.cache_root)
    with ThreadPoolExecutor(a.workers) as ex:
        got = [r for rows in ex.map(lambda p: _write_one(p, root, cache_root), parents) for r in rows]
    cm = pd.DataFrame(got)
    # identical audio twice (a piece equal to another piece or to a kept row: e.g. digital
    # silence padding, or one recording under two parents) enters once, as extend_manifest
    # drops a duplicate sha256
    dup = cm["sha256"].duplicated() | cm["sha256"].isin(set(m.loc[~cut.to_numpy(), "sha256"].astype(str)))
    dups = cm[dup]
    cm = cm[~dup].reset_index(drop=True)
    # a parent's own cached length can differ from its manifest duration by a few samples
    lost = sorted(set(parents) - set(cm["parent"]))
    base = m.set_index("file_id")
    new = base.loc[cm["parent"]].reset_index(drop=True)
    for c in ("file_id", "path", "sha256", "duration_s"):
        new[c] = cm[c].to_numpy()
    new["stage"] = "interim"
    mm = pd.concat([m[~cut.to_numpy()], new[m.columns]], ignore_index=True)
    for c in m.columns:
        if str(m[c].dtype) != str(mm[c].dtype):
            mm[c] = mm[c].astype(m[c].dtype)
    # the base's own sha256 duplicates (129 rows in v5: compspoof, sonics, mlaad) stay
    if mm["file_id"].duplicated().any() or new["sha256"].isin(set(mm["sha256"].astype(str))
                                                              .difference(new["sha256"])).any():
        raise SystemExit("FATAL: duplicate file_id, or a piece equal to a kept row, after the cut")
    validate_manifest(mm)
    fb = f.set_index("file_id")
    missing = sorted(set(cm["parent"]) - set(fb.index))
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} cut parents have no fold row, e.g. {missing[:3]}")
    nf = fb.loc[cm["parent"]].reset_index(drop=True)
    nf.insert(0, "file_id", cm["file_id"].to_numpy())
    ff = pd.concat([f[~f["file_id"].isin(set(parents))], nf[f.columns]], ignore_index=True)
    if set(ff["file_id"]) != set(mm["file_id"]):
        raise SystemExit("FATAL: fold table and manifest disagree after the cut")
    vg1 = check_split_integrity(ff)
    if not vg1.ok:
        raise SystemExit(f"FATAL: VG1 fails:\n{vg1}")
    out.mkdir(parents=True, exist_ok=True)
    mm.to_parquet(out / "manifest.parquet", index=False)
    # provenance ledger: every piece -> its original file, offset and licence
    par = base.loc[cm["parent"], ["path", "corpus", "licence_verdict"]].reset_index(drop=True)
    cm = cm.assign(parent_path=par["path"].to_numpy(), parent_corpus=par["corpus"].to_numpy(),
                   licence_verdict=par["licence_verdict"].to_numpy())
    cm.to_parquet(out / "cut_map.parquet", index=False)

    def hours(d: pd.DataFrame) -> dict:
        g = d[d["pool"].isin(["C", "D"]) & d["row_kind"].eq("component")]
        return {f"{p}/{c}": {"rows": int(len(x)), "hours": round(float(x["duration_s"].sum()) / 3600, 2)}
                for (p, c), x in g.groupby(["pool", "corpus"])}
    slices_before = f[f["file_id"].isin(set(parents))].groupby(["slice", "fold"], dropna=False).size()
    slices_after = nf.groupby(["slice", "fold"], dropna=False).size()
    rep = {"parents_cut": len(parents), "pieces": int(len(cm)), "parents_without_piece": lost,
           "pieces_duplicate_sha_dropped": int(len(dups)),
           "duplicate_examples": dups[["file_id", "parent"]].head(5).to_dict("records"),
           "pieces_near_silent_dropped": int(sum(len(pieces_of(d)) for d in m.loc[cut, "duration_s"])
                                             - len(cm) - len(dups)),
           "rule": {"piece_s": PIECE_S, "min_cut_s": MIN_CUT_S, "min_last_s": MIN_LAST_S},
           "music_before": hours(m), "music_after": hours(mm),
           "parent_slices": {f"{s}/{k}": int(v) for (s, k), v in slices_before.items()},
           "piece_slices": {f"{s}/{k}": int(v) for (s, k), v in slices_after.items()}}
    (out / "cut_report.json").write_text(json.dumps(rep, indent=1))
    write_outputs(out, FoldPlan(ff, (f"cut from {src}: {len(parents)} music rows -> {len(cm)} "
                                     f"{PIECE_S:g} s pieces, each on its parent's (slice, fold)",)),
                  vg1, rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "parents_without_piece"}, indent=1))
    print(f"parents without a piece: {len(lost)}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
