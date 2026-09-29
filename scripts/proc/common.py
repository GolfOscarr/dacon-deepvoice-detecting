"""Shared I/O for the S1 processing families (docs/training/09 §2 S1).

Contract read by processing/extend.py `_proc`:
  interim/proc/<family>/<side>/<name>.wav   mono PCM16 at the tool's native rate
  interim/proc/metadata.csv                 one row per written file (appended under a lock)
"""
from __future__ import annotations
import csv, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from filelock import FileLock

ROOT = Path("/data/project/private/dacon-corpus")
OUT = ROOT / "interim" / "proc"
META = OUT / "metadata.csv"
_LOCKS: dict[int, FileLock] = {}


def _lock() -> FileLock:
    """One FileLock per process (filelock refuses a lock inherited across fork)."""
    pid = os.getpid()
    if pid not in _LOCKS:
        _LOCKS[pid] = FileLock(str(OUT / ".metadata.lock"))
    return _LOCKS[pid]
COLS = ["file", "family", "side", "source_file_id", "source_path", "transform", "params_json",
        "tool", "tool_version", "licence", "seed", "duration_s", "sample_rate", "rms_dbfs",
        "kept", "drop_reason"]
MIN_S, MIN_RMS = 3.0, -45.0


def plan(family: str, shard: str = "0/1", limit: int | None = None) -> pd.DataFrame:
    """This family's drawn rows, minus files already in metadata.csv; sharded i/n."""
    d = pd.read_parquet(OUT / "_plan" / "draw.parquet")
    d = d[d.family == family].sort_values("file_id").reset_index(drop=True)
    i, n = (int(x) for x in shard.split("/"))
    d = d.iloc[i::n]
    done = done_files()
    d = d[~d.apply(lambda r: rel(family, r.side, r["name"]) in done, axis=1)]
    return d.head(limit) if limit else d


def rel(family: str, side: str, name: str) -> str:
    return f"{family}/{side}/{name}.wav"


_DONE: dict = {"off": 0, "set": set()}


def _refresh_done() -> set[str]:
    """Caller holds the lock. Reads only the rows appended since the last call."""
    if not META.exists():
        return _DONE["set"]
    with open(META, "rb") as fh:
        fh.seek(_DONE["off"])
        chunk = fh.read()
    end = chunk.rfind(b"\n") + 1
    for line in chunk[:end].decode().splitlines():
        f = line.split(",", 1)[0]
        if f and f != "file":
            _DONE["set"].add(f)
    _DONE["off"] += end
    return _DONE["set"]


def done_files() -> set[str]:
    with _lock():
        return set(_refresh_done())


def is_done(family: str, side: str, name: str) -> bool:
    with _lock():
        return rel(family, side, name) in _refresh_done()


def load(path: str, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Mono float32. Resampled (soxr, VHQ) to `sr` if given."""
    p = ROOT / path
    try:
        x, r = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception:                                    # mp3 on an old libsndfile
        r = sr or 48000
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "f32le", "-ac", "1",
                              "-ar", str(r), "-"], check=True, capture_output=True).stdout
        x = np.frombuffer(raw, dtype=np.float32)[:, None]
    x = x.mean(axis=1)
    if sr and r != sr:
        import soxr
        x = soxr.resample(x, r, sr, quality="VHQ").astype(np.float32)
        r = sr
    return x, r


def qc(y: np.ndarray, sr: int) -> tuple[float, float, bool, str]:
    dur = len(y) / sr
    finite = bool(np.isfinite(y).all())
    rms = float(np.sqrt(np.mean(np.square(y.astype(np.float64))))) if finite and len(y) else 0.0
    db = 20 * math.log10(rms) if rms > 0 else -math.inf
    reasons = []
    if not finite: reasons.append("non_finite")
    if dur < MIN_S: reasons.append("short")
    if finite and (np.max(np.abs(y)) if len(y) else 0) == 0: reasons.append("all_silent")
    elif db < MIN_RMS: reasons.append("quiet")
    return dur, db, not reasons, ";".join(reasons)


def write(row, y: np.ndarray, sr: int, *, transform: str, params: dict, tool: str,
          tool_version: str, licence: str) -> dict:
    """Write the WAV (atomic rename) and append its metadata row. Peak-limits to
    0.999 only when the tool overshoots full scale (recorded in params)."""
    family, side = row.family, row.side
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0) if not np.isfinite(y).all() else y
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.999:
        y = y * (0.999 / peak); params = {**params, "peak_normalised_from": round(peak, 4)}
    dur, db, kept, why = qc(y, sr)
    f = rel(family, side, row["name"])
    dst = OUT / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.{os.getpid()}.tmp.wav")
    sf.write(str(tmp), y, sr, subtype="PCM_16")
    rec = {"file": f, "family": family, "side": side, "source_file_id": row.file_id,
           "source_path": row.path, "transform": transform,
           "params_json": json.dumps(params, sort_keys=True), "tool": tool,
           "tool_version": tool_version, "licence": licence, "seed": int(row.seed),
           "duration_s": round(dur, 4), "sample_rate": int(sr),
           "rms_dbfs": round(db, 2) if math.isfinite(db) else -999.0,
           "kept": bool(kept), "drop_reason": why}
    with _lock():
        if f in _refresh_done():                    # another process got here first
            tmp.unlink(missing_ok=True)
            return {**rec, "duplicate_skipped": True}
        os.replace(tmp, dst)
        new = not META.exists()
        with open(META, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            if new: w.writeheader()
            w.writerow(rec)
        _DONE["set"].add(f)
    return rec


class Speed:
    """Audio seconds in / wall seconds, printed every `every` files."""
    def __init__(self, tag: str, every: int = 200):
        self.tag, self.every, self.t0, self.a, self.n = tag, every, time.time(), 0.0, 0
    def add(self, audio_s: float):
        self.a += audio_s; self.n += 1
        if self.n % self.every == 0: self.report()
    def report(self):
        w = time.time() - self.t0
        print(f"[{self.tag}] files={self.n} audio_h={self.a/3600:.2f} wall_s={w:.0f} "
              f"x_rt={self.a/max(w,1e-9):.1f}", flush=True)


# --------------------------------------------------------------------------- #
# generic runner: CPU process pool (workers > 0) or in-process on one device


_W: dict = {}


def _init(init_fn, proc_fn, threads):
    import torch
    torch.set_num_threads(threads)
    _W["proc"] = proc_fn
    _W["model"] = init_fn("cpu")
    _W["dev"] = "cpu"


def _one(row):
    proc_fn = _W["proc"]
    import traceback
    if is_done(row.family, row.side, row["name"]):
        return 0.0, None
    try:
        return proc_fn(_W["model"], _W["dev"], row), None
    except Exception:
        return 0.0, f"{row.file_id}: {traceback.format_exc(limit=4)}"


def run(tag: str, rows: pd.DataFrame, init_fn, proc_fn, *, workers: int, threads: int = 1,
        device: str = "cpu", reverse: bool = False) -> None:
    """proc_fn(model, device, row) -> written duration_s. Errors are logged and the
    file is left undone (a rerun retries it)."""
    print(f"[{tag}] todo={len(rows)} audio_h={rows.duration_s.sum()/3600:.2f} "
          f"device={device} workers={workers}", flush=True)
    sp, errs = Speed(tag), 0
    items = [r for _, r in rows.iterrows()]
    if reverse: items = items[::-1]
    if workers > 0:
        from multiprocessing import get_context
        with get_context("fork").Pool(workers, initializer=_init,
                                      initargs=(init_fn, proc_fn, threads)) as pool:
            for dur, err in pool.imap_unordered(_one, items, chunksize=2):
                if err: errs += 1; print("ERR", err, flush=True)
                else: sp.add(dur)
    else:
        _W["model"] = init_fn(device); _W["dev"] = device; _W["proc"] = proc_fn
        for it in items:
            dur, err = _one(it)
            if err: errs += 1; print("ERR", err, flush=True)
            else: sp.add(dur)
    sp.report(); print(f"[{tag}] done errors={errs}", flush=True)


def argp(default_workers: int = 0):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=default_workers)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--reverse", action="store_true", help="process the plan back to front")
    return ap
