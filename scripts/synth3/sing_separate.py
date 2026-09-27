#!/usr/bin/env python3
"""N2: Demucs `htdemucs` VOCAL stems, cut into vocal-active chunks, one shard of a selection.

    TORCH_HOME=/data/project/private/dacon-weights/demucs \\
    python scripts/synth3/sing_separate.py --selection <work>/selection.csv --tag main --shard 0 --nshards 1

The same code path for every side (fake SONICS / ACE-Step songs, real FMA /
MUSAN songs), so separation is a transform both labels carry, never a cue:

  1. decode, cut the excerpt (`offset_s`, `excerpt_s`);
  2. CANONICALISE: downmix to mono, resample to 16 kHz. SONICS is distributed
     at 16 kHz mono, so every source is brought to that before anything else:
     a real song's stem then has no band above 8 kHz that a fake one lacks;
  3. up to 44.1 kHz stereo (Demucs' rate), htdemucs, keep `vocals`
     (`shifts=0`, `overlap=0.25`, as `scripts/data_extra/separate_music.py`);
  4. back to 16 kHz mono; frame RMS (20 ms); a frame is active when its level
     is above max(-50 dBFS, p95 - 30 dB); gaps < 0.8 s are closed; each active
     region >= 4 s (padded 0.1 s) is one chunk;
  5. a chunk whose peak exceeds 0.99 is scaled down to 0.99 (a gain change, never
     clipping: mastered real songs clip far more often than SONICS stems, and
     hard clipping would be a label cue); `peak` records the value before;
     each chunk is written WAV PCM16 16 kHz mono to

       <corpus>/interim/sing-fake/<group>/<id>_<k>.wav   (side fake)
       <corpus>/interim/sing-real/<group>/<id>_<k>.wav   (side real)

Per-chunk rows go to `<work>/sep_<tag>_<shard>.csv` (a source with no chunk
gets one row with n_chunks=0). Rerunnable: a source whose `.done` marker
exists in `<work>/done_<tag>/` is skipped.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF

CORPUS = pathlib.Path("/data/project/private/dacon-corpus")
ROOTS = {"fake": CORPUS / "interim/sing-fake", "real": CORPUS / "interim/sing-real"}
CANON_SR, MODEL_SR, MODEL_CH = 16000, 44100, 2
FRAME = 320                                  # 20 ms at 16 kHz
FLOOR_DB, REL_DB, GAP_S, MIN_S, PAD_S = -50.0, 30.0, 0.8, 4.0, 0.1
COLS = ["sel_idx", "side", "group", "file", "source_path", "source_file_id", "offset_s",
        "excerpt_s", "seg_start_s", "duration_s", "sample_rate", "src_sr", "src_channels",
        "peak", "rms_dbfs", "active_ratio", "vox_rel_db", "n_chunks", "sep_ok", "sep_error"]


def frame_db(y: np.ndarray) -> np.ndarray:
    n = len(y) // FRAME
    f = y[:n * FRAME].reshape(n, FRAME)
    return 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)


def regions(db: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Active regions in frames, and the threshold used."""
    if db.size == 0:
        return [], FLOOR_DB
    thr = max(FLOOR_DB, float(np.percentile(db, 95)) - REL_DB)
    act = db > thr
    gap = int(round(GAP_S * CANON_SR / FRAME))
    idx = np.flatnonzero(act)
    out = []
    if idx.size:
        start = prev = idx[0]
        for i in idx[1:]:
            if i - prev > gap:
                out.append((start, prev + 1)); start = i
            prev = i
        out.append((start, prev + 1))
    minf = int(round(MIN_S * CANON_SR / FRAME))
    return [(a, b) for a, b in out if b - a >= minf], thr


def load_canon(row) -> tuple[torch.Tensor, int, int]:
    """(16 kHz mono excerpt (T,), src_sr, src_channels)."""
    wav, sr = torchaudio.load(str(CORPUS / row.path))
    a = int(round(row.offset_s * sr)); n = int(round(row.excerpt_s * sr))
    seg = wav[:, a:a + n].mean(0)
    if seg.shape[0] < sr:
        raise RuntimeError(f"excerpt too short: {seg.shape[0]} samples at {sr} Hz")
    if sr != CANON_SR:
        seg = AF.resample(seg, sr, CANON_SR)
    return seg.contiguous(), sr, wav.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selection", type=pathlib.Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--loaders", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-base", type=pathlib.Path, default=None,
                    help="write sing-fake/ and sing-real/ under this dir instead (smoke tests)")
    args = ap.parse_args()
    torch.set_num_threads(2)
    work = args.selection.parent
    if args.out_base:
        ROOTS.update({"fake": args.out_base / "sing-fake", "real": args.out_base / "sing-real"})
    done_dir = work / f"done_{args.tag}"
    done_dir.mkdir(parents=True, exist_ok=True)

    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    model = get_model("htdemucs")
    model.to(args.device).eval()
    assert model.sources == ["drums", "bass", "other", "vocals"], model.sources
    voc = model.sources.index("vocals")

    sel = pd.read_csv(args.selection)
    sel = sel[sel["sel_idx"] % args.nshards == args.shard]
    if args.limit:
        sel = sel.head(args.limit)
    todo = [r for r in sel.itertuples(index=False) if not (done_dir / f"{r.sel_idx}.done").exists()]
    print(f"shard {args.shard}/{args.nshards}: {len(sel)} rows, {len(todo)} to do", flush=True)

    log_path = work / f"sep_{args.tag}_{args.shard}.csv"
    new = not log_path.exists()
    fh = open(log_path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()

    def base(r, **kw):
        d = {c: None for c in COLS}
        d.update({"sel_idx": r.sel_idx, "side": r.side, "group": r.group, "source_path": r.path,
                  "source_file_id": r.file_id, "offset_s": r.offset_s, "excerpt_s": r.excerpt_s,
                  "sample_rate": CANON_SR, "sep_ok": False, "n_chunks": 0})
        d.update(kw)
        return d

    t0, done, audio_s, kept_s = time.time(), 0, 0.0, 0.0
    pool = ThreadPoolExecutor(args.loaders)
    # bounded prefetch: decode at most `ahead` windows ahead of the GPU
    ahead = args.batch * 8
    futures = [(r, pool.submit(load_canon, r)) for r in todo[:ahead]]
    i = 0
    while i < len(futures):
        batch, rows = [], []
        while i < len(futures) and len(batch) < args.batch:
            r, fut = futures[i]; futures[i] = (r, None); i += 1
            if len(futures) < len(todo):
                nxt = todo[len(futures)]
                futures.append((nxt, pool.submit(load_canon, nxt)))
            try:
                seg, src_sr, src_ch = fut.result()
            except Exception as exc:                          # noqa: BLE001
                w.writerow(base(r, sep_error=f"decode: {exc}"[:200])); fh.flush()
                (done_dir / f"{r.sel_idx}.done").touch()
                continue
            batch.append((seg, src_sr, src_ch)); rows.append(r)
        if not batch:
            continue
        ups = [AF.resample(s, CANON_SR, MODEL_SR) for s, *_ in batch]
        T = max(u.shape[0] for u in ups)
        mix = torch.zeros(len(batch), MODEL_CH, T)
        for k, u in enumerate(ups):
            mix[k, :, :u.shape[0]] = u
        ref = mix.mean(1, keepdim=True)
        mean, std = ref.mean(dim=2, keepdim=True), ref.std(dim=2, keepdim=True) + 1e-8
        with torch.no_grad():
            out = apply_model(model, ((mix - mean) / std).to(args.device), shifts=0, split=True,
                              overlap=0.25, progress=False, device=args.device)
        vox = (out[:, voc].cpu() * std + mean).mean(1)          # (B, T) mono 44.1k
        vox16 = AF.resample(vox, MODEL_SR, CANON_SR)
        for k, r in enumerate(rows):
            seg, src_sr, src_ch = batch[k]
            y = vox16[k, :seg.shape[0]].numpy().astype(np.float32)
            x = seg.numpy()
            db = frame_db(y)
            regs, thr = regions(db)
            pad = int(PAD_S * CANON_SR)
            n = 0
            for a, b in regs:
                s0 = max(0, a * FRAME - pad); s1 = min(len(y), b * FRAME + pad)
                c = y[s0:s1]
                peak = float(np.abs(c).max())
                rms = float(np.sqrt(np.mean(c ** 2)))
                act = float(np.mean(db[a:b] > thr))
                mix_e = float(np.mean(x[s0:s1] ** 2)) + 1e-12
                rel_db = 10 * np.log10((rms ** 2 + 1e-12) / mix_e)
                rel_file = f"{r.out_prefix}_{n}.wav"
                dst = ROOTS[r.side] / rel_file
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(".part.wav")
                if peak > 0.99:
                    c = c * (0.99 / peak)
                    rms *= 0.99 / peak
                sf.write(tmp, c, CANON_SR, subtype="PCM_16")
                os.replace(tmp, dst)
                w.writerow(base(r, file=rel_file, seg_start_s=round(s0 / CANON_SR, 3),
                                duration_s=round(len(c) / CANON_SR, 3), src_sr=src_sr,
                                src_channels=src_ch, peak=round(peak, 4),
                                rms_dbfs=round(20 * np.log10(rms + 1e-12), 2),
                                active_ratio=round(act, 3), vox_rel_db=round(rel_db, 2),
                                sep_ok=True))
                n += 1; kept_s += len(c) / CANON_SR
            if n == 0:
                w.writerow(base(r, src_sr=src_sr, src_channels=src_ch, sep_ok=True,
                                duration_s=0.0))
            (done_dir / f"{r.sel_idx}.done").touch()
            done += 1; audio_s += seg.shape[0] / CANON_SR
        fh.flush()
        if done % (args.batch * 10) < args.batch:
            el = time.time() - t0
            print(f"  {done}/{len(todo)}  in {audio_s/3600:.2f} h  chunks {kept_s/3600:.2f} h  "
                  f"{audio_s/max(el,1e-6):.0f}x realtime  {el/60:.1f} min", flush=True)
    pool.shutdown()
    fh.close()
    print(f"shard {args.shard} done: {done} sources, {audio_s/3600:.2f} h in, "
          f"{kept_s/3600:.2f} h of chunks, {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
