#!/usr/bin/env python3
"""Demucs `htdemucs` `no_vocals` stems for one shard of `selection.csv`.

    TORCH_HOME=/data/project/private/dacon-weights/demucs \\
    python scripts/data_extra/separate_music.py --work <work> --shard 0 --nshards 2

Reads `<work>/selection.csv` (from `select_music_sep.py`), takes rows
`sel_idx % nshards == shard`, and for each: decode the source file, cut the
middle excerpt (`offset_s`, `excerpt_s`), convert to 44.1 kHz stereo, run
htdemucs, sum drums + bass + other (= `no_vocals`, exactly what
`demucs --two-stems vocals` writes) and save WAV PCM16 44.1 kHz stereo to

    <corpus>/interim/sonics-sep/<version>/<id>.wav      (side == fake)
    <corpus>/interim/realmusic-sep/<corpus>/<id>.wav    (side == real)

One pipeline, one code path, both sides -- separation is a transform both
labels carry, never a cue. `shifts=0`, `overlap=0.25`, output clamped to
[-1, 1] (peak recorded so clipping is visible). Rerunnable: an output that
already exists is skipped. Per-file rows go to `<work>/sep_<shard>.csv`
(rerun appends; `screen_stems.py` dedups on out_rel, last row wins).
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

CORPUS = pathlib.Path("/data/project/private/dacon-corpus")
ROOTS = {"fake": CORPUS / "interim/sonics-sep", "real": CORPUS / "interim/realmusic-sep"}
MODEL_SR, MODEL_CH = 44100, 2
COLS = ["sel_idx", "side", "out_rel", "source_path", "source_file_id", "group",
        "offset_s", "duration_s", "sample_rate", "src_sr", "src_channels",
        "peak", "sep_ok", "sep_error"]


def load_excerpt(row) -> tuple[torch.Tensor, int, int, float]:
    """(wav at 44.1k stereo, src_sr, src_channels, actual excerpt seconds)."""
    from demucs.audio import convert_audio
    wav, sr = torchaudio.load(str(CORPUS / row.path))       # (C, T) float32
    a = int(round(row.offset_s * sr))
    n = int(round(row.excerpt_s * sr))
    seg = wav[:, a:a + n]
    if seg.shape[1] < sr:                                    # < 1 s decoded: refuse
        raise RuntimeError(f"excerpt too short: {seg.shape[1]} samples at {sr} Hz")
    seg = convert_audio(seg, sr, MODEL_SR, MODEL_CH)
    return seg, sr, wav.shape[0], seg.shape[1] / MODEL_SR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=pathlib.Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--loaders", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N rows only")
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    model = get_model("htdemucs")
    model.to(args.device).eval()
    assert model.sources == ["drums", "bass", "other", "vocals"], model.sources
    novoc = [i for i, s in enumerate(model.sources) if s != "vocals"]

    sel = pd.read_csv(args.work / "selection.csv")
    sel = sel[sel["sel_idx"] % args.nshards == args.shard]
    if args.limit:
        sel = sel.head(args.limit)
    todo = [r for r in sel.itertuples(index=False)
            if not (ROOTS[r.side] / r.out_rel).exists()]
    print(f"shard {args.shard}/{args.nshards}: {len(sel)} rows, {len(todo)} to do", flush=True)

    log_path = args.work / f"sep_{args.shard}.csv"
    new = not log_path.exists()
    fh = open(log_path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if new:
        w.writeheader()

    def base(r, **kw):
        return {"sel_idx": r.sel_idx, "side": r.side, "out_rel": r.out_rel,
                "source_path": r.path, "source_file_id": r.file_id, "group": r.group,
                "offset_s": r.offset_s, "duration_s": None, "sample_rate": MODEL_SR,
                "src_sr": None, "src_channels": None, "peak": None,
                "sep_ok": False, "sep_error": None, **kw}

    t0, done, audio_s = time.time(), 0, 0.0
    pool = ThreadPoolExecutor(args.loaders)
    # Prefetch decodes a few batches ahead of the GPU.
    futures = [(r, pool.submit(load_excerpt, r)) for r in todo]
    i = 0
    while i < len(futures):
        batch, rows = [], []
        while i < len(futures) and len(batch) < args.batch:
            r, fut = futures[i]; i += 1
            try:
                seg, src_sr, src_ch, dur = fut.result()
            except Exception as exc:                          # noqa: BLE001
                w.writerow(base(r, sep_error=f"decode: {exc}"[:200])); fh.flush()
                continue
            batch.append((seg, src_sr, src_ch, dur)); rows.append(r)
        if not batch:
            continue
        T = max(s.shape[1] for s, *_ in batch)
        mix = torch.zeros(len(batch), MODEL_CH, T)
        for k, (s, *_) in enumerate(batch):
            mix[k, :, :s.shape[1]] = s
        ref = mix.mean(1, keepdim=True)
        mean, std = ref.mean(dim=2, keepdim=True), ref.std(dim=2, keepdim=True) + 1e-8
        mix_n = (mix - mean) / std
        with torch.no_grad():
            out = apply_model(model, mix_n.to(args.device), shifts=0, split=True,
                              overlap=0.25, progress=False, device=args.device)
        out = out.cpu() * std[:, None] + mean[:, None]           # (B, S, C, T)
        stems = out[:, novoc].sum(1)                            # (B, C, T)
        for k, r in enumerate(rows):
            seg, src_sr, src_ch, dur = batch[k]
            y = stems[k, :, :seg.shape[1]].numpy().T           # (T, C)
            peak = float(np.abs(y).max()) if y.size else 0.0
            y = np.clip(y, -1.0, 1.0)
            dst = ROOTS[r.side] / r.out_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".part.wav")
            sf.write(tmp, y, MODEL_SR, subtype="PCM_16")
            os.replace(tmp, dst)
            w.writerow(base(r, duration_s=round(dur, 3), src_sr=src_sr, src_channels=src_ch,
                            peak=round(peak, 4), sep_ok=True))
            done += 1; audio_s += dur
        fh.flush()
        if done % (args.batch * 10) < args.batch:
            el = time.time() - t0
            print(f"  {done}/{len(todo)}  {audio_s/3600:.2f} h  {audio_s/max(el,1e-6):.0f}x realtime  "
                  f"{el/60:.1f} min", flush=True)
    pool.shutdown()
    fh.close()
    print(f"shard {args.shard} done: {done} files, {audio_s/3600:.2f} h in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
