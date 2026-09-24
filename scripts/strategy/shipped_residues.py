"""Acoustic residues on the audio the model receives (docs/processing/03 §7 item 3).

    python scripts/strategy/shipped_residues.py --n 6000 --workers 16 --fold 0

The stream harness audits per-FILE scalars the EDA measured; this script
measures the SAMPLE: ``processing.render`` (the decode cache, tiles, layer,
augments, normalize) followed by ``processing.ship`` (downmix, dc_offset, the
band edge), then the EDA's own level and spectral extractors over the shipped
waveform, and the harness's CV AUC per head -- ungrouped and under the source
holdout. The pre-ship rendered waveform is measured too, as the control that
shows what SHIP-4/5 removed.

Reads ``configs/processing_v1.yaml`` and the built manifest; writes
``eda/out/_strategy/shipped_residues.parquet`` (one row per sample) and
``shipped_residues_audit.parquet`` (one row per head / stratum / stage).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eda.extract.level import level_row                            # noqa: E402
from eda.extract.spectral import spectral_row                      # noqa: E402
from processing.config import load_processing_config              # noqa: E402
from processing.render import ManifestIndex, render                # noqa: E402
from processing.sampler import Sampler                             # noqa: E402
from processing.ship import ship_sample                            # noqa: E402
from scripts.strategy.stream_harness import GATE, SEED, _cv_auc    # noqa: E402
from training.folds import apply_folds                             # noqa: E402
from training.spec import SampleSpec, stratum_of                   # noqa: E402
from eda.analyze.shortcut import univariate_auc                    # noqa: E402

OUT = ROOT / "eda" / "out" / "_strategy"
MANIFEST_DIR = Path("/data/project/private/dacon-corpus/manifests/strategy-v2")

#: The residues the harness audited per file, now measured per sample.
FEATURES = ("rms_dbfs", "peak_dbfs", "crest_factor_db", "dc_offset",
            "effective_bandwidth_hz", "near_nyquist_ratio", "hf_ratio_8k",
            "spectral_flatness", "spectral_centroid_hz", "band_energy_0")
#: Two families, judged apart. RESIDUES are what the pipeline can
#: manufacture and the test chain cannot carry: the level (06 D7 equalises
#: it), a DC offset and the resampler's near-Nyquist shelf (SHIP-4/5 remove
#: them). CONTENT is the spectral shape and the dynamics -- a vocoder's
#: high band, a generator's flatness, music's density -- which IS the signal
#: the heads are asked for, transfers across sources because it is real,
#: and is reported, not gated. The acceptance item (03 §7 item 3) binds on
#: the residues. Caveat: with the level equalised, the peak IS the crest
#: factor (music is denser than speech), so it sits with the content.
RESIDUES = ("rms_dbfs", "dc_offset", "near_nyquist_ratio")
CONTENT = ("peak_dbfs", "crest_factor_db", "effective_bandwidth_hz", "hf_ratio_8k",
           "spectral_flatness", "spectral_centroid_hz", "band_energy_0")
HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")

_STATE: dict = {}


def _init(cfg_path: str, manifest_path: str) -> None:
    torch.set_num_threads(1)
    cfg = load_processing_config(cfg_path)
    m = pd.read_parquet(manifest_path)
    _STATE.update(cfg=cfg, index=ManifestIndex.coerce(m),
                  source=m.set_index("file_id")["source_name"])


def _measure(wav: np.ndarray, sr: int, prefix: str) -> dict:
    row = {**level_row(wav, sr), **spectral_row(wav, sr)}
    return {f"{prefix}{k}": row.get(k) for k in FEATURES}


def _one(d: dict) -> dict:
    spec = SampleSpec.from_dict(d)
    cfg, index, source = _STATE["cfg"], _STATE["index"], _STATE["source"]
    out = render(spec, index, cfg.render)
    sr = out.sample_rate
    pre = out.wav.mean(dim=0, keepdim=True).numpy()         # the plain downmix, as a control
    shipped = ship_sample(out, cfg.ship)[None].numpy()
    roles = {}
    for c in spec.components:
        if c.snr_db is None and c.role not in roles:
            roles[c.role] = source.get(c.file_id, "none")
    layer = next((source.get(c.file_id, "none") for c in spec.components if c.snr_db is not None),
                 None)
    row = {"sample_id": spec.sample_id, "cell": spec.cell, "stratum": stratum_of(spec.cell),
           "duration_s": spec.duration_s, "composed": spec.render_mode == "composed",
           "voice_source": roles.get("voice"), "music_source": roles.get("music"),
           "noise_source": roles.get("noise", layer),
           "normalize_container": spec.normalize.get("container"),
           "telephone": spec.normalize.get("telephone_hz") is not None,
           **{h: spec.labels[h] for h in HEADS}}
    row.update(_measure(pre, sr, "pre__"))
    row.update(_measure(shipped, sr, "ship__"))
    return row


def _group_for(s: pd.DataFrame, head: str) -> pd.Series:
    if head.startswith("music"):
        primary = s["music_source"]
    elif head.startswith("voice"):
        primary = s["voice_source"]
    else:
        primary = s["music_source"].fillna(s["voice_source"])
    return primary.fillna(s["noise_source"]).fillna("none")


def audit(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, family in (("pre", "residue"), ("ship", "residue"), ("ship", "content")):
        names = RESIDUES if family == "residue" else CONTENT
        for head in HEADS:
            cols = [f"{stage}__{f}" for f in names]
            y_all = pd.to_numeric(s[head], errors="coerce")
            scored = y_all.notna()
            if head == "file_fake":
                scored &= s["cell"] != 9
            strata = ["pooled"] + (sorted(s.loc[scored, "stratum"].unique())
                                   if head.endswith("fake") else [])
            for stratum in strata:
                mask = scored & ((s["stratum"] == stratum) if stratum != "pooled" else True)
                sub = s.loc[mask, cols]
                keep = [c for c in cols if sub[c].notna().any() and sub[c].nunique() > 1]
                sub = sub[keep]
                y = y_all[mask].astype(int).to_numpy()
                if not keep or len(np.unique(y)) < 2 or len(y) < 50:
                    continue
                x = sub.fillna(sub.median()).to_numpy(dtype=float)
                groups = _group_for(s, head)[mask].to_numpy()
                uni = sorted(((univariate_auc(x[:, i], y), keep[i].split("__", 1)[1])
                              for i in range(x.shape[1])), reverse=True)[:5]
                rows.append({
                    "stage": stage, "family": family, "head": head, "stratum": stratum,
                    "n": int(len(y)),
                    "positive_rate": float(y.mean()),
                    "auc": _cv_auc(x, y, groups, grouped=False),
                    "auc_source_grouped": _cv_auc(x, y, groups, grouped=True),
                    "n_groups": int(len(np.unique(groups))),
                    "top_features": "; ".join(f"{n}={a:.3f}" for a, n in uni)})
    out = pd.DataFrame(rows)
    judged = out["auc_source_grouped"].fillna(out["auc"])
    out["gate"] = np.where(out["family"] == "content", "reported",
                           np.where(judged < GATE, "pass", "fail"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--processing-config", default="configs/processing_v1.yaml")
    ap.add_argument("--manifest-dir", default=str(MANIFEST_DIR))
    ap.add_argument("--fold", type=int, default=0, help="the fold view to draw the TRAIN slice of")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = load_processing_config(args.processing_config)
    # 05 A7: the raw manifest carries no fold; the sampler refuses it
    manifest = apply_folds(pd.read_parquet(Path(args.manifest_dir) / "manifest.parquet"),
                           pd.read_parquet(Path(args.manifest_dir) / "folds.parquet"), args.fold)
    manifest_path = str(Path(args.manifest_dir) / "manifest.parquet")
    sampler = Sampler(manifest, cfg.draw, slice_="train", fold=args.fold)
    specs = [s.to_dict() for s in sampler.epoch_specs(args.n, epoch=0, seed=args.seed)]
    print(f"drew {len(specs)} specs; rendering + shipping with {args.workers} workers",
          flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(args.processing_config, manifest_path)) as pool:
        for i, row in enumerate(pool.map(_one, specs, chunksize=8)):
            rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(specs)}", flush=True)
    s = pd.DataFrame(rows)
    s.to_parquet(OUT / "shipped_residues.parquet", index=False)

    a = audit(s)
    a.to_parquet(OUT / "shipped_residues_audit.parquet", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(a.to_string(index=False))
    summary = {"n": len(s), "seed": args.seed, "fold": args.fold,
               "ship_dc_offset_abs_max": float(s["ship__dc_offset"].abs().max()),
               "ship_near_nyquist_max": float(s["ship__near_nyquist_ratio"].max()),
               "pre_near_nyquist_max": float(s["pre__near_nyquist_ratio"].max()),
               "fails": a[a["gate"] == "fail"][["stage", "family", "head", "stratum"]]
                        .to_dict(orient="records")}
    (OUT / "shipped_residues_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
