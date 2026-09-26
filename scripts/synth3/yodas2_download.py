"""N3: download a second, disjoint sample of Emilia-YODAS/KO tars (round 2 of S3).

Round 1 (scripts/synth2/ko_emilia_download.py) took 8 of the 208 tars at B000000, 026, ..., 182
(index round(k*208/8)). This round takes the midpoints between them by default
(B000013, 039, 065, 091, 117, 143, 169, 195), so the sample stays evenly spread over the upload
batches and shares no tar with round 1. Files land in interim/emilia-ko/_raw/Emilia-YODAS/KO/.
Resumable (hf_hub_download skips complete files). Auth: the cached HF login (gated dataset).

  python scripts/synth3/yodas2_download.py [--tars KO-B000013,KO-B000039,...]
"""
import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "amphion/Emilia-Dataset"
RAW = Path("/data/project/private/dacon-corpus/interim/emilia-ko/_raw")
ROUND1 = {f"KO-B{round(k * 208 / 8):06d}" for k in range(8)}
DEFAULT = ",".join(f"KO-B{round(k * 208 / 8) + 13:06d}" for k in range(8))

ap = argparse.ArgumentParser()
ap.add_argument("--tars", default=DEFAULT, help="comma-separated tar stems")
a = ap.parse_args()
stems = [s.strip() for s in a.tars.split(",") if s.strip()]
assert not ROUND1 & set(stems), f"round-1 tars requested: {ROUND1 & set(stems)}"
files = {f.path for f in HfApi().list_repo_tree(REPO, "Emilia-YODAS/KO", repo_type="dataset")}
todo = [f"Emilia-YODAS/KO/{s}.tar" for s in stems]
missing = [p for p in todo if p not in files]
assert not missing, f"not in repo: {missing}"
print(f"{len(files)} YODAS/KO files in repo; downloading {stems}", flush=True)
for p in todo:
    t = time.time()
    for attempt in range(5):
        try:
            hf_hub_download(REPO, p, repo_type="dataset", local_dir=str(RAW))
            break
        except Exception as e:  # noqa: BLE001
            print("retry", p, type(e).__name__, str(e)[:200], flush=True)
            time.sleep(10)
    sz = (RAW / p).stat().st_size if (RAW / p).exists() else -1
    print(f"OK {p} {sz / 1e6:.0f} MB {time.time() - t:.0f}s", flush=True)
print("DONE", flush=True)
