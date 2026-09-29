"""S3: download a deterministic sample of Emilia KO (or EN) tars (both parts) + READMEs.

Emilia/KO: all 40 tars (~5 GB). Emilia-YODAS/KO: N_YODAS tars spread evenly over the 208
(index round(k*208/N)), so the sample spans many upload batches. Files land in
interim/emilia-ko/_raw/<repo path>. Resumable (hf_hub_download skips complete files).

  python scripts/synth2/ko_emilia_download.py [--n-yodas 8]
  python scripts/synth2/ko_emilia_download.py --lang en --n-emilia 8 --n-yodas 8
    EN: Emilia/EN has 1,140 tars (~2 GB, ~40 h each) and YODAS/EN 1,362 (~1 GB), so both
    parts are sampled evenly (--n-emilia / --n-yodas) into interim/emilia-en/_raw/.
"""
import argparse, os, sys, time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "amphion/Emilia-Dataset"

ap = argparse.ArgumentParser(); ap.add_argument("--n-yodas", type=int, default=8)
ap.add_argument("--only", default="")
ap.add_argument("--lang", default="ko", choices=["ko", "en"])
ap.add_argument("--n-emilia", type=int, default=0, help="0 = all Emilia tars of the language")
a = ap.parse_args()
L = a.lang.upper()
RAW = Path(f"/data/project/private/dacon-corpus/interim/emilia-{a.lang}/_raw")
api = HfApi()
files = sorted(f.path for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True)
               if hasattr(f, "size"))
ko_em = [p for p in files if p.startswith(f"Emilia/{L}/") and p.endswith(".tar")]
ko_yo = [p for p in files if p.startswith(f"Emilia-YODAS/{L}/") and p.endswith(".tar")]
if a.n_emilia:
    ko_em = [ko_em[round(k * len(ko_em) / a.n_emilia)] for k in range(a.n_emilia)]
readmes = [p for p in files if p.lower().endswith("readme.md") or "licen" in p.lower()]
pick_yo = [ko_yo[round(k * len(ko_yo) / a.n_yodas)] for k in range(a.n_yodas)]
todo = readmes + ko_em + pick_yo
if a.only == "readme": todo = readmes
print(f"{len(ko_em)} Emilia {L}, {len(ko_yo)} YODAS {L}; picking {ko_em} {pick_yo}", flush=True)
for p in todo:
    t = time.time()
    for attempt in range(5):
        try:
            hf_hub_download(REPO, p, repo_type="dataset", local_dir=str(RAW))
            break
        except Exception as e:
            print("retry", p, e, flush=True); time.sleep(10)
    sz = (RAW / p).stat().st_size if (RAW / p).exists() else -1
    print(f"OK {p} {sz/1e6:.0f} MB {time.time()-t:.0f}s", flush=True)
print("DONE", flush=True)
