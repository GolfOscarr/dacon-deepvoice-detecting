"""Family ``bark``: suno/bark (MIT) with the Korean presets v2/ko_speaker_0..9,
via transformers BarkModel (semantic + coarse + fine GPT-style token LMs, EnCodec 24 kHz).

venv: /data/project/private/dacon-venvs/synth-bark
  uv pip install torch torchaudio transformers soundfile numpy scipy huggingface_hub

Bark generates at most ~13-14 s per call, so each file is one Zeroth sentence
(source duration 4-10 s, ≤ 75 chars, to stay under the cap). Speaker cycles over the 10 Korean presets; the
transformers generation is seeded per batch.

  HF_HOME=... python scripts/synth/synth_bark.py --n-files 2000 --shard 0/2
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_common as sc  # noqa: E402

FAMILY = "bark"
MODEL = "suno/bark"
LICENCE = "MIT"
PRESETS = [f"v2/ko_speaker_{i}" for i in range(10)]


def make_bark_jobs(n_files: int, seed: int, utts: list[sc.Utt]) -> list[sc.Job]:
    rng = random.Random(sc._family_seed(FAMILY, seed))
    pool = [u for u in utts if 4.0 <= u.duration_s <= 10.0 and len(u.text) <= 75]
    jobs = []
    for i in range(n_files):
        u = rng.choice(pool)
        jobs.append(sc.Job(i, f"{FAMILY}_{i:06d}", u.text, [u.utt_id],
                           prompt_speaker=f"{FAMILY}/{PRESETS[i % len(PRESETS)]}", seed=seed * 1_000_003 + i))
    return jobs


def main() -> None:
    ap = sc.standard_argparser(FAMILY, default_files=2000, default_seed=707)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, BarkModel
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(args.model))
    revision = snap.name
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(snap)
    model = BarkModel.from_pretrained(snap, torch_dtype=torch.float16).to(dev).eval()
    sr = model.generation_config.sample_rate

    utts = sc.build_index()
    jobs = sc.shard_jobs(make_bark_jobs(args.n_files, args.seed, utts), args.shard)
    w = sc.FamilyWriter(FAMILY, args.model, revision, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    sc.log(f"{FAMILY}: {len(jobs)} jobs in shard, {len(todo)} to do, device={dev}, sr={sr}")

    # group by preset so a batch shares one voice preset
    by_preset: dict[str, list[sc.Job]] = {}
    for j in todo:
        by_preset.setdefault(j.prompt_speaker, []).append(j)
    for preset, js in by_preset.items():
        vp = preset.split("/", 1)[1]
        for b in range(0, len(js), args.batch):
            batch = js[b:b + args.batch]
            inputs = proc([j.text for j in batch], voice_preset=vp, return_tensors="pt").to(dev)
            torch.manual_seed(batch[0].seed)
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=True, semantic_temperature=0.7,
                                     coarse_temperature=0.7, fine_temperature=0.5)
            wavs = out.float().cpu().numpy()
            for j, wav in zip(batch, wavs):
                w.write(j, wav, sr)
                w.progress()
                if args.limit and w.n_written >= args.limit:
                    print(w.summary()); return
    print(w.summary())


if __name__ == "__main__":
    main()
