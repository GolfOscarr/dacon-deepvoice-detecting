"""Family ``melo``: MeloTTS-Korean (myshell-ai/MeloTTS, MIT; VITS with BERT prosody, 44.1 kHz).

venv: /data/project/private/dacon-venvs/synth-melo
  uv pip install "git+https://github.com/myshell-ai/MeloTTS.git" soundfile
Weights: HF myshell-ai/MeloTTS-Korean (single built-in speaker "KR"); the
Korean text front end also pulls kykim/bert-kor-base from HF.

Per-file variation: seeded speed / noise_scale / noise_scale_w / sdp_ratio.

  HF_HOME=... python scripts/synth/synth_melo.py --n-files 3000 --shard 0/1
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_common as sc  # noqa: E402

FAMILY = "melo"
MODEL = "myshell-ai/MeloTTS-Korean"
LICENCE = "MIT"


def load_melo(device: str):
    """Returns (melo TTS object, speaker id, sample rate, HF revision)."""
    from melo.api import TTS
    from huggingface_hub import hf_hub_download
    ckpt = Path(hf_hub_download(MODEL, "checkpoint.pth"))
    revision = ckpt.resolve().parent.name if "snapshots" in str(ckpt.resolve()) else "unknown"
    tts = TTS(language="KR", device=device)
    return tts, tts.hps.data.spk2id["KR"], tts.hps.data.sampling_rate, revision


def melo_synth(tts, spk: int, text: str, rng: random.Random) -> np.ndarray:
    import torch
    torch.manual_seed(rng.getrandbits(31))
    return tts.tts_to_file(text, spk, output_path=None, quiet=True,
                           speed=rng.uniform(0.85, 1.15), noise_scale=rng.uniform(0.4, 0.8),
                           noise_scale_w=rng.uniform(0.6, 1.0), sdp_ratio=rng.uniform(0.1, 0.5))


def main() -> None:
    ap = sc.standard_argparser(FAMILY, default_files=3000, default_seed=404)
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tts, spk, sr, revision = load_melo(dev)

    utts = sc.build_index()
    jobs = sc.shard_jobs(sc.make_jobs(FAMILY, args.n_files, args.seed, utts, cloning=False,
                                      builtin_speakers=["KR"]), args.shard)
    w = sc.FamilyWriter(FAMILY, MODEL, revision, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    sc.log(f"{FAMILY}: {len(jobs)} jobs in shard, {len(todo)} to do, device={dev}, sr={sr}")

    for j in todo:
        w.write(j, melo_synth(tts, spk, j.text, random.Random(j.seed)), sr)
        w.progress()
        if args.limit and w.n_written >= args.limit:
            break
    print(w.summary())


if __name__ == "__main__":
    main()
