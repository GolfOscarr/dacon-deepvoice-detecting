"""Family ``qwen3tts`` (round 3, docs/training/15 N1-ko): Qwen/Qwen3-TTS-12Hz-1.7B-Base zero-shot
cloning (a Qwen3 LM over the 12.5 Hz multi-codebook Qwen3-TTS-Tokenizer, with a sub-talker for the
residual codebooks and the tokenizer's own decoder, 24 kHz, Apache-2.0; Korean is one of its 10
languages). ICL mode: the prompt's audio codes + its dataset transcript, language="Korean".

venv: /data/project/private/dacon-venvs/synth3-qwen3tts (python 3.12, qwen-tts, torch 2.14,
transformers 4.57.3, no flash-attn: sdpa). Weights: HF Qwen/Qwen3-TTS-12Hz-1.7B-Base and
Qwen/Qwen3-TTS-Tokenizer-12Hz under HF_HOME=/data/project/private/dacon-weights/synth3/hf.
Prompts are levelled (-26 dBFS RMS, peak <= 0.5) as in round 2. Batched --batch jobs per call.
Per-batch seeded draws: temperature U(0.75, 0.95), top_p U(0.85, 1.0); at most 24 s per file.

  python scripts/synth3/ko_synth_qwen3tts.py --n-files 6000 --shard 0/2 --max-family-hours 9
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth2"))
import ko_common as kc  # noqa: E402

FAMILY = "qwen3tts"
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base (+ Qwen3-TTS-Tokenizer-12Hz)"
LICENCE = "Apache-2.0"
# 12.5 codec frames/s; texts are <= 13 s of source speech and QC drops > 20 s. The model default
# (8192 frames) lets one row that never emits EOS hold a whole batch for minutes.
MAX_FRAMES = 300


def level(path: Path) -> tuple[np.ndarray, int]:
    a, sr = kc.load_audio(path)
    a = a * (10 ** (-26 / 20) / max(float(np.sqrt(np.mean(a ** 2))), 1e-6))
    return (a * min(1.0, 0.5 / max(float(np.abs(a).max()), 1e-6))).astype(np.float32), sr


def main() -> None:
    ap = kc.standard_argparser(FAMILY, default_files=6000, default_seed=3101)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel

    snap = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
    tts = Qwen3TTSModel.from_pretrained(str(snap), device_map="cuda:0", dtype=torch.bfloat16,
                                        attn_implementation="sdpa")
    utts = kc.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, snap.name, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do")
    for b in range(0, len(todo), args.batch):
        if kc.should_stop(args, w):
            break
        batch = todo[b:b + args.batch]
        rng = random.Random(batch[0].seed)
        torch.manual_seed(batch[0].seed)
        try:
            wavs, sr = tts.generate_voice_clone(
                text=[j.text for j in batch], language=["Korean"] * len(batch),
                ref_audio=[level(j.prompt_utts[0].abs_path) for j in batch],
                ref_text=[j.prompt_utts[0].text for j in batch],
                do_sample=True, temperature=rng.uniform(0.75, 0.95), top_p=rng.uniform(0.85, 1.0),
                max_new_tokens=MAX_FRAMES)
        except Exception as e:  # noqa: BLE001
            kc.log(f"{batch[0].name}+{len(batch)}: {type(e).__name__}: {e}")
            continue
        for j, wav in zip(batch, wavs):
            w.write(j, np.asarray(wav, dtype=np.float32), sr)
            w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
