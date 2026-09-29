"""Family ``sparktts``: Spark-TTS 0.5B (Qwen2.5-0.5B LLM that predicts single-stream BiCodec semantic
tokens, 50 tokens/s, conditioned on BiCodec global speaker tokens; BiCodec decoder at 16 kHz). Zero-shot
cloning with the prompt transcript (the LLM continues the prompt's semantic tokens).

Weights SparkAudio/Spark-TTS-0.5B (CC-BY-NC-SA-4.0, model card); code github.com/SparkAudio/Spark-TTS
(Apache-2.0) at /data/project/private/dacon-weights/synth3/repos/Spark-TTS (imported from the checkout).
venv: /data/project/private/dacon-venvs/synth3en-sparktts (the repo's requirements: torch 2.5.1,
transformers 4.46.2).
Prompt: one real utterance (3-10 s, levelled); text: 1-2 other transcripts. Per-file seeded draws:
temperature U(0.7, 0.95), top_p U(0.85, 0.95), top_k 50.

  python scripts/synth3/en_synth_sparktts.py --n-files 4000 --shard 0/1 --max-family-hours 5.6
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import en_common as ec  # noqa: E402

kc = ec.kc
FAMILY = "sparktts"
MODEL = "SparkAudio/Spark-TTS-0.5B (Qwen2.5-0.5B + BiCodec)"
LICENCE = "CC-BY-NC-SA-4.0 (weights); code Apache-2.0"
REPO = Path(os.environ.get("SPARKTTS_REPO", "/data/project/private/dacon-weights/synth3/repos/Spark-TTS"))


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=4000, default_seed=2303).parse_args()
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from cli.SparkTTS import SparkTTS

    snap = Path(snapshot_download("SparkAudio/Spark-TTS-0.5B"))
    tts = SparkTTS(snap, torch.device("cuda:0"))
    sr = tts.sample_rate
    tmp = Path(tempfile.mkdtemp(prefix="sparkref-"))

    utts = ec.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, snap.name, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do, sr={sr}")
    fails = 0
    for j in todo:
        if kc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        np.random.seed(j.seed % 2 ** 32)
        p = j.prompt_utts[0]
        try:
            ref = ec.level_to_wav(p.abs_path, tmp / f"{j.name}.wav", 16000)
            wav = tts.inference(j.text, ref, prompt_text=p.text, temperature=rng.uniform(0.7, 0.95),
                                top_k=50, top_p=rng.uniform(0.85, 0.95))
            ref.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            if os.environ.get("EN3_DEBUG"):
                raise
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            fails += 1
            if fails >= 25:
                sys.exit(f"{fails} consecutive failures, giving up")
            if "CUDA error" in str(e) or "device-side assert" in str(e):
                sys.exit(3)
            continue
        w.write(j, np.asarray(wav, dtype=np.float32), sr)
        fails = 0
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
