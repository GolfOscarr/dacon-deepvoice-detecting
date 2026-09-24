"""Family ``cosyvoice``: FunAudioLLM/CosyVoice2-0.5B zero-shot cloning
(Qwen-based speech-token LM + flow matching + HiFT vocoder, 24 kHz, Apache-2.0).

venv: /data/project/private/dacon-venvs/synth-cosyvoice
  built from cosyvoice-repo/requirements.txt without deepspeed/tensorrt/gradio and
  with openai-whisper>=20240930 (the pinned 20231117 fails to build under uv).
Repo: /data/project/private/dacon-weights/synth/cosyvoice-repo (git clone --recursive).
Weights: HF FunAudioLLM/CosyVoice2-0.5B.

Prompt: one Zeroth utterance (3-10 s) with its transcript; text: 1-2 other
Zeroth sentences, synthesised one at a time with the model's text front end
disabled (it is zh/en only) and joined with a seeded pause.

  HF_HOME=... python scripts/synth/synth_cosyvoice.py --n-files 2800 --shard 0/2
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_common as sc  # noqa: E402

FAMILY = "cosyvoice"
MODEL = "FunAudioLLM/CosyVoice2-0.5B"
LICENCE = "Apache-2.0"
REPO = Path(os.environ.get("COSYVOICE_REPO", "/data/project/private/dacon-weights/synth/cosyvoice-repo"))


def main() -> None:
    ap = sc.standard_argparser(FAMILY, default_files=2800, default_seed=505)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "third_party" / "Matcha-TTS"))
    import torch
    from huggingface_hub import snapshot_download
    from cosyvoice.cli.cosyvoice import CosyVoice2

    snap = Path(snapshot_download(MODEL))
    revision = snap.name
    cv = CosyVoice2(str(snap), load_jit=False, load_trt=False, load_vllm=False, fp16=args.fp16)
    sr = cv.sample_rate

    utts = sc.build_index()
    by_id = {u.utt_id: u for u in utts}
    jobs = sc.shard_jobs(sc.make_jobs(FAMILY, args.n_files, args.seed, utts, cloning=True,
                                      prompt_min_s=3.0, prompt_max_s=10.0, n_prompt_files=1), args.shard)
    w = sc.FamilyWriter(FAMILY, MODEL, revision, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    sc.log(f"{FAMILY}: {len(jobs)} jobs in shard, {len(todo)} to do, sr={sr}")

    for j in todo:
        p = j.prompt_utts[0]
        prompt_wav = str(p.abs_path)  # the frontend loads the path itself (load_wav -> 16 kHz)
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        parts = []
        for k, uid in enumerate(j.text_utts):
            chunks = [o["tts_speech"] for o in cv.inference_zero_shot(
                by_id[uid].text, p.text, prompt_wav, stream=False,
                speed=rng.uniform(0.9, 1.1), text_frontend=False)]
            wav = torch.cat(chunks, dim=1)[0].float().cpu().numpy()
            if k:
                parts.append(np.zeros(int(sr * rng.uniform(0.2, 0.45)), np.float32))
            parts.append(sc.trim_silence(wav, sr))
        w.write(j, np.concatenate(parts), sr)
        w.progress()
        if args.limit and w.n_written >= args.limit:
            break
    print(w.summary())


if __name__ == "__main__":
    main()
