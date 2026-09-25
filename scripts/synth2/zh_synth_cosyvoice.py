"""zh family ``cosyvoice2``: FunAudioLLM/CosyVoice2-0.5B zero-shot cloning in Mandarin
(Qwen speech-token LM + flow matching + HiFT, 24 kHz, Apache-2.0).

Round 1's venv and repo (scripts/synth/synth_cosyvoice.py), read-only:
  venv  /data/project/private/dacon-venvs/synth-cosyvoice
  repo  /data/project/private/dacon-weights/synth/cosyvoice-repo @ 074ca6d
  HF_HOME /data/project/private/dacon-weights/synth/hf
Text front end ON: CosyVoice's own zh path (bracket removal, sentence split) with
WeTextProcessing zh TN. The venv's wetext 0.0.4 fetches its FSTs from ModelScope at start-up,
which fails here (auth), so the FSTs were downloaded to dacon-weights/synth2/wetext and the
normaliser is attached after construction.

Prompt: one zh REAL utterance (3-10 s) with its Whisper large-v3 transcript
(_index/prompt_transcripts.csv); text: 1-2 FLEURS sentences (never the prompt's).

  python scripts/synth2/zh_synth_cosyvoice.py --n-files 4000 --max-hours 7.2
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zh_common as zc  # noqa: E402

FAMILY = "cosyvoice2"
MODEL = "FunAudioLLM/CosyVoice2-0.5B"
LICENCE = "Apache-2.0"
REPO = Path(os.environ.get("COSYVOICE_REPO", "/data/project/private/dacon-weights/synth/cosyvoice-repo"))
WETEXT = Path("/data/project/private/dacon-weights/synth2/wetext")


def main() -> None:
    ap = zc.standard_argparser(FAMILY, default_files=4000, default_seed=515)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "third_party" / "Matcha-TTS"))
    import torch
    from huggingface_hub import snapshot_download
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from wetext import Normalizer

    snap = Path(snapshot_download(MODEL))
    cv = CosyVoice2(str(snap), load_jit=False, load_trt=False, load_vllm=False, fp16=args.fp16)
    fe = cv.frontend
    fe.zh_tn_model = Normalizer(tagger_path=str(WETEXT / "zh/tn/tagger.fst"),
                                verbalizer_path=str(WETEXT / "zh/tn/verbalizer.fst"), lang="zh", operator="tn")
    fe.en_tn_model = Normalizer(tagger_path=str(WETEXT / "en/tn/tagger.fst"),
                                verbalizer_path=str(WETEXT / "en/tn/verbalizer.fst"), lang="en", operator="tn")
    fe.text_frontend = "wetext"
    sr = cv.sample_rate

    prompts = zc.load_prompts(with_text=True)
    jobs = zc.shard_jobs(zc.make_jobs(FAMILY, args.n_files, args.seed, prompts), args.shard)
    w = zc.FamilyWriter(FAMILY, MODEL, snap.name, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    zc.log(f"{FAMILY}: {len(prompts)} prompts, {len(jobs)} jobs in shard, {len(todo)} to do, sr={sr}")

    for j in todo:
        if zc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        try:
            chunks = [o["tts_speech"] for o in cv.inference_zero_shot(
                j.text, j.prompt.text, str(j.prompt.abs_path), stream=False,
                speed=rng.uniform(0.9, 1.1), text_frontend=True)]
        except Exception as e:  # one bad job must not stop the shard
            zc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        parts = []
        for k, c in enumerate(chunks):
            if k:
                parts.append(np.zeros(int(sr * rng.uniform(0.15, 0.35)), np.float32))
            parts.append(zc.trim_silence(c[0].float().cpu().numpy(), sr))
        w.write(j, np.concatenate(parts), sr)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
