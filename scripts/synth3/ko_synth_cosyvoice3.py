"""Family ``cosyvoice3`` (round 3, docs/training/15 N1-ko): FunAudioLLM/Fun-CosyVoice3-0.5B-2512
zero-shot cloning (CosyVoice 3: Qwen speech-token LM with the v3 supervised multi-task speech
tokenizer + DiT flow matching + HiFT/causal vocoder, 24 kHz, Apache-2.0; Korean is one of its 9
languages). Same draw and levelling as round 2's cosyvoice (CosyVoice2), which it succeeds.

venv: /data/project/private/dacon-venvs/synth-cosyvoice (round 1's, read-only). Repo: round 1's
/data/project/private/dacon-weights/synth/cosyvoice-repo (@074ca6d = upstream HEAD, which ships
CosyVoice3). Weights under HF_HOME=/data/project/private/dacon-weights/synth3/hf.
Prompt text is "You are a helpful assistant.<|endofprompt|>" + the prompt transcript (model card).
Output: interim/ko-synth3/cosyvoice3/ (KO_SYNTH2_ROOT, set by scripts/synth3/ko_run.sbatch).

  python scripts/synth3/ko_synth_cosyvoice3.py --n-files 6000 --shard 0/2 --max-family-hours 9
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth2"))
import ko_common as kc  # noqa: E402

FAMILY = "cosyvoice3"
MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
LICENCE = "Apache-2.0"
PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
REPO = Path(os.environ.get("COSYVOICE_REPO", "/data/project/private/dacon-weights/synth/cosyvoice-repo"))


def main() -> None:
    ap = kc.standard_argparser(FAMILY, default_files=6000, default_seed=3303)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "third_party" / "Matcha-TTS"))
    import torch
    from huggingface_hub import snapshot_download
    from cosyvoice.cli.cosyvoice import CosyVoice3

    snap = Path(snapshot_download(MODEL))
    cv = CosyVoice3(str(snap), load_trt=False, load_vllm=False, fp16=args.fp16)
    sr = cv.sample_rate
    utts = kc.load_pool()
    by_id = {u.utt_id: u for u in utts}
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, snap.name, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do, sr={sr}")
    import tempfile
    import soundfile as sf
    tmpd = Path(tempfile.mkdtemp(prefix="cosy-prompt-"))
    for j in todo:
        if kc.should_stop(args, w):
            break
        p = j.prompt_utts[0]
        # level the prompt (-26 dBFS RMS, peak <= 0.5): HiFT clamps its output at +-0.99 and
        # loud Emilia prompts gave flat-topped (QC "clipped") output, 23 % of the first 160 files
        a, psr = kc.load_audio(p.abs_path)
        a = a * (10 ** (-26 / 20) / max(float(np.sqrt(np.mean(a ** 2))), 1e-6))
        a = a * min(1.0, 0.5 / max(float(np.abs(a).max()), 1e-6))
        prompt_wav = str(tmpd / "prompt.wav")
        sf.write(prompt_wav, a, psr, subtype="PCM_16")
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        try:
            parts = []
            for k, uid in enumerate(j.text_utts):
                chunks = [o["tts_speech"] for o in cv.inference_zero_shot(
                    by_id[uid].text, PROMPT_PREFIX + p.text, prompt_wav, stream=False,
                    speed=rng.uniform(0.9, 1.1), text_frontend=False)]
                wav = torch.cat(chunks, dim=1)[0].float().cpu().numpy()
                if k:
                    parts.append(np.zeros(int(sr * rng.uniform(0.2, 0.45)), np.float32))
                parts.append(kc.trim_silence(wav, sr))
        except Exception as e:  # noqa: BLE001
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        w.write(j, np.concatenate(parts), sr)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
