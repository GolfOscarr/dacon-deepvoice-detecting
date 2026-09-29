"""Family ``f5tts``: F5-TTS v1 Base (non-autoregressive flow matching over mel with a DiT backbone,
ConvNeXt text encoder, Vocos 24 kHz vocoder; no codec, no duration model), zero-shot cloning.

Weights SWivid/F5-TTS F5TTS_v1_Base/model_1250000.safetensors (CC-BY-NC-4.0) + charactr/vocos-mel-24khz
(MIT); code github.com/SWivid/F5-TTS (MIT) installed from /data/project/private/dacon-weights/synth3/repos/F5-TTS.
venv: /data/project/private/dacon-venvs/synth3en-f5tts (torch 2.5.1).
Prompt: one real utterance (3-10 s, levelled to -26 dBFS RMS, peak <= 0.5) + its transcript; text: 1-2
other transcripts. Per-file seeded draws: nfe_step {24..32}, cfg_strength U(1.6, 2.4), speed U(0.9, 1.1).

  python scripts/synth3/en_synth_f5tts.py --n-files 4000 --shard 0/1 --max-family-hours 5.6
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
FAMILY = "f5tts"
MODEL = "SWivid/F5-TTS F5TTS_v1_Base (+ charactr/vocos-mel-24khz)"
LICENCE = "CC-BY-NC-4.0 (weights); code MIT"


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=4000, default_seed=2101).parse_args()
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from f5_tts.api import F5TTS

    snap = Path(snapshot_download("SWivid/F5-TTS", allow_patterns=["F5TTS_v1_Base/*"]))
    voc = snapshot_download("charactr/vocos-mel-24khz")
    tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors"),
                vocoder_local_path=voc, device="cuda")
    tmp = Path(tempfile.mkdtemp(prefix="f5ref-"))

    utts = ec.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, snap.name, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do")
    fails = 0
    for j in todo:
        if kc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        p = j.prompt_utts[0]
        try:
            ref = ec.level_to_wav(p.abs_path, tmp / f"{j.name}.wav")
            wav, sr, _ = tts.infer(str(ref), p.text, j.text, show_info=lambda *a: None, progress=None,
                                   nfe_step=rng.randint(24, 32), cfg_strength=rng.uniform(1.6, 2.4),
                                   speed=rng.uniform(0.9, 1.1), seed=j.seed % (2 ** 31))
            ref.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            if os.environ.get("EN3_DEBUG"):
                raise
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            fails += 1
            if fails >= 25:
                sys.exit(f"{fails} consecutive failures, giving up")
            continue
        w.write(j, np.asarray(wav, dtype=np.float32), sr)
        fails = 0
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
