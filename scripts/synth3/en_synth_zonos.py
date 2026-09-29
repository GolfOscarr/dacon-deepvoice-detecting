"""Family ``zonos``: Zonos v0.1 transformer (autoregressive transformer over the 9-codebook DAC 44.1 kHz
codec with delay pattern; text as eSpeak phonemes; speaker = a ResNet speaker embedding + LDA of the
prompt, plus conditioning on pitch spread, speaking rate and emotion). Zero-shot cloning.

Weights Zyphra/Zonos-v0.1-transformer + Zyphra/Zonos-v0.1-speaker-embedding (Apache-2.0),
descript/dac_44khz (DAC, MIT); code github.com/Zyphra/Zonos (Apache-2.0) at
/data/project/private/dacon-weights/synth3/repos/Zonos (installed editable).
venv: /data/project/private/dacon-venvs/synth3en-zonos (torch 2.5.1); espeak-ng from espeakng-loader
(the node has no system espeak-ng).
Prompt: one real utterance (3-10 s, levelled); text: 1-2 other transcripts. Per-file seeded draws:
speaking_rate U(12, 17), pitch_std U(20, 50), cfg_scale U(1.8, 2.5).

  python scripts/synth3/en_synth_zonos.py --n-files 4000 --shard 0/1 --max-family-hours 5.6
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import en_common as ec  # noqa: E402

kc = ec.kc
FAMILY = "zonos"
MODEL = "Zyphra/Zonos-v0.1-transformer (+ Zonos-v0.1-speaker-embedding, descript/dac_44khz)"
LICENCE = "Apache-2.0"


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=4000, default_seed=2202).parse_args()
    import espeakng_loader
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
    os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from zonos.conditioning import make_cond_dict
    from zonos.model import Zonos

    snap = Path(snapshot_download("Zyphra/Zonos-v0.1-transformer"))
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")
    model.requires_grad_(False).eval()
    # SpeakerEmbeddingLDA builds its torchaudio mel filterbank inside `with torch.device("cuda")`,
    # which fails in torchaudio 2.5.1 (cpu/cuda mix); build it on CPU and move it
    from zonos.speaker_cloning import SpeakerEmbeddingLDA
    spk_model = SpeakerEmbeddingLDA(device="cpu").to("cuda")
    spk_model.device = spk_model.model.device = "cuda"
    model.spk_clone_model = spk_model
    sr = model.autoencoder.sampling_rate

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
            a, psr = ec.level(p.abs_path)
            with torch.no_grad():
                spk = model.make_speaker_embedding(torch.from_numpy(a).unsqueeze(0), psr)
                cond = make_cond_dict(text=j.text, speaker=spk, language="en-us",
                                      speaking_rate=rng.uniform(12, 17), pitch_std=rng.uniform(20, 50))
                # ~86 codec frames per second; allow 2x a 12-chars-per-second reading, <= 30 s
                max_tok = int(min(30.0, max(4.0, len(j.text) / 12 * 2.0)) * 86)
                codes = model.generate(model.prepare_conditioning(cond), max_new_tokens=max_tok,
                                       cfg_scale=rng.uniform(1.8, 2.5), progress_bar=False)
                wav = model.autoencoder.decode(codes).float().cpu().numpy()[0]
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
        w.write(j, wav, sr)
        fails = 0
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
