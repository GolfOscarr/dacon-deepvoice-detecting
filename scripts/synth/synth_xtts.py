"""Family ``xtts``: coqui/XTTS-v2 zero-shot cloning (GPT codec LM + HiFi-GAN decoder, 24 kHz).

Licence: Coqui Public Model License 1.0.0 (CPML) — non-commercial only.
venv: /data/project/private/dacon-venvs/synth-xtts
  uv pip install coqui-tts torch torchaudio torchcodec "transformers>=4.52,<5" soundfile

Each job clones a Zeroth speaker from 3 of that speaker's utterances (≤ 30 s of
reference) and speaks 1-2 Zeroth sentences from *other* utterances. Sentences
are synthesised one at a time (XTTS's Korean character limit is 95) and joined
with a short seeded pause.

  HF_HOME=... python scripts/synth/synth_xtts.py --n-files 2800 --shard 0/2
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_common as sc  # noqa: E402

FAMILY = "xtts"
MODEL = "coqui/XTTS-v2"
LICENCE = "CPML-1.0.0 (Coqui Public Model License, non-commercial)"
SR = 24000


def main() -> None:
    ap = sc.standard_argparser(FAMILY, default_files=2800, default_seed=303)
    ap.add_argument("--n-prompt", type=int, default=3)
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    snap = Path(snapshot_download(MODEL))
    revision = snap.name
    cfg = XttsConfig()
    cfg.load_json(str(snap / "config.json"))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=str(snap), eval=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    assert cfg.audio.output_sample_rate == SR

    utts = sc.build_index()
    by_id = {u.utt_id: u for u in utts}
    jobs = sc.shard_jobs(sc.make_jobs(FAMILY, args.n_files, args.seed, utts, cloning=True,
                                      n_prompt_files=args.n_prompt), args.shard)
    w = sc.FamilyWriter(FAMILY, MODEL, revision, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    sc.log(f"{FAMILY}: {len(jobs)} jobs in shard, {len(todo)} to do, device={dev}")

    latents: dict[str, tuple] = {}
    for j in todo:
        if j.prompt_speaker not in latents:
            # conditioning from the raw 16 kHz flacs; XTTS resamples internally
            latents[j.prompt_speaker] = model.get_conditioning_latents(
                audio_path=[str(u.abs_path) for u in j.prompt_utts], max_ref_length=30,
                gpt_cond_len=30, gpt_cond_chunk_len=6)
        gpt_cond, spk_emb = latents[j.prompt_speaker]
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        parts = []
        for k, uid in enumerate(j.text_utts):
            out = model.inference(by_id[uid].text, "ko", gpt_cond, spk_emb,
                                  temperature=rng.uniform(0.6, 0.85), top_p=0.85,
                                  repetition_penalty=5.0, length_penalty=1.0, speed=rng.uniform(0.9, 1.1))
            wav = np.asarray(out["wav"], dtype=np.float32)
            if k:
                parts.append(np.zeros(int(SR * rng.uniform(0.2, 0.45)), np.float32))
            parts.append(sc.trim_silence(wav, SR))
        w.write(j, np.concatenate(parts), SR)
        w.progress()
        if args.limit and w.n_written >= args.limit:
            break
    print(w.summary())


if __name__ == "__main__":
    main()
