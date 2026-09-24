"""Family ``mms``: facebook/mms-tts-kor (VITS, single speaker, 16 kHz, CC-BY-NC-4.0).

venv: /data/project/private/dacon-venvs/synth-mms
  uv pip install torch torchaudio transformers soundfile numpy scipy uroman huggingface_hub

Korean text is romanised with uroman (the MMS-TTS kor tokenizer expects
romanised input, ``is_uroman`` in tokenizer_config). Per-file variation comes
from seeded noise_scale / noise_scale_w / speaking_rate draws.

  HF_HOME=/data/project/private/dacon-weights/synth/hf \
  python scripts/synth/synth_mms.py --n-files 3000 --shard 0/2
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_common as sc  # noqa: E402

FAMILY = "mms"
MODEL = "facebook/mms-tts-kor"
LICENCE = "CC-BY-NC-4.0"


def main() -> None:
    ap = sc.standard_argparser(FAMILY, default_files=3000, default_seed=101)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import torch
    from transformers import VitsModel, AutoTokenizer
    from huggingface_hub import snapshot_download
    import uroman as ur

    snap = Path(snapshot_download(MODEL))
    revision = snap.name
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(snap)
    model = VitsModel.from_pretrained(snap).to(dev).eval()
    sr = model.config.sampling_rate
    uroman = ur.Uroman()

    utts = sc.build_index()
    jobs = sc.shard_jobs(sc.make_jobs(FAMILY, args.n_files, args.seed, utts, cloning=False,
                                      builtin_speakers=["kor"]), args.shard)
    w = sc.FamilyWriter(FAMILY, MODEL, revision, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    sc.log(f"{FAMILY}: {len(jobs)} jobs in shard, {len(todo)} to do, device={dev}, sr={sr}")

    for b in range(0, len(todo), args.batch):
        batch = todo[b:b + args.batch]
        roman = [uroman.romanize_string(j.text, lcode="kor") for j in batch]
        enc = tok(roman, return_tensors="pt", padding=True).to(dev)
        # one variation draw per batch-leading job keeps it deterministic per shard layout
        rng = random.Random(batch[0].seed)
        model.noise_scale = 0.667 * rng.uniform(0.7, 1.3)
        model.noise_scale_duration = 0.8 * rng.uniform(0.6, 1.4)
        model.speaking_rate = rng.uniform(0.85, 1.2)
        torch.manual_seed(batch[0].seed)
        with torch.no_grad():
            out = model(**enc)
        wavs = out.waveform.float().cpu().numpy()
        lens = out.sequence_lengths.cpu().numpy() if hasattr(out, "sequence_lengths") and out.sequence_lengths is not None else [wavs.shape[1]] * len(batch)
        for j, wav, n in zip(batch, wavs, lens):
            w.write(j, wav[: int(n)], sr)
            w.progress()
            if args.limit and w.n_written >= args.limit:
                print(w.summary()); return
    print(w.summary())


if __name__ == "__main__":
    main()
