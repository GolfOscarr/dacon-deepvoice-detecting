"""Family ``f5tts`` (round 3, docs/training/15 N1-ko): F5-TTS (non-autoregressive DiT conditional
flow matching over 100-band mels, ConvNeXt text encoder, Vocos 24 kHz vocoder), Korean model
team-lucid/F5-TTS-ko (F5TTS_Base shape: dim 1024, depth 22; HF tag apache-2.0, empty card).
Its vocabulary is IPA + Hangul conjoining jamo, so text and prompt transcript are NFD-decomposed
(syllables -> U+1100/U+1161/U+11A8 jamo) before tokenising.

venv: /data/project/private/dacon-venvs/synth3-f5tts (python 3.11, torch 2.6.0, pip f5-tts).
Weights: the HF snapshot under HF_HOME=/data/project/private/dacon-weights/synth3/hf, re-saved for
the f5_tts loader as /data/project/private/dacon-weights/synth3/f5tts-ko/{model_90ade5f.pt,vocab.txt}
(the same tensors; vocab.json -> vocab.txt in index order). Vocoder: charactr/vocos-mel-24khz (MIT).
Prompts are levelled (-26 dBFS RMS, peak <= 0.5). Per-file seeded draws: nfe_step {24..32},
cfg_strength U(1.8, 2.4), speed U(0.9, 1.1); sway sampling -1 (package default).

  python scripts/synth3/ko_synth_f5tts.py --n-files 6000 --shard 0/2 --max-family-hours 9
"""
from __future__ import annotations

import random
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth2"))
import ko_common as kc  # noqa: E402

FAMILY = "f5tts"
MODEL = "team-lucid/F5-TTS-ko (F5TTS_Base DiT + charactr/vocos-mel-24khz)"
LICENCE = "Apache-2.0 (HF tag; card empty); f5-tts code MIT; Vocos MIT"
W = Path("/data/project/private/dacon-weights/synth3/f5tts-ko")
REV = "90ade5fb63434ddd74c46dae74adfbf008dff97e"


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=6000, default_seed=3505).parse_args()
    import soundfile as sf
    import torch
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
    from f5_tts.model import DiT

    cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    model = load_model(DiT, cfg, str(W / "model_90ade5f.pt"), vocab_file=str(W / "vocab.txt"), use_ema=False)
    voc = load_vocoder("vocos")
    utts = kc.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, REV, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do")
    tmpd = Path(tempfile.mkdtemp(prefix="f5-prompt-"))
    quiet = lambda *a, **k: None  # noqa: E731
    for j in todo:
        if kc.should_stop(args, w):
            break
        p = j.prompt_utts[0]
        a, psr = kc.load_audio(p.abs_path)
        a = a * (10 ** (-26 / 20) / max(float(np.sqrt(np.mean(a ** 2))), 1e-6))
        a = a * min(1.0, 0.5 / max(float(np.abs(a).max()), 1e-6))
        prompt_wav = str(tmpd / f"{j.name}.wav")
        sf.write(prompt_wav, a, psr, subtype="PCM_16")
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        try:
            ref_audio, ref_text = preprocess_ref_audio_text(prompt_wav, unicodedata.normalize("NFD", p.text), show_info=quiet)
            wav, sr, _ = infer_process(ref_audio, ref_text, unicodedata.normalize("NFD", j.text), model, voc,
                                       show_info=quiet, progress=None, nfe_step=rng.randint(24, 32),
                                       cfg_strength=rng.uniform(1.8, 2.4), speed=rng.uniform(0.9, 1.1))
        except Exception as e:  # noqa: BLE001
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        finally:
            Path(prompt_wav).unlink(missing_ok=True)
        w.write(j, np.asarray(wav, dtype=np.float32), sr)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
