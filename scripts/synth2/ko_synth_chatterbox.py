"""Family ``chatterbox``: ResembleAI/chatterbox multilingual (T3 0.5B Llama speech-token LM +
S3Gen flow matching + HiFT, 24 kHz, MIT), zero-shot cloning with language_id="ko" (the
package's own Korean normaliser).

venv: /data/project/private/dacon-venvs/synth2-chatterbox (chatterbox-tts 0.1.7).
Weights: HF ResembleAI/chatterbox under HF_HOME=/data/project/private/dacon-weights/synth2/hf.
The Perth implicit watermark is replaced by a no-op (as round 1 disabled OpenVoice's wavmark),
so a watermark cannot become a shortcut feature for the detector.

Prompt: one real utterance (3-10 s, Emilia/Zeroth/FLEURS); text: 1-2 other transcripts.
Per-file seeded draws: exaggeration U(0.35, 0.7), cfg_weight U(0.3, 0.6), temperature U(0.7, 0.9).

  python scripts/synth2/ko_synth_chatterbox.py --n-files 6000 --shard 0/2 --max-hours 4.5
English: SYNTH2_LANG=en (language_id="en", output interim/en-synth2/chatterbox/).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ko_common as kc  # noqa: E402

FAMILY = "chatterbox"
MODEL = "ResembleAI/chatterbox (multilingual t3_mtl23ls_v2)"
LICENCE = "MIT"


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=6000, default_seed=1101).parse_args()
    import perth
    import torch
    from huggingface_hub import snapshot_download

    class _NoWatermark:
        def apply_watermark(self, wav, sample_rate):
            return wav
    perth.PerthImplicitWatermarker = _NoWatermark
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    snap = Path(snapshot_download("ResembleAI/chatterbox", allow_patterns=["ve.pt"]))
    tts = ChatterboxMultilingualTTS.from_pretrained(device=dev)
    tts.watermarker = _NoWatermark()
    sr = tts.sr

    utts = kc.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, snap.name, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do, sr={sr}, dev={dev}")
    for j in todo:
        if kc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        try:
            wav = tts.generate(j.text, language_id=kc.LANG, audio_prompt_path=str(j.prompt_utts[0].abs_path),
                               exaggeration=rng.uniform(0.35, 0.7), cfg_weight=rng.uniform(0.3, 0.6),
                               temperature=rng.uniform(0.7, 0.9))
        except Exception as e:  # noqa: BLE001
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        w.write(j, wav[0].float().cpu().numpy(), sr)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
