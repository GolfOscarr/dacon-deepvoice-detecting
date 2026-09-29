"""Family ``higgs`` (round 3, docs/training/15 N1-ko extension): Higgs Audio v2 zero-shot cloning
(bosonai/higgs-audio-v2-generation-3B-base: a Llama-3.2-3B LM with a "DualFFN" audio adapter
emitting the 8 RVQ codebooks of Boson's own 25 Hz semantic+acoustic audio tokenizer in a delay
pattern, decoded to 24 kHz). The card lists Korean among its languages.
Licence: Boson Higgs Audio 2 Community License (built on the Meta Llama 3 Community License):
naming clause for distributed models trained with its outputs ("Higgs Audio 2" prefix), the Llama
AUP, and no use of outputs to improve another LLM.

venv: /data/project/private/dacon-venvs/synth3-higgs (python 3.10, torch 2.6.0, transformers 4.46,
the repo boson-ai/higgs-audio installed -e). Weights under HF_HOME=/data/project/private/dacon-weights/synth3/hf.
Clone prompt = the repo's generation.py pattern: system scene prompt, user = the prompt transcript,
assistant = the prompt audio, user = the text. Prompts are levelled (-26 dBFS RMS, peak <= 0.5).
Per-file seeded draws: temperature U(0.3, 0.7), top_p 0.95, top_k 50 (repo defaults otherwise).

  python scripts/synth3/ko_synth_higgs.py --n-files 6000 --shard 0/2 --max-family-hours 8
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth2"))
import ko_common as kc  # noqa: E402

FAMILY = "higgs"
MODEL = "bosonai/higgs-audio-v2-generation-3B-base (+ higgs-audio-v2-tokenizer)"
LICENCE = "Boson Higgs Audio 2 Community License (Llama 3 based)"
SCENE = "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
# the last revisions in the repo code's own format (2026-04 "trfms-support" moved both HF repos to
# the transformers-native HiggsAudioV2 layout, which boson_multimodal cannot load)
MODEL_REV, TOK_REV = "10840182ca4ad5d9d9113b60b9bb3c1ef1ba3f84", "9d4988fbd4ad07b4cac3a5fa462741a41810dbec"
MAX_TOKENS = 25 * 24 + 16  # 25 frames/s: at most ~24 s (QC drops > 20 s)


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=6000, default_seed=3606).parse_args()
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from boson_multimodal.data_types import AudioContent, ChatMLSample, Message
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

    snap = Path(snapshot_download("bosonai/higgs-audio-v2-generation-3B-base", revision=MODEL_REV))
    tok = Path(snapshot_download("bosonai/higgs-audio-v2-tokenizer", revision=TOK_REV))
    eng = HiggsAudioServeEngine(str(snap), str(tok), device="cuda")
    utts = kc.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, f"{snap.name[:12]}+tok {tok.name[:12]}", LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do")
    tmpd = Path(tempfile.mkdtemp(prefix="higgs-prompt-"))
    for j in todo:
        if kc.should_stop(args, w):
            break
        p = j.prompt_utts[0]
        a, psr = kc.load_audio(p.abs_path)
        a = a * (10 ** (-26 / 20) / max(float(np.sqrt(np.mean(a ** 2))), 1e-6))
        a = a * min(1.0, 0.5 / max(float(np.abs(a).max()), 1e-6))
        prompt_wav = str(tmpd / "prompt.wav")
        sf.write(prompt_wav, a, psr, subtype="PCM_16")
        rng = random.Random(j.seed)
        msgs = [Message(role="system", content=SCENE), Message(role="user", content=p.text),
                Message(role="assistant", content=AudioContent(audio_url=prompt_wav)),
                Message(role="user", content=j.text)]
        try:
            out = eng.generate(chat_ml_sample=ChatMLSample(messages=msgs), max_new_tokens=MAX_TOKENS,
                               temperature=rng.uniform(0.3, 0.7), top_p=0.95, top_k=50,
                               stop_strings=["<|end_of_text|>", "<|eot_id|>"], seed=j.seed % 2**31)
        except Exception as e:  # noqa: BLE001
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        if out.audio is None:
            kc.log(f"{j.name}: no audio")
            continue
        w.write(j, np.asarray(out.audio, dtype=np.float32), out.sampling_rate)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
