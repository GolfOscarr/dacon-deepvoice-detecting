"""Family ``gptsovits`` (round 3, docs/training/15 N1-ko): GPT-SoVITS v2ProPlus zero-shot cloning
with the released base weights, no per-speaker fine-tune (the webui's "no training" mode): an AR
GPT (s1v3) over 25 Hz chinese-hubert semantic tokens + a SoVITS (VITS) decoder conditioned on the
reference spectrogram and an ERes2NetV2 speaker vector, 32 kHz. Korean front end: the repo's
korean.py (g2pk2 + python-mecab-ko), text_lang = prompt_lang = "ko".
Licence: MIT (code github.com/RVC-Boss/GPT-SoVITS and weights HF lj1995/GPT-SoVITS).

venv: /data/project/private/dacon-venvs/synth3-gptsovits (python 3.10, torch 2.5.1 cu124).
Repo: /data/project/private/dacon-weights/synth3/repos/GPT-SoVITS (pretrained_models/* are
symlinks into the HF snapshot under /data/project/private/dacon-weights/synth3/hf).
Prompts are levelled (-26 dBFS RMS, peak <= 0.5); GPT-SoVITS requires a 3-10 s reference.
Per-file seeded draws: top_k {10..20}, temperature U(0.9, 1), speed U(0.95, 1.05) (top_p 1).

  python scripts/synth3/ko_synth_gptsovits.py --n-files 6000 --shard 0/2 --max-family-hours 9
"""
from __future__ import annotations

import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth2"))
import ko_common as kc  # noqa: E402

FAMILY = "gptsovits"
MODEL = "GPT-SoVITS v2ProPlus (lj1995/GPT-SoVITS s1v3.ckpt + v2Pro/s2Gv2ProPlus.pth), zero-shot"
LICENCE = "MIT (code and weights)"
REPO = Path(os.environ.get("GPTSOVITS_REPO", "/data/project/private/dacon-weights/synth3/repos/GPT-SoVITS"))
CODE_SHA = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
VERSIONS = {  # version -> (GPT, SoVITS) under pretrained_models/
    "v2ProPlus": ("s1v3.ckpt", "v2Pro/s2Gv2ProPlus.pth"),
    "v2": ("gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
           "gsv-v2final-pretrained/s2G2333k.pth"),
}


def main() -> None:
    ap = kc.standard_argparser(FAMILY, default_files=6000, default_seed=3404)
    ap.add_argument("--version", default="v2ProPlus", choices=list(VERSIONS))
    args = ap.parse_args()
    args.out_root = args.out_root.resolve()
    os.chdir(REPO)   # the config's weight paths are relative to the repo root
    sys.path[:0] = [str(REPO), str(REPO / "GPT_SoVITS")]
    import soundfile as sf
    import torch
    from TTS_infer_pack.TTS import TTS, TTS_Config

    pm = "GPT_SoVITS/pretrained_models"
    snap = Path(os.path.realpath(f"{pm}/s1v3.ckpt")).parent.name
    model = MODEL.replace("v2ProPlus (lj1995/GPT-SoVITS s1v3.ckpt + v2Pro/s2Gv2ProPlus.pth)",
                          f"{args.version} (lj1995/GPT-SoVITS {' + '.join(VERSIONS[args.version])})")
    cfg = TTS_Config({"custom": {
        "device": "cuda" if torch.cuda.is_available() else "cpu", "is_half": torch.cuda.is_available(),
        "version": args.version, "t2s_weights_path": f"{pm}/{VERSIONS[args.version][0]}",
        "vits_weights_path": f"{pm}/{VERSIONS[args.version][1]}",
        "bert_base_path": f"{pm}/chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": f"{pm}/chinese-hubert-base"}})
    tts = TTS(cfg)
    utts = kc.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, model, f"{snap} (code {CODE_SHA[:7]})", LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do")
    tmpd = Path(tempfile.mkdtemp(prefix="gsv-prompt-"))
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
        try:
            parts, sr = [], 0
            for sr, chunk in tts.run({
                    "text": j.text, "text_lang": "ko", "ref_audio_path": prompt_wav,
                    "prompt_text": p.text, "prompt_lang": "ko", "top_k": rng.randint(10, 20),
                    "top_p": 1.0, "temperature": rng.uniform(0.9, 1.0),
                    "text_split_method": "cut5", "batch_size": 1, "speed_factor": rng.uniform(0.95, 1.05),
                    "seed": j.seed % 2**31, "parallel_infer": True, "repetition_penalty": 1.35}):
                parts.append(np.asarray(chunk))
            wav = np.concatenate(parts).astype(np.float32)
            if np.issubdtype(parts[0].dtype, np.integer):
                wav /= 32768.0
        except Exception as e:  # noqa: BLE001
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        w.write(j, wav, sr)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
