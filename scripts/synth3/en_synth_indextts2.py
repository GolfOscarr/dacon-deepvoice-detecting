"""Family ``indextts2`` (backup, 5th English family): IndexTTS2 (autoregressive GPT over semantic codes
with explicit duration control, a flow-matching semantic-to-mel stage (s2mel) conditioned on a CAMPPlus
speaker embedding, BigVGAN v2 22 kHz vocoder). Zero-shot cloning from one prompt utterance.

Weights IndexTeam/IndexTTS-2 under the "bilibili Model Use License Agreement" (royalty-free; outputs may
be used to improve only non-commercial AI models, §3.4c; no unlawful / privacy-infringing content);
auxiliaries facebook/w2v-bert-2.0 (MIT), amphion/MaskGCT semantic codec (CC-BY-NC-4.0), funasr/campplus,
nvidia/bigvgan_v2_22khz_80band_256x (MIT). Code github.com/index-tts/index-tts at
/data/project/private/dacon-weights/synth3/repos/index-tts. venv: /data/project/private/dacon-venvs/
synth3en-indextts2 (`uv sync --frozen`: torch 2.8, transformers 4.52.1).
Prompt: one real utterance (3-10 s, levelled); text: 1-2 other transcripts. Emotion comes from the
prompt itself (no emotion vector). Per-file seeded draws: temperature U(0.7, 0.9), top_p U(0.75, 0.9).

  python scripts/synth3/en_synth_indextts2.py --n-files 4000 --shard 0/1 --max-family-hours 5.6
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
FAMILY = "indextts2"
MODEL = "IndexTeam/IndexTTS-2 (GPT + s2mel + BigVGAN v2 22 kHz)"
LICENCE = "bilibili Model Use License Agreement (IndexTTS-2)"
REPO = Path(os.environ.get("INDEXTTS_REPO", "/data/project/private/dacon-weights/synth3/repos/index-tts"))


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=4000, default_seed=2505).parse_args()
    out_root = args.out_root.resolve()
    sys.path.insert(0, str(REPO))
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from indextts.infer_v2 import IndexTTS2

    snap = Path(snapshot_download("IndexTeam/IndexTTS-2"))
    aux = {"w2v_bert": snapshot_download("facebook/w2v-bert-2.0"),
           "semantic_codec": hf_hub_download("amphion/MaskGCT", "semantic_codec/model.safetensors"),
           "campplus": hf_hub_download("funasr/campplus", "campplus_cn_common.bin"),
           "bigvgan": snapshot_download("nvidia/bigvgan_v2_22khz_80band_256x")}
    tts = IndexTTS2(cfg_path=str(snap / "config.yaml"), model_dir=str(snap), use_fp16=True, device="cuda:0",
                    use_cuda_kernel=False, use_qwen_emo=False, aux_paths=aux)
    tmp = Path(tempfile.mkdtemp(prefix="idxtts-"))

    utts = ec.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, snap.name, LICENCE, out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do")
    fails = 0
    for j in todo:
        if kc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        np.random.seed(j.seed % 2 ** 32)
        try:
            ref = ec.level_to_wav(j.prompt_utts[0].abs_path, tmp / f"{j.name}_ref.wav")
            out = tmp / f"{j.name}.wav"
            tts.infer(spk_audio_prompt=str(ref), text=j.text, output_path=str(out), verbose=False,
                      temperature=rng.uniform(0.7, 0.9), top_p=rng.uniform(0.75, 0.9))
            wav, sr = sf.read(str(out), dtype="float32")
            ref.unlink(missing_ok=True)
            out.unlink(missing_ok=True)
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
