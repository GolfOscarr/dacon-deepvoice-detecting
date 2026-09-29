"""Family ``maskgct``: amphion/MaskGCT (non-autoregressive masked generative codec transformer:
text-to-semantic over w2v-bert-2.0 tokens + semantic-to-acoustic, 24 kHz, CC-BY-NC-4.0),
zero-shot cloning with language "ko" (Amphion's phonemizer/espeak-ng G2P).

venv: /data/project/private/dacon-venvs/synth2-maskgct (torch 2.5.1, transformers 4.46.3,
phonemizer + espeakng-loader for the espeak-ng library/data the node lacks, onnxruntime).
Repo: /data/project/private/dacon-weights/synth2/repos/Amphion (the script chdirs into it:
configs and w2v-bert stats load by relative path). Weights amphion/MaskGCT +
facebook/w2v-bert-2.0 under HF_HOME=/data/project/private/dacon-weights/synth2/hf.
Import traps handled here: pyopenjtalk (Japanese G2P, needs a C++ build) and LangSegment
(its PyPI releases are broken; only used for language="auto") are stubbed; Korean needs neither.

Prompt: one real utterance (3-10 s) + its transcript; text: 1-2 other transcripts. Target
length = the pipeline's rule (prompt length x target/prompt phone count) x seeded U(0.9, 1.1).

  python scripts/synth2/ko_synth_maskgct.py --n-files 6000 --shard 0/2 --max-hours 4.5
English: SYNTH2_LANG=en (language "en": the pipeline's zh/en G2P path, espeak en-us).
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ko_common as kc  # noqa: E402

FAMILY = "maskgct"
MODEL = "amphion/MaskGCT (+ facebook/w2v-bert-2.0)"
LICENCE = "CC-BY-NC-4.0"
REPO = Path(os.environ.get("AMPHION_REPO", "/data/project/private/dacon-weights/synth2/repos/Amphion"))


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=6000, default_seed=1202).parse_args()
    import espeakng_loader
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
    os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
    sys.modules["pyopenjtalk"] = types.ModuleType("pyopenjtalk")
    ls = types.ModuleType("LangSegment")
    ls.setfilters = lambda f: None
    sys.modules["LangSegment"] = ls
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import numpy as np
    import safetensors.torch
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from models.tts.maskgct import maskgct_utils as mu

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    snap = Path(snapshot_download("amphion/MaskGCT"))
    code_sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
    rev = f"{snap.name[:12]}+Amphion@{code_sha}"
    cfg = mu.load_config("./models/tts/maskgct/config/maskgct.json")
    sem, sem_mean, sem_std = mu.build_semantic_model(dev)
    sem_codec = mu.build_semantic_codec(cfg.model.semantic_codec, dev)
    enc, dec = mu.build_acoustic_codec(cfg.model.acoustic_codec, dev)
    t2s = mu.build_t2s_model(cfg.model.t2s_model, dev)
    s2a1 = mu.build_s2a_model(cfg.model.s2a_model.s2a_1layer, dev)
    s2af = mu.build_s2a_model(cfg.model.s2a_model.s2a_full, dev)
    for m, f in ((sem_codec, "semantic_codec/model.safetensors"), (enc, "acoustic_codec/model.safetensors"),
                 (dec, "acoustic_codec/model_1.safetensors"), (t2s, "t2s_model/model.safetensors"),
                 (s2a1, "s2a_model/s2a_model_1layer/model.safetensors"),
                 (s2af, "s2a_model/s2a_model_full/model.safetensors")):
        # safetensors.torch.load_model refuses the codec's shared istft.window buffer in
        # this safetensors release; load the state dict directly and allow only buffers missing
        sd = safetensors.torch.load_file(hf_hub_download("amphion/MaskGCT", filename=f))
        missing, unexpected = m.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.endswith("window")] + list(unexpected)
        assert not bad, f"{f}: state dict mismatch {bad[:5]}"
    pipe = mu.MaskGCT_Inference_Pipeline(sem, sem_codec, enc, dec, t2s, s2a1, s2af, sem_mean, sem_std, dev)
    sr = 24000

    utts = kc.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, rev, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do, dev={dev}")
    for j in todo:
        if kc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        np.random.seed(j.seed % (2 ** 32))
        p = j.prompt_utts[0]
        try:
            with torch.no_grad():
                s16, _ = kc.load_audio(p.abs_path, 16000)
                s24, _ = kc.load_audio(p.abs_path, 24000)
                pp = mu.g2p_(p.text, kc.LANG)[1]
                tp = mu.g2p_(j.text, kc.LANG)[1]
                tlen = len(s16) / 16000 * len(tp) / max(len(pp), 1) * rng.uniform(0.9, 1.1)
                tlen = min(max(tlen, 2.5), 20.0)
                comb, _ = pipe.text2semantic(s16, p.text, kc.LANG, j.text, kc.LANG, tlen)
                ac = pipe.extract_acoustic_code(torch.tensor(s24).unsqueeze(0).to(dev))
                _, wav = pipe.semantic2acoustic(comb, ac)
        except Exception as e:  # noqa: BLE001
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        w.write(j, wav, sr)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
