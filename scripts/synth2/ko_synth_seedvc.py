"""Family ``seedvc``: Seed-VC zero-shot voice conversion (Whisper-small content encoder + DiT
flow matching + BigVGAN v2 22 kHz; speech model DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned).

Source = a REAL Korean utterance (Emilia / Zeroth / FLEURS) of speaker A; reference = one
utterance (3-10 s) of a different speaker B. The output is B's voice speaking A's words, which
GENERATES a new voice -> FAKE. `text` = the source transcript, `prompt_speaker` = B.

Code: HF Space Plachta/Seed-VC (the GitHub repo Plachta/seed-vc is gone, 404), snapshot at
/data/project/private/dacon-weights/synth2/repos/seed-vc, GPL-3.0 (covers the code, not our
output). Weights: HF Plachta/Seed-VC (GPL-3.0 per the model card), openai/whisper-small (Apache-2.0),
funasr/campplus, nvidia/bigvgan_v2_22khz_80band_256x (MIT).
venv: /data/project/private/dacon-venvs/synth2-seedvc (torch 2.6.0, transformers 4.46.3).
Source and reference are levelled to -26 dBFS RMS (peak <= 0.5) first (see _level).
Per-file seeded draws: diffusion steps {25..30}, inference_cfg_rate U(0.5, 0.8), length_adjust U(0.95, 1.05).

  python scripts/synth2/ko_synth_seedvc.py --n-files 6000 --shard 0/2 --max-hours 4.5
"""
from __future__ import annotations

import os
import random
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ko_common as kc  # noqa: E402

FAMILY = "seedvc"
MODEL = "Plachta/Seed-VC (DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned, 22.05 kHz)"
LICENCE = "GPL-3.0 (code and weights, model card); outputs not covered"
REPO = Path(os.environ.get("SEEDVC_REPO", "/data/project/private/dacon-weights/synth2/repos/seed-vc"))
SPACE_SHA = "84a7891703108b9d1167cd38c31906d070386f25"


def _level(path, rms_db: float = -26.0, max_peak: float = 0.5):
    """Source/reference as an in-memory wav at a fixed level: BigVGAN clamps to +-1 and the
    output level follows the input mel, so loud Emilia sources came out flat-topped (smoke:
    1.6 % samples at full scale). librosa.load accepts the file-like object."""
    import io
    import numpy as np
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    rms = float(np.sqrt(np.mean(a ** 2))) or 1e-6
    g = min(10 ** (rms_db / 20) / rms, max_peak / max(float(np.abs(a).max()), 1e-6))
    buf = io.BytesIO()
    sf.write(buf, a * g, sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=6000, default_seed=1606).parse_args()
    sys.path.insert(0, str(REPO))
    sys.modules["spaces"] = types.SimpleNamespace(GPU=lambda f: f)  # HF-Space decorator, no-op here
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    import hf_utils

    def _load(repo_id, model_filename="pytorch_model.bin", config_filename=None):
        m = hf_hub_download(repo_id, model_filename)   # normal HF cache, not ./checkpoints
        return m if config_filename is None else (m, hf_hub_download(repo_id, config_filename))
    hf_utils.load_custom_model_from_hf = _load
    import seed_vc_wrapper as svw
    svw.load_custom_model_from_hf = _load

    class Wrapper(svw.SeedVCWrapper):
        def __init__(self, device):
            self.device = device
            self._load_base_model()
            from modules.campplus.DTDNN import CAMPPlus
            from modules.bigvgan import bigvgan
            self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
            self.campplus_model.load_state_dict(torch.load(_load("funasr/campplus", "campplus_cn_common.bin"), map_location="cpu"))
            self.campplus_model.eval().to(device)
            self.bigvgan_model = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)
            self.bigvgan_model.remove_weight_norm()
            self.bigvgan_model = self.bigvgan_model.eval().to(device)
            self.model_f0 = self.to_mel_f0 = self.bigvgan_44k_model = None
            self.overlap_frame_len = 16
            self.bitrate = "320k"

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snap = Path(snapshot_download("Plachta/Seed-VC", allow_patterns=["config_dit_mel_seed_uvit_whisper_small_wavenet.yml"]))
    vc = Wrapper(dev)
    sr = 22050

    utts = kc.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts, vc=True), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, f"{snap.name} (code space {SPACE_SHA[:7]})", LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do, sr={sr}, dev={dev}")
    for j in todo:
        if kc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        np.random.seed(j.seed % 2**32)
        try:
            gen = vc.convert_voice(_level(j.src_utts[0].abs_path), _level(j.prompt_utts[0].abs_path),
                                   diffusion_steps=int(os.environ.get("SEEDVC_DEBUG_STEPS", 0)) or rng.randint(25, 30),
                                   length_adjust=rng.uniform(0.95, 1.05),
                                   inference_cfg_rate=rng.uniform(0.5, 0.8),
                                   f0_condition=False, stream_output=False)
            try:
                while True:
                    next(gen)
            except StopIteration as e:  # the wrapper is a generator; stream_output=False returns
                wav = e.value
        except Exception as e:  # noqa: BLE001
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        if wav is None:
            kc.log(f"{j.name}: no audio")
            continue
        w.write(j, np.asarray(wav, dtype=np.float32), sr)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
