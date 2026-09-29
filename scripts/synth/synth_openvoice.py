"""Family ``openvoice``: OpenVoice V2 tone-colour conversion (MIT, 22.05 kHz) of
MeloTTS-Korean speech onto Zeroth speakers.

venv: /data/project/private/dacon-venvs/synth-openvoice
  uv pip install "git+https://github.com/myshell-ai/MeloTTS.git" soundfile python-mecab-ko "setuptools<70"
  uv pip install --no-deps "git+https://github.com/myshell-ai/OpenVoice.git"
  python -m unidic download
Weights: HF myshell-ai/OpenVoiceV2 (converter + base_speakers/ses/kr.pth), HF myshell-ai/MeloTTS-Korean.

Pipeline per file: MeloTTS KR (seeded prosody draw, own text draw — a different
seed from the ``melo`` family, so the texts differ) -> resample to 22.05 kHz ->
ToneColorConverter.voice_conversion(src_se = kr base speaker embedding,
tgt_se = mean ref-encoder embedding of 3 Zeroth utterances of the prompt
speaker). The audio watermark OpenVoice adds by default is disabled.

  HF_HOME=... python scripts/synth/synth_openvoice.py --n-files 3000 --shard 0/1
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_common as sc  # noqa: E402
from synth_melo import load_melo, melo_synth  # noqa: E402

FAMILY = "openvoice"
MODEL = "myshell-ai/OpenVoiceV2 (converter) + myshell-ai/MeloTTS-Korean (base)"
LICENCE = "MIT"


def resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, target)
    return resample_poly(audio, target // g, sr // g).astype(np.float32)


def main() -> None:
    ap = sc.standard_argparser(FAMILY, default_files=3000, default_seed=606)
    ap.add_argument("--n-prompt", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.3)
    args = ap.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from openvoice.api import ToneColorConverter, OpenVoiceBaseClass
    from openvoice.mel_processing import spectrogram_torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    snap = Path(snapshot_download("myshell-ai/OpenVoiceV2"))
    # build without the wavmark watermark model (not installed; the base class
    # of this OpenVoice build does not accept enable_watermark)
    conv = ToneColorConverter.__new__(ToneColorConverter)
    OpenVoiceBaseClass.__init__(conv, str(snap / "converter" / "config.json"), device=dev)
    conv.watermark_model = None
    conv.version = getattr(conv.hps, "_version_", "v1")
    conv.load_ckpt(str(snap / "converter" / "checkpoint.pth"))
    hps = conv.hps
    csr = hps.data.sampling_rate
    src_se = torch.load(snap / "base_speakers" / "ses" / "kr.pth", map_location=dev)
    tts, spk, msr, melo_rev = load_melo(dev)
    revision = f"OpenVoiceV2@{snap.name};MeloTTS-Korean@{melo_rev}"

    def spec_of(audio: np.ndarray) -> torch.Tensor:
        y = torch.from_numpy(audio).float().to(dev)[None]
        return spectrogram_torch(y, hps.data.filter_length, csr, hps.data.hop_length,
                                 hps.data.win_length, center=False).to(dev)

    utts = sc.build_index()
    jobs = sc.shard_jobs(sc.make_jobs(FAMILY, args.n_files, args.seed, utts, cloning=True,
                                      n_prompt_files=args.n_prompt), args.shard)
    w = sc.FamilyWriter(FAMILY, MODEL, revision, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    sc.log(f"{FAMILY}: {len(jobs)} jobs in shard, {len(todo)} to do, device={dev}, sr={csr}")

    tgt_ses: dict[str, torch.Tensor] = {}
    for j in todo:
        if j.prompt_speaker not in tgt_ses:
            gs = []
            with torch.no_grad():
                for u in j.prompt_utts:
                    a, _ = sc.load_prompt(u, csr)
                    gs.append(conv.model.ref_enc(spec_of(a).transpose(1, 2)).unsqueeze(-1))
            tgt_ses[j.prompt_speaker] = torch.stack(gs).mean(0)
        rng = random.Random(j.seed)
        base = melo_synth(tts, spk, j.text, rng)
        base = resample(base, msr, csr)
        with torch.no_grad():
            spec = spec_of(base)
            lengths = torch.LongTensor([spec.size(-1)]).to(dev)
            out = conv.model.voice_conversion(spec, lengths, sid_src=src_se, sid_tgt=tgt_ses[j.prompt_speaker],
                                              tau=args.tau)[0][0, 0].float().cpu().numpy()
        w.write(j, out, csr)
        w.progress()
        if args.limit and w.n_written >= args.limit:
            break
    print(w.summary())


if __name__ == "__main__":
    main()
