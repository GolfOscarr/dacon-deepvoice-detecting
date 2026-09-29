"""Family ``styletts2``: StyleTTS 2 LibriTTS multi-speaker model (style diffusion over a 256-d style
vector + adversarially trained non-autoregressive decoder with explicit durations/F0/energy, iSTFTNet/
HiFi-GAN-style decoder at 24 kHz, PL-BERT text encoder; no codec, no LM). Zero-shot cloning from the
prompt's style vector (the model card's LibriTTS inference recipe, Demo/Inference_LibriTTS.ipynb).

Weights yl4579/StyleTTS2-LibriTTS (Models/LibriTTS/epochs_2nd_00020.pth); the repo's
pre-trained-model terms: listeners must be told the speech is synthetic unless the voice's use is
permitted; the README says the rule does not apply to reference speakers from open-access datasets
(ours are Emilia / LibriTTS-R). Code github.com/yl4579/StyleTTS2 (MIT) at
/data/project/private/dacon-weights/synth3/repos/StyleTTS2 (the script chdirs into it: Utils/ ASR,
JDC and PLBERT load by relative path). Phonemes: phonemizer + espeak-ng (espeakng-loader).
venv: /data/project/private/dacon-venvs/synth3en-styletts2 (torch 2.5.1).
Prompt: one real utterance (3-10 s, levelled, 24 kHz, trimmed at top_db 30 as in the recipe); text:
1-2 other transcripts. Per-file seeded draws: alpha U(0.1, 0.4), beta U(0.5, 0.9), diffusion_steps
{5..10}, embedding_scale U(1, 1.5).

  python scripts/synth3/en_synth_styletts2.py --n-files 4000 --shard 0/1 --max-family-hours 5.6
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import en_common as ec  # noqa: E402

kc = ec.kc
FAMILY = "styletts2"
MODEL = "yl4579/StyleTTS2-LibriTTS (epochs_2nd_00020)"
LICENCE = "code MIT; pre-trained model terms: disclose synthetic speech unless voice use is permitted"
REPO = Path(os.environ.get("STYLETTS2_REPO", "/data/project/private/dacon-weights/synth3/repos/StyleTTS2"))


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=4000, default_seed=2404).parse_args()
    import espeakng_loader
    os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path())
    os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
    out_root = args.out_root.resolve()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import librosa
    import numpy as np
    import torch
    import torchaudio
    import yaml
    import phonemizer
    from huggingface_hub import snapshot_download
    from nltk.tokenize import word_tokenize
    from models import build_model, load_ASR_models, load_F0_models
    from utils import recursive_munch
    from text_utils import TextCleaner
    from Utils.PLBERT.util import load_plbert
    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

    dev = "cuda"
    snap = Path(snapshot_download("yl4579/StyleTTS2-LibriTTS"))
    config = yaml.safe_load(open(snap / "Models/LibriTTS/config.yml"))
    text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
    pitch_extractor = load_F0_models(config["F0_path"])
    plbert = load_plbert(config["PLBERT_dir"])
    mp = recursive_munch(config["model_params"])
    model = build_model(mp, text_aligner, pitch_extractor, plbert)
    params = torch.load(snap / "Models/LibriTTS/epochs_2nd_00020.pth", map_location="cpu")["net"]
    for k in model:
        if k in params:
            sd = params[k]
            if next(iter(sd)).startswith("module."):
                sd = {n[7:]: v for n, v in sd.items()}
            model[k].load_state_dict(sd, strict=True)
    for k in model:
        model[k].eval().to(dev)
    sampler = DiffusionSampler(model.diffusion.diffusion, sampler=ADPM2Sampler(),
                               sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), clamp=False)
    phon = phonemizer.backend.EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
    cleaner = TextCleaner()
    to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    sr = 24000

    def mel(a):
        m = to_mel(torch.from_numpy(a).float())
        return ((torch.log(1e-5 + m.unsqueeze(0)) + 4) / 4).to(dev)

    def style(a):
        a, _ = librosa.effects.trim(a, top_db=30)
        m = mel(a).unsqueeze(1)
        return torch.cat([model.style_encoder(m), model.predictor_encoder(m)], dim=1)

    def synth(text, ref_s, alpha, beta, steps, emb_scale):
        ps = " ".join(word_tokenize(phon.phonemize([text.strip()])[0]))
        tokens = torch.LongTensor([0] + cleaner(ps)).to(dev).unsqueeze(0)
        n = torch.LongTensor([tokens.shape[-1]]).to(dev)
        mask = torch.gt(torch.arange(n.max()).unsqueeze(0).to(dev) + 1, n.unsqueeze(1))
        t_en = model.text_encoder(tokens, n, mask)
        bert_dur = model.bert(tokens, attention_mask=(~mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(dev), embedding=bert_dur,
                         embedding_scale=emb_scale, features=ref_s, num_steps=steps).squeeze(1)
        ref = alpha * s_pred[:, :128] + (1 - alpha) * ref_s[:, :128]
        s = beta * s_pred[:, 128:] + (1 - beta) * ref_s[:, 128:]
        d = model.predictor.text_encoder(d_en, s, n, mask)
        x, _ = model.predictor.lstm(d)
        dur = torch.round(torch.sigmoid(model.predictor.duration_proj(x)).sum(axis=-1).squeeze()).clamp(min=1)
        aln = torch.zeros(int(n), int(dur.sum()))
        c = 0
        for i in range(aln.size(0)):
            aln[i, c:c + int(dur[i])] = 1
            c += int(dur[i])
        aln = aln.unsqueeze(0).to(dev)
        en = d.transpose(-1, -2) @ aln
        asr = t_en @ aln
        if mp.decoder.type == "hifigan":
            en = torch.cat([en[:, :, :1], en[:, :, :-1]], dim=2)
            asr = torch.cat([asr[:, :, :1], asr[:, :, :-1]], dim=2)
        f0, nn_ = model.predictor.F0Ntrain(en, s)
        out = model.decoder(asr, f0, nn_, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()[..., :-50]   # the recipe drops a pulse at the end

    utts = ec.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, snap.name, LICENCE, out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do, sr={sr}")
    fails = 0
    for j in todo:
        if kc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        torch.manual_seed(j.seed)
        np.random.seed(j.seed % 2 ** 32)
        try:
            a, _ = ec.level(j.prompt_utts[0].abs_path, sr)
            with torch.no_grad():
                wav = synth(j.text, style(a), rng.uniform(0.1, 0.4), rng.uniform(0.5, 0.9),
                            rng.randint(5, 10), rng.uniform(1.0, 1.5))
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
