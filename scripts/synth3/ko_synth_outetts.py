"""Family ``outetts`` (round 3, docs/training/15 N1-ko): OuteAI/Llama-OuteTTS-1.0-1B zero-shot
cloning (a Llama-3.2-1B LM emitting the two codebooks of IBM's DAC.speech.v1.0 24 kHz 1.5 kbps codec
word by word, decoded by DAC; the package resamples to 44.1 kHz, which is what users get).
Korean is one of its 12 "high training data" languages. Licence: CC-BY-NC-SA-4.0 (OuteAI's
parts) + Llama 3.2 Community License (base).

venv: /data/project/private/dacon-venvs/synth3-outetts (outetts 0.4.4, HF backend, torch 2.14,
transformers 4.52.3). Weights: HF OuteAI/Llama-OuteTTS-1.0-1B under
HF_HOME=/data/project/private/dacon-weights/synth3/hf; DAC in ~/.cache/outeai/dac.
A speaker profile needs word timestamps of the prompt: outetts runs openai-whisper "turbo" (cached
once here, in /data/project/private/dacon-weights/synth3/whisper) with language="ko".
Prompts are levelled (-26 dBFS RMS, peak <= 0.5). Per-batch seeded temperature U(0.35, 0.55), the
package's other sampler defaults (top_k 40, top_p 0.9, min_p 0.05, repetition penalty 1.1 over the
last 64 tokens).
Speed: the package's HF path generates one file at a time and applies its windowed repetition
penalty in a Python loop over tokens (0.08x real time). Here the same penalty is vectorised and
--batch prompts are generated together (left-padded) with the package's own prompt builder and
code extraction.

  python scripts/synth3/ko_synth_outetts.py --n-files 6000 --shard 0/2 --max-family-hours 9
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth2"))
import ko_common as kc  # noqa: E402

FAMILY = "outetts"
MODEL = "OuteAI/Llama-OuteTTS-1.0-1B (+ ibm-research/DAC.speech.v1.0)"
LICENCE = "CC-BY-NC-SA-4.0 + Llama 3.2 Community License"
WHISPER_ROOT = "/data/project/private/dacon-weights/synth3/whisper"
# ~75 frames/s x 2 codebooks + word/feature tokens: ~24 s of audio; a row that never ends must not
# hold its batch until the 8192-token context is full (QC drops > 20 s anyway)
MAX_NEW = 4000
TOK_PER_S = 200   # per batch: 1.5x the longest source speech + 2 s


def main() -> None:
    ap = kc.standard_argparser(FAMILY, default_files=6000, default_seed=3202)
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()
    import soundfile as sf
    import torch
    import torchaudio
    import whisper

    def _sf_load(src, *a, **k):  # torchaudio >= 2.9 needs torchcodec for load; soundfile instead
        x, sr = sf.read(src, dtype="float32", always_2d=True)
        return torch.from_numpy(x.T.copy()), sr
    torchaudio.load = _sf_load
    import outetts
    from huggingface_hub import snapshot_download
    from outetts.version.v3 import audio_processor as ap

    asr = whisper.load_model("turbo", device="cuda", download_root=WHISPER_ROOT)

    def _words(audio_path, model="turbo", device=None, language=None):  # cached model, Korean
        return asr.transcribe(audio_path, word_timestamps=True, language="ko")
    ap.transcribe_once_word_level = _words

    from outetts.version.playback import ModelOutput
    from transformers import LogitsProcessor
    import transformers.generation.utils as gu

    class _WindowPenalty(LogitsProcessor):  # outetts' windowed penalty (last 64 tokens), vectorised
        def __init__(self, penalty: float):
            self.penalty = penalty

        @torch.no_grad()
        def __call__(self, input_ids, scores):
            hit = torch.zeros_like(scores, dtype=torch.bool)
            hit.scatter_(1, input_ids[:, -64:].clamp(max=scores.shape[-1] - 1), True)
            pen = torch.where(scores <= 0, scores * self.penalty, scores / self.penalty)
            return torch.where(hit, pen, scores)
    gu.RepetitionPenaltyLogitsProcessor = _WindowPenalty

    snap = Path(snapshot_download("OuteAI/Llama-OuteTTS-1.0-1B"))
    iface = outetts.Interface(config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B, backend=outetts.Backend.HF))
    lm, tok, pp = iface.model.model, iface.model.tokenizer, iface.prompt_processor
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    utts = kc.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, snap.name, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    dur = {u.utt_id: u.duration_s for u in utts}
    src_s = {j.name: sum(dur.get(t, 13.0) for t in j.text_utts) for j in todo}
    # similar lengths share a batch (a batch runs as long as its longest row): sort each window
    # of 8 batches by source-speech length
    win = 8 * args.batch
    todo = [j for k in range(0, len(todo), win) for j in sorted(todo[k:k + win], key=lambda j: src_s[j.name])]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do, dtype={lm.dtype}")
    tmpd = Path(tempfile.mkdtemp(prefix="oute-prompt-"))
    for b in range(0, len(todo), args.batch):
        if kc.should_stop(args, w):
            break
        batch, enc = [], []
        for j in todo[b:b + args.batch]:
            a, psr = kc.load_audio(j.prompt_utts[0].abs_path)
            a = a * (10 ** (-26 / 20) / max(float(np.sqrt(np.mean(a ** 2))), 1e-6))
            a = a * min(1.0, 0.5 / max(float(np.abs(a).max()), 1e-6))
            prompt_wav = str(tmpd / "prompt.wav")
            sf.write(prompt_wav, a, psr, subtype="PCM_16")
            try:
                spk = iface.create_speaker(prompt_wav)
                enc.append(tok.encode(pp.get_completion_prompt(j.text, spk), add_special_tokens=False))
                batch.append(j)
            except Exception as e:  # noqa: BLE001
                kc.log(f"{j.name}: speaker: {type(e).__name__}: {e}")
        if not batch:
            continue
        n = max(len(x) for x in enc)
        ids = torch.tensor([[pad] * (n - len(x)) + x for x in enc], device=lm.device)
        mask = torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in enc], device=lm.device)
        rng = random.Random(batch[0].seed)
        torch.manual_seed(batch[0].seed)
        max_new = min(MAX_NEW, int(TOK_PER_S * (1.5 * max(src_s[j.name] for j in batch) + 2)))
        try:
            out = lm.generate(ids, attention_mask=mask, max_new_tokens=max_new, do_sample=True,
                              temperature=rng.uniform(0.35, 0.55), repetition_penalty=1.1,
                              top_k=40, top_p=0.9, min_p=0.05, pad_token_id=pad)
        except Exception as e:  # noqa: BLE001
            kc.log(f"{batch[0].name}+{len(batch)}: {type(e).__name__}: {e}")
            continue
        for j, row in zip(batch, out[:, n:].tolist()):
            codes = pp.extract_audio_from_tokens(row)
            if not codes[0]:
                kc.log(f"{j.name}: no audio codes")
                continue
            audio = iface.audio_codec.decode(torch.tensor([codes], dtype=torch.int64, device=iface.audio_codec.device))
            o = ModelOutput(audio, iface.audio_codec.sr)
            w.write(j, o.audio.detach().float().cpu().numpy().squeeze(), o.sr)
            w.progress(every=10)
    print(w.summary())


if __name__ == "__main__":
    main()
