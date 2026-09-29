"""Family ``fishspeech``: fishaudio/fish-speech-1.5 (dual-AR LLaMA over FireflyGAN FSQ tokens +
Firefly-GAN VQ decoder, 44.1 kHz, CC-BY-NC-SA-4.0), zero-shot cloning in-process (models loaded once).

Code: fish-speech repo @ tag v1.5.1 (58046ea), /data/project/private/dacon-weights/synth2/repos/fish-speech,
installed --no-deps into /data/project/private/dacon-venvs/synth2-fishspeech (pyaudio / gradio /
funasr / modelscope / silero-vad / faster-whisper left out: server/webui only).
Prompt: one real utterance (3-10 s) + its transcript; text: 1-2 other transcripts.
Per-file seeded draws: temperature U(0.6, 0.8), top_p U(0.6, 0.8), repetition_penalty U(1.1, 1.3).

  python scripts/synth2/ko_synth_fishspeech.py --n-files 6000 --shard 0/2 --max-hours 4.5
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ko_common as kc  # noqa: E402

FAMILY = "fishspeech"
MODEL = "fishaudio/fish-speech-1.5 (code v1.5.1 58046ea)"
LICENCE = "CC-BY-NC-SA-4.0"
REPO = Path(os.environ.get("FISH_REPO", "/data/project/private/dacon-weights/synth2/repos/fish-speech"))


def main() -> None:
    ap = kc.standard_argparser(FAMILY, default_files=6000, default_seed=1505)
    ap.add_argument("--compile", action="store_true", help="torch.compile the decode step (repo option; eager is ~15 tok/s = 0.7x RT)")
    args = ap.parse_args()
    os.chdir(REPO)  # hydra config_path is relative to the package; references/ dir is cwd-relative
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.models.vqgan.inference import load_model as load_decoder_model
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    snap = Path(snapshot_download("fishaudio/fish-speech-1.5"))
    precision = torch.bfloat16
    llama_queue = launch_thread_safe_queue(checkpoint_path=str(snap), device=dev, precision=precision, compile=args.compile)
    decoder = load_decoder_model(config_name="firefly_gan_vq",
                                 checkpoint_path=str(snap / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"), device=dev)
    engine = TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder, precision=precision, compile=args.compile)
    sr = decoder.spec_transform.sample_rate

    utts = kc.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, snap.name, LICENCE, args.out_root)
    # a CUDA device-side assert (seen in compiled decode: `index out of bounds: 0 <= tmp5 < 1024`,
    # ~1 per 400 files) poisons the context; the job is recorded in _skip.txt and the process
    # exits 3 so ko_run.sbatch relaunches it (resume skips done + skipped jobs).
    skip_f = w.dir / "_skip.txt"
    skip = set(skip_f.read_text().split()) if skip_f.exists() else set()
    todo = [j for j in jobs if not w.is_done(j) and j.name not in skip]
    kc.log(f"{FAMILY}: {kc.pool_summary(utts)}; {len(jobs)} jobs in shard, {len(todo)} to do, sr={sr}, dev={dev}")
    for j in todo:
        if kc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        p = j.prompt_utts[0]
        try:
            req = ServeTTSRequest(
                text=j.text, references=[ServeReferenceAudio(audio=p.abs_path.read_bytes(), text=p.text)],
                seed=j.seed % 2**31, max_new_tokens=512,  # ~24 s at 21.5 tok/s; bounds run-ons (smoke: one 30.6 s output)
                chunk_length=200, normalize=False,
                temperature=round(rng.uniform(0.6, 0.8), 3), top_p=round(rng.uniform(0.6, 0.8), 3),
                repetition_penalty=round(rng.uniform(1.1, 1.3), 3), format="wav")
            wav = None
            for res in engine.inference(req):
                if res.code == "error":
                    raise res.error
                if res.code == "final":
                    wav = res.audio[1]
        except Exception as e:  # noqa: BLE001
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            if "CUDA error" in str(e) or "device-side assert" in str(e):
                with skip_f.open("a") as fh:
                    fh.write(j.name + "\n")
                print(w.summary(), flush=True)
                os._exit(3)
            continue
        if wav is None:
            continue
        w.write(j, np.asarray(wav, dtype=np.float32), sr)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
