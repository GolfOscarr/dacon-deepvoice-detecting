"""Family ``supertonic`` (round 3, docs/training/15 N1-ko extension): Supertone/supertonic-2, a
lightweight ONNX TTS (character-level text encoder + duration predictor + latent flow-matching
vector estimator with few denoising steps + its own vocoder, 44.1 kHz). Korean is one of its
languages (lang="ko"). NOT a cloner: it has 10 preset voice styles (F1-F5, M1-M5), so
`prompt_speaker` = "supertonic-2/<style>" and `prompt_files` = the style json.
Licence: weights BigScience OpenRAIL-M (use-based restrictions, no output restriction); sample code MIT.

venv: /data/project/private/dacon-venvs/synth3-supertonic (onnxruntime, CPU provider: the
package marks GPU as untested). Weights: round 2's HF snapshot Supertone/supertonic-2 @75e6727
(dacon-weights/synth2/hf); code github supertone-inc/supertonic @1e9799e (py/helper.py).
Long texts: the helper splits Korean at 120 characters and joins chunks with EXACT-ZERO silence,
which the zero-run screen would drop and which is a shortcut, so chunks are joined directly here.
Texts: the same draw as every family (kc.make_jobs; the drawn prompt is ignored).
Style = round-robin over the 10 by job index (the smoke's first 12 files drew it at random);
per-file seeded draws: total_step {5..10}, speed U(0.9, 1.15). Capped at ~5 h kept (lead: with
10 fixed voices, more hours mostly teach the voices, not the artifact).

  python scripts/synth3/ko_synth_supertonic.py --n-files 6000 --shard 0/1 --max-family-hours 5.4
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth2"))
import ko_common as kc  # noqa: E402

FAMILY = "supertonic"
MODEL = "Supertone/supertonic-2 (ONNX, preset voice styles)"
LICENCE = "BigScience OpenRAIL-M (weights); MIT (code)"
SNAP = Path("/data/project/private/dacon-weights/synth2/hf/hub/models--Supertone--supertonic-2/snapshots/75e6727618a02f323c720cba9478152d4bc16ca4")
CODE = Path("/data/project/private/dacon-weights/synth3/repos/supertonic/py")
STYLES = [f"{g}{i}" for g in "FM" for i in range(1, 6)]


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=6000, default_seed=3909).parse_args()
    sys.path.insert(0, str(CODE))
    import onnxruntime as ort
    from helper import chunk_text, load_text_to_speech, load_voice_style
    _so = ort.SessionOptions

    def _threads():  # ORT's default spawns one pinned thread per host core (144) inside an
        o = _so()    # 8-CPU job: affinity errors and a 4x slowdown; use the job's CPUs
        o.intra_op_num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
        o.inter_op_num_threads = 1
        return o
    ort.SessionOptions = _threads
    tts = load_text_to_speech(str(SNAP / "onnx"), use_gpu=False)
    sr = tts.sample_rate
    styles = {s: load_voice_style([str(SNAP / "voice_styles" / f"{s}.json")]) for s in STYLES}
    utts = kc.load_pool()
    jobs = kc.shard_jobs(kc.make_jobs(FAMILY, args.n_files, args.seed, utts), args.shard)
    w = kc.FamilyWriter(FAMILY, MODEL, SNAP.name, LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {len(jobs)} jobs in shard, {len(todo)} to do, sr={sr}")
    for j in todo:
        if kc.should_stop(args, w):
            break
        rng = random.Random(j.seed)
        s = STYLES[j.idx % len(STYLES)]   # even spread over the 10 styles (lead, 06:20)
        j.prompt_speaker = f"supertonic-2/{s}"
        j.prompt_utts = [kc.Utt(s, j.prompt_speaker, f"voice_styles/{s}.json", "", 0.0, "preset")]
        steps, speed = rng.randint(5, 10), rng.uniform(0.9, 1.15)
        try:
            parts = []
            for chunk in chunk_text(j.text, max_len=120):
                wav, dur = tts.batch([chunk], ["ko"], styles[s], steps, speed)
                parts.append(wav[0, : int(sr * float(np.asarray(dur).reshape(-1)[0]))])
            audio = np.concatenate(parts)
        except Exception as e:  # noqa: BLE001
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        w.write(j, audio, sr)
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
