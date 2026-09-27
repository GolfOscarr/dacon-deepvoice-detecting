"""Family ``rvc`` (round 3, docs/training/15 N1-ko extension): retrieval-based voice conversion
(RVC v2 via Applio, MIT: ContentVec content features + faiss retrieval of the target's own
features, RMVPE f0, NSF-HiFi-GAN generator at 40 kHz) with per-target models trained by
ko_rvc_train.py on 16 real Korean voices (10 YODAS videos, 6 Zeroth train_val speakers).

Source = a REAL Korean utterance (Emilia/YODAS 70 %, Zeroth/FLEURS 30 %) of another speaker,
3-15 s, never from the target's training set; output = the target's voice speaking the source's
words -> FAKE. `text` = the source transcript, `prompt_speaker` = the target's pool key,
`prompt_files` = the target's training files; <family>/sources.csv adds source_speaker per file.
Targets and sources are all strategy-v4 train_val speakers (checked in folds.parquet: every
emilia-ko/fleurs-ko/zeroth-korean speaker there is train_val; Zeroth is the allow-list). Jobs: 75 % YODAS targets, 25 % Zeroth targets,
round-robin within each.
Transposition: round(12 log2(target median f0 / source median f0)), clipped to +-12, as users
set it by hand. Sources are levelled (-26 dBFS RMS, peak <= 0.5). Per-file seeded draws:
index_rate U(0.5, 0.9), protect U(0.25, 0.45).

venv: /data/project/private/dacon-venvs/synth3-rvc (python 3.12, Applio requirements).
  python scripts/synth3/ko_synth_rvc.py --n-files 6000 --shard 0/2 --max-family-hours 8
"""
from __future__ import annotations

import csv
import glob
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth2"))
import ko_common as kc  # noqa: E402

FAMILY = "rvc"
MODEL = "RVC v2 40 kHz (Applio, f0G40k/f0D40k pretrained, ContentVec, RMVPE), per-target fine-tunes"
LICENCE = "MIT (Applio code and pretrained); targets trained here"
A = Path("/data/project/private/dacon-weights/synth3/repos/Applio")
ROOT = Path("/data/project/private/dacon-weights/synth3/rvc")
# sidecar per written file (metadata.csv keeps the ko-synth2 schema): both speakers of a
# conversion, so a fold build can put the target and the source speaker in one atom
SOURCE_COLUMNS = ["file", "source_utt", "source_speaker", "source_file", "target", "target_speaker",
                  "pitch_semitones"]
APPLIO_SHA = "1fd0250006512103eeec7c6c61d2dd59824533b5"


def median_f0(x: np.ndarray, sr: int) -> float:
    import librosa
    f0 = librosa.yin(x, fmin=65, fmax=500, sr=sr, frame_length=1024)
    rms = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    f0 = f0[rms > 0.3 * np.median(rms)]   # voiced-ish frames only
    return float(np.median(f0)) if len(f0) else 150.0


def main() -> None:
    args = kc.standard_argparser(FAMILY, default_files=6000, default_seed=3808).parse_args()
    args.out_root = args.out_root.resolve()
    import soundfile as sf
    utts = kc.load_pool()
    by_rel = {u.rel_path: u for u in utts}
    targets = []
    for r in csv.DictReader((ROOT / "targets.csv").open(encoding="utf-8")):
        w = sorted(glob.glob(str(A / "logs" / r["target"] / f"{r['target']}_*e_*s.pth")))
        idx = sorted(glob.glob(str(A / "logs" / r["target"] / "*.index")))
        if w and idx:
            files = r["files"].split("|")
            targets.append({**r, "pth": w[-1], "index": idx[-1], "utts": [by_rel[f] for f in files if f in by_rel]})
    if not targets:
        raise SystemExit("no trained RVC targets")
    for t in targets:
        f0s = [median_f0(*kc.load_audio(u.abs_path, 16000)) for u in t["utts"][:8]]
        t["f0"] = float(np.median(f0s))
        t["train_ids"] = {u.utt_id for u in t["utts"]}
    yod = [t for t in targets if t["source"] != "zeroth"]
    zer = [t for t in targets if t["source"] == "zeroth"]
    # deterministic jobs over the trained targets
    rng = random.Random(f"{FAMILY}:{args.seed}:{len(targets)}")
    pool = [u for u in utts if kc.text_ok(u.text) and 3.0 <= u.duration_s <= 15.0]
    em = [u for u in pool if u.source in ("emilia", "yodas")]
    ot = [u for u in pool if u.source not in ("emilia", "yodas")]
    jobs, ci = [], {"y": 0, "z": 0}
    for i in range(args.n_files):
        k = "y" if (yod and (not zer or rng.random() < 0.75)) else "z"
        lst = yod if k == "y" else zer
        t = lst[ci[k] % len(lst)]
        ci[k] += 1
        while True:
            src = rng.choice(em if rng.random() < 0.7 else ot)
            if src.speaker != t["speaker"] and src.utt_id not in t["train_ids"]:
                break
        j = kc.Job(i, f"{FAMILY}_{i:06d}", src.text, [src.utt_id], prompt_speaker=t["speaker"],
                   prompt_utts=t["utts"], src_utts=[src], seed=args.seed * 1_000_003 + i)
        j.target = t
        jobs.append(j)
    jobs = kc.shard_jobs(jobs, args.shard)
    os.chdir(A)
    sys.path.insert(0, str(A))
    from rvc.infer.infer import VoiceConverter
    vc = VoiceConverter()
    w = kc.FamilyWriter(FAMILY, MODEL, f"Applio {APPLIO_SHA[:7]}", LICENCE, args.out_root)
    todo = [j for j in jobs if not w.is_done(j)]
    kc.log(f"{FAMILY}: {len(targets)} targets ({len(yod)} yodas, {len(zer)} zeroth); {len(jobs)} jobs in shard, {len(todo)} to do")
    tmpd = Path(tempfile.mkdtemp(prefix="rvc-"))
    for j in todo:
        if kc.should_stop(args, w):
            break
        t = j.target
        rng = random.Random(j.seed)
        a, sr = kc.load_audio(j.src_utts[0].abs_path)
        a = a * (10 ** (-26 / 20) / max(float(np.sqrt(np.mean(a ** 2))), 1e-6))
        a = a * min(1.0, 0.5 / max(float(np.abs(a).max()), 1e-6))
        src_wav, out_wav = str(tmpd / "src.wav"), str(tmpd / "out.wav")
        sf.write(src_wav, a, sr, subtype="PCM_16")
        a16, _ = kc.load_audio(j.src_utts[0].abs_path, 16000)
        semis = int(np.clip(round(12 * np.log2(t["f0"] / max(median_f0(a16, 16000), 50.0))), -12, 12))
        try:
            vc.convert_audio(audio_input_path=src_wav, audio_output_path=out_wav, model_path=t["pth"],
                             index_path=t["index"], pitch=semis, f0_method="rmvpe",
                             index_rate=rng.uniform(0.5, 0.9), protect=rng.uniform(0.25, 0.45),
                             volume_envelope=1.0, embedder_model="contentvec", export_format="WAV")
            y, ysr = sf.read(out_wav, dtype="float32")
        except Exception as e:  # noqa: BLE001
            kc.log(f"{j.name}: {type(e).__name__}: {e}")
            continue
        if w.write(j, y, ysr):
            src = j.src_utts[0]
            kc.append_row(w.dir / "sources.csv", SOURCE_COLUMNS, {
                "file": f"{j.name}.wav", "source_utt": src.utt_id, "source_speaker": src.speaker,
                "source_file": src.rel_path, "target": t["target"], "target_speaker": t["speaker"],
                "pitch_semitones": semis})
        w.progress()
    print(w.summary())


if __name__ == "__main__":
    main()
