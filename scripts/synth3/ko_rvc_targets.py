"""RVC targets for the ``rvc`` family (round 3): pick the target voices and write their training
sets as 16-bit wavs (CPU only).

RVC needs minutes of one voice per target, which Emilia (one segment per speaker) cannot give.
Targets are therefore YODAS videos (the Emilia-YODAS part of the pool, keyed by video like every
other clone's prompt_speaker) with >= 60 s of kept segments, and Zeroth train_val speakers
(3-5 min each). Seed-shuffled; --n-yodas / --n-zeroth of each.
Output: /data/project/private/dacon-weights/synth3/rvc/datasets/<target>/*.wav and targets.csv
(target, speaker, source, n_files, seconds, files).

  python scripts/synth3/ko_rvc_targets.py --n-yodas 10 --n-zeroth 6
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth2"))
import ko_common as kc  # noqa: E402

ROOT = Path("/data/project/private/dacon-weights/synth3/rvc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-yodas", type=int, default=10)
    ap.add_argument("--n-zeroth", type=int, default=6)
    ap.add_argument("--zeroth-seconds", type=float, default=270.0)
    ap.add_argument("--seed", type=int, default=3707)
    a = ap.parse_args()
    import soundfile as sf
    rng = random.Random(a.seed)
    spk = kc.by_speaker(kc.load_pool())
    yod = sorted(s for s, v in spk.items() if v[0].source == "yodas" and sum(u.duration_s for u in v) >= 60)
    zer = sorted(s for s, v in spk.items() if v[0].source == "zeroth")
    rng.shuffle(yod)
    rng.shuffle(zer)
    rows = []
    for k, s in enumerate(yod[: a.n_yodas] + zer[: a.n_zeroth]):
        us = sorted(spk[s], key=lambda u: u.utt_id)
        if spk[s][0].source == "zeroth":
            rng.shuffle(us)
            pick, tot = [], 0.0
            for u in us:
                if tot >= a.zeroth_seconds:
                    break
                pick.append(u)
                tot += u.duration_s
            us = pick
        name = f"t{k:02d}"
        d = ROOT / "datasets" / name
        d.mkdir(parents=True, exist_ok=True)
        for u in us:
            x, sr = kc.load_audio(u.abs_path)
            sf.write(str(d / f"{u.utt_id}.wav"), x, sr, subtype="PCM_16")
        rows.append({"target": name, "speaker": s, "source": spk[s][0].source, "n_files": len(us),
                     "seconds": f"{sum(u.duration_s for u in us):.1f}",
                     "files": "|".join(u.rel_path for u in us)})
        print(name, s, len(us), rows[-1]["seconds"])
    with (ROOT / "targets.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
