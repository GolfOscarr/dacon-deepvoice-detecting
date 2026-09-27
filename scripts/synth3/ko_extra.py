"""N5 (docs/training/15 §4): more hours of round 2's ko maskgct / seedvc with NEW prompt speakers,
into interim/ko-synth2-extra/<family>/ (KO_SYNTH2_ROOT, set by scripts/synth3/ko_run.sbatch).

Runs scripts/synth2/ko_synth_<family>.py unchanged, except that the utterance pool loses every
Emilia/YODAS speaker that already prompts interim/ko-synth2/<family>/ (so those speakers are
neither prompts nor sources here). Zeroth keeps all 115 train_val speakers: round 2 used them all,
so there is no new Zeroth speaker to take. Pass a --seed other than round 2's.

  python scripts/synth3/ko_extra.py maskgct --seed 5202 --shard 0/1 --max-family-hours 4.4
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

S2 = Path(__file__).resolve().parents[1] / "synth2"
sys.path.insert(0, str(S2))
import ko_common as kc  # noqa: E402


def main() -> None:
    family = sys.argv.pop(1)
    assert family in ("maskgct", "seedvc"), family
    used = {r["prompt_speaker"] for r in kc.read_csv(kc.INTERIM / "ko-synth2" / family / "synth.csv")}
    used = {s for s in used if s.startswith("emilia-ko/")}
    load = kc.load_pool

    def load_pool():
        u = load()
        kept = [x for x in u if x.speaker not in used]
        kc.log(f"{family} extra: {len(used)} round-2 Emilia prompt speakers excluded, "
               f"{len(u) - len(kept)} utterances dropped from the pool")
        return kept
    kc.load_pool = load_pool
    importlib.import_module(f"ko_synth_{family}").main()


if __name__ == "__main__":
    main()
