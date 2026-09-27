"""N5: more English seedvc hours (docs/training/15 §4 N5) into interim/en-synth2-extra/seedvc/ with NEW prompt
speakers. Round 2's scripts/synth2/ko_synth_seedvc.py unchanged (same model, revision, sampling), with the
pool swapped for en_common.load_pool (v4 train_val speakers only) and EN3_EXCLUDE_SPEAKERS (the
en-synth2 prompt speakers, _index/en_synth2_prompt_speakers.txt) removed from the prompt side. Use a --seed other than round 2's so
the texts differ too. venv: /data/project/private/dacon-venvs/synth2-seedvc (read-only reuse).

  EN3_ROOT=/data/project/private/dacon-corpus/interim/en-synth2-extra \
  EN3_EXCLUDE_SPEAKERS=/data/project/private/dacon-corpus/interim/en-synth3/_index/en_synth2_prompt_speakers.txt \
  sbatch scripts/synth3/en_run.sbatch seedvc --seed 3606 --max-family-hours 4.4
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import en_common as ec  # noqa: E402  (sets SYNTH2_LANG=en before ko_common loads)

ec.kc.load_pool = ec.load_pool
import ko_synth_seedvc as m  # noqa: E402

if __name__ == "__main__":
    assert "extra" in str(ec.kc.OUT_ROOT), f"N5 writes to en-synth2-extra, not {ec.kc.OUT_ROOT}"
    m.main()
