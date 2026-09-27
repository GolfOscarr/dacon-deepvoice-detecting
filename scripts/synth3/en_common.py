"""Shared helper for the English round-3 families (en-synth3, docs/training/15 §2 N1-en; N5 extra).

A thin layer over round 2's ``scripts/synth2/ko_common.py`` in its English mode: the same job list,
shards, resumable writer and metadata schema (so ``processing/extend.py`` ingests en-synth3 exactly
as it ingests en-synth2). Import this module BEFORE anything reads ko_common's globals: it sets
SYNTH2_LANG=en and KO_SYNTH2_ROOT (default interim/en-synth3; N5 uses interim/en-synth2-extra).

Differences from round 2:
* the prompt/text pool is restricted to speakers whose strategy-v4 folds.parquet rows are all
  slice train_val (never probe): _index/en_train_val_speakers.txt;
* ``EN3_EXCLUDE_SPEAKERS=<file>``: the speakers listed there are never PROMPT speakers (N5: the
  en-synth2 prompt speakers, so the extra hours come from new voices); their utterances can still be
  texts / VC sources. Every LibriTTS-R speaker was an en-synth2 prompt, so N5 prompts are 100 % Emilia.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["SYNTH2_LANG"] = "en"
os.environ.setdefault("KO_SYNTH2_ROOT", "/data/project/private/dacon-corpus/interim/en-synth3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth2"))
import ko_common as kc  # noqa: E402

IDX3 = kc.INTERIM / "en-synth3" / "_index"
ALLOW = IDX3 / "en_train_val_speakers.txt"


def _lines(p: Path) -> set[str]:
    return {s.strip() for s in p.read_text().splitlines() if s.strip()}


def allowed_speakers() -> set[str]:
    return _lines(ALLOW)


_kc_load_pool = kc.load_pool   # the N5 wrappers replace kc.load_pool with load_pool below


def load_pool() -> list:
    allow = allowed_speakers()
    return [u for u in _kc_load_pool() if u.speaker in allow]


def _exclude_prompt_speakers() -> None:
    """kc.make_jobs takes prompt speakers from kc.by_speaker(utts) and texts from utts itself, so
    filtering by_speaker removes the excluded speakers from the prompt side only."""
    ex = os.environ.get("EN3_EXCLUDE_SPEAKERS")
    if not ex:
        return
    bad = _lines(Path(ex))
    orig = kc.by_speaker
    kc.by_speaker = lambda utts: {k: v for k, v in orig(utts).items() if k not in bad}


_exclude_prompt_speakers()


def level(path: Path, target_sr: int | None = None, rms_db: float = -26.0, max_peak: float = 0.5):
    """(audio, sr) at a fixed level (round 2's CosyVoice/Seed-VC clipping fix): RMS rms_db, peak <= max_peak."""
    import numpy as np
    a, sr = kc.load_audio(Path(path), target_sr)
    rms = float(np.sqrt(np.mean(a ** 2))) or 1e-6
    g = min(10 ** (rms_db / 20) / rms, max_peak / max(float(np.abs(a).max()), 1e-6))
    return (a * g).astype(np.float32), sr


def level_to_wav(path: Path, out: Path, target_sr: int | None = None) -> Path:
    import soundfile as sf
    a, sr = level(path, target_sr)
    sf.write(str(out), a, sr, subtype="FLOAT")
    return out
