"""C tier: what is actually *in* the audio, as opposed to what it measures like.

docs/EDA/09 step 7, and the only thing **`G-EDA6`** waits on: *"every row's
asserted components are evidenced (VAD/PANNs/energy), or carry a validity mask,
or are dropped with a recorded reason"*. A pool-C file asserts
`voice_present = 0`; this is what can contradict it.

🔴 **Chain plane only** (`eda.extract.run_content`). Every other tier runs twice
and suffixes its columns, because R1's question is a subtraction. This tier's
question -- *is there a voice here?* -- is a property of the recording, and its
model is trained at one rate. So the columns here have no `_native` twin, and
that asymmetry is deliberate rather than an omission.

⚠️ **Two thresholds, not one.** docs/EDA/03 C2 asks for 0.5 and 0.4 together,
because the number that matters is not either ratio but how much the answer
moves between them: a file whose speech ratio is 0.02 at 0.5 and 0.60 at 0.4 is
a file the VAD is unsure about, and treating it as confidently instrumental is
how a mislabelled pool gets built.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eda.extract import register_content
from models.vendor.silero_vad import CHUNK, SAMPLE_RATE, speech_probabilities

__all__ = ["THRESHOLDS", "vad_row"]

#: docs/EDA/03 C2's pair. The first is Silero's own default.
THRESHOLDS = (0.5, 0.4)
#: A span shorter than this is not treated as speech. One 512-sample chunk is
#: 32 ms; a lone chunk over threshold inside music is far more often a drum hit
#: than a word, and `vad_spans` is meant to count utterances.
MIN_SPAN_CHUNKS = 4

COLUMNS = (
    *(f"vad_speech_ratio_{int(t * 100)}" for t in THRESHOLDS),
    *(f"vad_spans_{int(t * 100)}" for t in THRESHOLDS),
    "vad_speech_prob_mean", "vad_speech_prob_max", "vad_chunks",
    "vad_ok", "vad_error",
)


def _spans(mask: np.ndarray, min_len: int) -> int:
    """Runs of `True` at least `min_len` long."""
    if not mask.any():
        return 0
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    lengths = edges[1::2] - edges[::2]
    return int((lengths >= min_len).sum())


@register_content("vad", COLUMNS, "vad_ok")
def vad_row(wav: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Silero VAD over one chain-plane waveform.

    Channels are downmixed first, as every other extractor does: the chain plane
    is already mono by policy, and doing it here too means this cannot silently
    measure channel 0 of something that was not.

    ⚠️ A file shorter than one 512-sample chunk (32 ms) produces no probability
    at all. That is reported as a failure with its reason rather than as a
    speech ratio of 0.0, which would read as *"measured, and there is no
    voice"*.
    """
    failed = {c: None for c in COLUMNS}
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        return {**failed, "vad_ok": False, "vad_error": "empty waveform"}
    mono = wav.mean(axis=0) if wav.ndim > 1 else wav
    if mono.size < CHUNK:
        return {**failed, "vad_ok": False,
                "vad_error": f"{mono.size} sample(s); one VAD chunk is {CHUNK}"}
    if sample_rate != SAMPLE_RATE:
        return {**failed, "vad_ok": False,
                "vad_error": f"content tier expects {SAMPLE_RATE} Hz, got {sample_rate}"}

    probs = speech_probabilities(mono, sample_rate)
    if probs.size == 0:
        return {**failed, "vad_ok": False, "vad_error": "no complete VAD chunk"}

    row: dict[str, Any] = {
        "vad_speech_prob_mean": float(probs.mean()),
        "vad_speech_prob_max": float(probs.max()),
        "vad_chunks": int(probs.size),
        "vad_ok": True, "vad_error": None,
    }
    for threshold in THRESHOLDS:
        mask = probs >= threshold
        tag = int(threshold * 100)
        row[f"vad_speech_ratio_{tag}"] = float(mask.mean())
        row[f"vad_spans_{tag}"] = _spans(mask, MIN_SPAN_CHUNKS)
    return row
