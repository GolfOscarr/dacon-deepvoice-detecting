"""Silero VAD, vendored as a TorchScript archive.

Vendored rather than `torch.hub.load`ed for the reason the BEATs encoder beside
it is: **the submission installs and runs offline**
(docs/competition/02-submission.md), and `torch.hub` resolves a GitHub ref at
call time. A content measurement the EDA can make and the submission cannot
reproduce is a measurement that decides nothing.

The upstream distribution *is* this file -- a `torch.jit` archive, not source --
so there is nothing to keep byte-for-byte except the archive itself. Provenance
and the licence are in `PROVENANCE.md` and `LICENSE` (MIT, read at origin).

🔴 **It is 16 kHz-only in practice, and that is why the content tier runs on the
chain plane alone.** The model accepts 8 kHz and 16 kHz and wants exactly 512
samples per call at 16 kHz; the corpus's native plane is anything from 16 kHz to
44.1 kHz. Running it there would mean either resampling inside an extractor --
which is the render chain's job, not an extractor's -- or measuring a different
thing per source. `eda.extract.run_content` runs the chain plane and nothing
else.

⚠️ `reset_states()` between files is mandatory. The model is recurrent: its
hidden state carries across calls, so a file measured after a loud one scores
differently than the same file measured first. That is a reproducibility bug
that no assertion about the output would catch -- the numbers are all plausible.
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch

__all__ = ["CHUNK", "SAMPLE_RATE", "MODEL_SHA256", "load_vad", "speech_probabilities"]

#: The archive, beside this file.
MODEL_PATH = Path(__file__).resolve().parent / "silero_vad.jit"
#: Recorded so a swapped or truncated model is a loud failure rather than a
#: quietly different set of numbers. Verified at load.
MODEL_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"

#: Samples per call at 16 kHz. Upstream v5 requires exactly this.
CHUNK = 512
SAMPLE_RATE = 16000

_LOCK = threading.Lock()
_MODEL: torch.jit.ScriptModule | None = None


def load_vad() -> torch.jit.ScriptModule:
    """The model, loaded once per process and verified against its digest.

    ⚠️ Guarded by a lock because `eda.driver` runs its extractors in a thread
    pool. Two threads racing `torch.jit.load` on the first file is a rare,
    machine-dependent crash, and the fix is cheaper than the diagnosis.
    """
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            import hashlib

            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"{MODEL_PATH} is not on disk. It is vendored into the "
                    f"repository on purpose; a checkout that lacks it cannot "
                    f"reproduce the content tier")
            digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
            if digest != MODEL_SHA256:
                raise RuntimeError(
                    f"{MODEL_PATH} hashes {digest[:12]}, expected "
                    f"{MODEL_SHA256[:12]}. A different model produces different "
                    f"numbers under the same column names")
            model = torch.jit.load(str(MODEL_PATH))
            model.eval()
            _MODEL = model
    return _MODEL


def speech_probabilities(wav, sample_rate: int = SAMPLE_RATE):
    """Per-chunk speech probability for one mono 16 kHz waveform.

    Returns one probability per `CHUNK` samples; a trailing partial chunk is
    dropped rather than zero-padded, because padding invents silence at the end
    of every file and biases `speech_ratio` downward by up to one chunk.
    """
    import numpy as np

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"silero_vad is used at {SAMPLE_RATE} Hz here, got {sample_rate}. "
            f"The content tier runs on the chain plane for exactly this reason")
    model = load_vad()
    x = torch.as_tensor(np.ascontiguousarray(wav), dtype=torch.float32)
    out = []
    with _LOCK:
        # 🔴 State reset *and* inference under one lock. The model is recurrent
        # and shared across the driver's threads, so a reset by one thread
        # between another's chunks would silently corrupt both files.
        model.reset_states()
        with torch.no_grad():
            for start in range(0, len(x) - CHUNK + 1, CHUNK):
                out.append(float(model(x[start:start + CHUNK], SAMPLE_RATE)))
    return np.asarray(out, dtype=np.float32)
