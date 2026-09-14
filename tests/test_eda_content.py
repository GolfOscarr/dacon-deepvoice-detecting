"""The content tier: `eda.extract.content`, docs/EDA/09 step 7.

Every invariant here is paired with the mutation that breaks it, in the house
style -- a green suite is not evidence that a check can fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from eda.extract import CONTENT, ExtractorError, run_content
from eda.extract.content import COLUMNS, MIN_SPAN_CHUNKS, THRESHOLDS, _spans, vad_row
from eda.planes import CHAIN, NATIVE, Plane
from models.vendor.silero_vad import CHUNK, MODEL_SHA256, SAMPLE_RATE

SR = SAMPLE_RATE


def _noise(seconds=2.0, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((1, int(seconds * SR))) * scale).astype(np.float32)


# --------------------------------------------------------------------------- #
# one plane, and it is the chain
# --------------------------------------------------------------------------- #

def test_the_content_tier_runs_on_the_chain_plane_and_says_so_when_it_cannot():
    """🔴 Every other tier runs twice and suffixes; this one runs once and does
    not. Silero wants exactly 16 kHz, and the native plane is anything up to
    44.1 kHz -- running it there would mean resampling inside an extractor,
    which is the render chain's job.

    Mutation: `run_content` made to iterate `planes` like `run_signal`. The
    native call then fails per file and every `vad_*` column doubles.
    """
    wav = _noise()
    planes = {NATIVE: Plane(NATIVE, wav, 44100), CHAIN: Plane(CHAIN, wav, SR)}
    out = run_content(planes)
    assert set(out) == set(COLUMNS), "unsuffixed, one plane"
    assert not any(k.endswith(("_native", "_chain")) for k in out)

    with pytest.raises(KeyError, match="runs on the 'chain' plane"):
        run_content({NATIVE: Plane(NATIVE, wav, 44100)})


def test_two_content_extractors_cannot_claim_one_column():
    """The same guard the other tiers have. Mutation: the `clash` check removed
    -- the second extractor silently overwrites the first."""
    from eda.extract import Registry

    reg = Registry("C", 2, ("wav", "sample_rate"))
    reg.add("a", ("x", "ok"), "ok", lambda wav, sample_rate: {"x": 1, "ok": True})
    reg.add("b", ("x", "ok2"), "ok2", lambda wav, sample_rate: {"x": 2, "ok2": True})
    import eda.extract as ex

    original, ex.CONTENT = ex.CONTENT, reg
    try:
        with pytest.raises(ExtractorError, match="already written"):
            ex.run_content({CHAIN: Plane(CHAIN, _noise(), SR)})
    finally:
        ex.CONTENT = original


# --------------------------------------------------------------------------- #
# the model is the one we vendored
# --------------------------------------------------------------------------- #

def test_the_vendored_model_is_verified_against_its_digest():
    """🔴 A swapped or truncated model produces different numbers under the same
    column names, which is the failure nothing downstream can see.

    Mutation: the digest comparison in `load_vad` removed.
    """
    import models.vendor.silero_vad as vendor

    assert vendor.MODEL_PATH.exists()
    real = vendor.MODEL_SHA256
    # ⚠️ Both the per-thread cache and the once-only verify flag have to be
    # cleared, or the check is skipped and this passes for the wrong reason.
    vendor._LOCAL = vendor.threading.local()
    vendor._VERIFIED = False
    vendor.MODEL_SHA256 = "0" * 64
    try:
        with pytest.raises(RuntimeError, match="expected 000000000000"):
            vendor.load_vad()
    finally:
        vendor.MODEL_SHA256 = real
        vendor._LOCAL = vendor.threading.local()
        vendor._VERIFIED = False


def test_the_vad_refuses_a_rate_it_was_not_vendored_for():
    """Mutation: the `sample_rate != SAMPLE_RATE` guard removed from
    `speech_probabilities` -- Silero accepts 8 kHz too, so a mis-planed call
    returns plausible numbers for the wrong audio."""
    from models.vendor.silero_vad import speech_probabilities

    with pytest.raises(ValueError, match="chain plane"):
        speech_probabilities(_noise()[0], 44100)


# --------------------------------------------------------------------------- #
# what it emits
# --------------------------------------------------------------------------- #

def test_both_thresholds_are_emitted():
    """docs/EDA/03 C2 asks for 0.5 and 0.4 together: the number that matters is
    how much the answer moves between them.

    Mutation: `THRESHOLDS` narrowed to one value.
    """
    assert THRESHOLDS == (0.5, 0.4)
    out = vad_row(_noise(), SR)
    for t in THRESHOLDS:
        assert f"vad_speech_ratio_{int(t * 100)}" in out
        assert f"vad_spans_{int(t * 100)}" in out


def test_a_file_too_short_to_measure_is_a_failure_not_a_zero_ratio():
    """🔴 A speech ratio of 0.0 reads as *"measured, and there is no voice"*.
    A file of 100 samples was never measured at all, and G-EDA6 turns on
    exactly that difference.

    Mutation: the `mono.size < CHUNK` guard removed -- `probs` is then empty and
    the row reports 0.0 with `vad_ok` true.
    """
    out = vad_row(np.zeros((1, CHUNK - 1), dtype=np.float32), SR)
    assert out["vad_ok"] is False
    assert "one VAD chunk is" in out["vad_error"]
    assert out["vad_speech_ratio_50"] is None


def test_a_wrong_rate_is_a_row_not_an_exception():
    out = vad_row(_noise(), 44100)
    assert out["vad_ok"] is False and "expects 16000" in out["vad_error"]


def test_a_lone_chunk_over_threshold_is_not_a_span():
    """⚠️ 512 samples is 32 ms. A single chunk over threshold inside music is
    far more often a drum hit than a word, and `vad_spans` counts utterances.

    Mutation: `MIN_SPAN_CHUNKS` set to 1 -- every transient becomes a span.
    """
    mask = np.zeros(20, dtype=bool)
    mask[3] = True                        # one chunk
    mask[10:10 + MIN_SPAN_CHUNKS] = True  # a real span
    assert _spans(mask, MIN_SPAN_CHUNKS) == 1
    assert _spans(mask, 1) == 2


def test_spans_counts_runs_not_chunks():
    mask = np.array([True] * 8 + [False] * 3 + [True] * 8)
    assert _spans(mask, MIN_SPAN_CHUNKS) == 2
    assert _spans(np.zeros(10, dtype=bool), MIN_SPAN_CHUNKS) == 0


# --------------------------------------------------------------------------- #
# it rides the same decode
# --------------------------------------------------------------------------- #

def test_importing_the_driver_populates_the_content_registry():
    """🔴 The defect that already happened once, with the vector tier: a module
    written, tested and wired in whose registration import was never added, so
    the pass reported success over an empty registry.

    Run in a clean interpreter, and deliberately not importing
    `eda.extract.content` here.
    """
    import subprocess
    import sys
    from pathlib import Path

    probe = ("import eda.driver; from eda.extract import CONTENT;"
             "assert sorted(CONTENT) == ['vad'], sorted(CONTENT); print('ok')")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert r.returncode == 0, r.stderr


def test_a_decode_failure_emits_every_content_column_unsuffixed(tmp_path):
    """Mutation: `_failed_content()` dropped from `_signal_row`'s failure path --
    a part where every row failed writes a table with no `vad_*` columns at all,
    and the merge across parts then fills them with nulls that read as measured.

    ⚠️ Goes through `_signal_row`, not `_failed_content`. Calling the helper
    directly tests that the helper works and says nothing about whether anything
    calls it -- measured: the mutant survived that version of this test.
    """
    import pandas as pd

    from eda.config import EdaConfig
    from eda.driver import _signal_row

    cfg = EdaConfig(root=tmp_path, out=tmp_path / "out")
    row = pd.Series({"file_id": "s:missing.wav", "source_name": "s",
                     "path": "missing.wav", "orig_sr": 16000})
    scalars, _ = _signal_row(cfg, row, "A")
    assert scalars["signal_ok"] is False
    assert set(COLUMNS) <= set(scalars), "every content column, on a failed row"
    assert scalars["vad_ok"] is False and scalars["vad_error"] == "decode failed"


def test_batching_the_chunks_is_not_the_same_as_streaming_them():
    """🔴 The model takes `(batch, 512)` and returns `(batch, 1)`, which looks
    like a free speed-up. It is not: the batch dimension is independent
    streams, so batching one file's chunks throws away the recurrent state.

    This test does not guard our code -- it guards a *temptation*. Measured on
    this clip: sequential mean 0.0186 against batched 0.0467, max absolute
    difference 0.107. Both are plausible numbers and only one is the model used
    as designed, which is why the difference is asserted here rather than left
    in a comment.
    """
    import torch

    from models.vendor.silero_vad import load_vad, speech_probabilities

    rng = np.random.default_rng(1)
    x = np.concatenate([rng.standard_normal(SR * 2) * 0.005,
                        rng.standard_normal(SR * 2) * 0.20,
                        rng.standard_normal(SR * 2) * 0.005]).astype(np.float32)
    sequential = speech_probabilities(x, SR)

    model = load_vad()
    n = len(sequential)
    chunks = torch.stack([torch.from_numpy(x[i * CHUNK:(i + 1) * CHUNK])
                          for i in range(n)])
    model.reset_states()
    with torch.no_grad():
        batched = model(chunks, SR).numpy().ravel()

    assert np.abs(sequential - batched).max() > 0.05, (
        "if these ever agree, re-read the upstream model: this test encodes "
        "that they must not, and a version that batches correctly would change "
        "what `speech_probabilities` should do")
