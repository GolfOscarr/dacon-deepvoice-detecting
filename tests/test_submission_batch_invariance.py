"""What the submitted file may and may not depend on, measured on the CSV.

Rule 2.4 forbids a prediction for one test file from using any other file in the
cohort. The masking half of that is already pinned deterministically by
`tests/test_heads.py::test_frame_max_is_padding_safe` and exercised through the
model by `tests/test_model.py`'s blend fixture; this file does not duplicate it.

What is here is the part those do not reach: the end of the pipeline, and a
separation that matters for reproducing a submission. Measured on the trained
candidate-A model (real BEATs weights) on one H200, for one fixed file:

    different neighbour CONTENT, same batch size   0.0        (bitwise)
    neighbour 50x louder,        same batch size   0.0        (bitwise)
    padding 10 s -> 30 s,        same batch size   1.2e-07
    batch size 8 -> 32,          same content      6.8e-04

Only the first two bear on rule 2.4, and they hold exactly: no other file's
content reaches a prediction. The last line is NOT a rule-2.4 violation -- it is
cuBLAS choosing a different reduction for a different batch shape, and it appears
with identical neighbours -- but at 6.8e-04 across 1200 files (median 1.5e-05 on
VOICE_FAKE_PROB, 833/1200 above 1e-05) it is far larger than the ranking
resolution this project protects elsewhere, so the submission has to be written
at a pinned batch size.

⚠️ `clip_weight` is forced on in the fixture. At its 1.0 default the submitted
score ignores `frame_max`, which makes any frame_max assertion vacuous --
models/config.py says so and tells tests not to inherit it.
"""
import dataclasses
import inspect
import tempfile
from pathlib import Path

import pytest
import torch

from metrics.dacon import PREDICTION_COLUMNS
from metrics.submission import write_submission
from models.config import AudioConfig, load_model_config
from models.model import DeepVoiceNet
from models.outputs import branch_logit, to_probability

HEADS = ("file", "voice", "music", "v_pres", "m_pres")


@pytest.fixture(scope="module")
def stub_model():
    cfg = load_model_config("configs/a_stub.yaml")
    cfg = dataclasses.replace(cfg, branches={
        name: dataclasses.replace(br, head=dataclasses.replace(br.head, clip_weight=0.5))
        for name, br in cfg.branches.items()})
    torch.manual_seed(0)
    return DeepVoiceNet(cfg).eval()


def _submitted(model, wav, lengths):
    """The probabilities we would upload -- not `clip_logits`.

    `clip_logits` is attention-pooled and masked inside the head, so it is
    invariant to padding even where the blend is not. Asserting on it is how an
    earlier version of this defect escaped (PROGRESS: the submitted probability
    moved 0.519 -> 0.847 while the test watched `clip_logits`).
    """
    with torch.no_grad():
        out = model(wav, lengths)
    return {k: to_probability(branch_logit(out[k], model.cfg.branches[k].head),
                              model.cfg.output).double() for k in out}


def test_a_prediction_never_reads_another_files_content(stub_model):
    """Rule 2.4 itself: hold the batch SHAPE fixed and vary only what else is in
    it. Any movement here is another file's content reaching a prediction, which
    is the thing the rule forbids -- as distinct from the batch-shape jitter
    documented in the module docstring, which is not.

    🔴 MUTATION (observed): add `wav = wav / wav.abs().max()` to
    `Frontend.forward` -- a batch-peak normalisation instead of a per-row one --
    and this fails, as does the CSV test below. Two mutations that did NOT break
    it are worth recording so nobody re-tries them: removing the per-row length
    zeroing (row 0 is already zeros past its length), and a batch-max
    normalisation in `prepare_waveform` (the training loop calls that, not
    `model.forward`). The model has no cross-row operation anywhere, which is
    why this passes -- and why the guard is against one being introduced."""
    sr = AudioConfig().sample_rate
    g = torch.Generator().manual_seed(5)
    target = torch.randn(sr * 5, generator=g) * 0.1

    def batch(seed, scale):
        gg = torch.Generator().manual_seed(seed)
        wav = torch.zeros(6, sr * 9)
        wav[0, : sr * 5] = target
        for i in range(1, 6):
            wav[i] = torch.randn(sr * 9, generator=gg) * scale
        return _submitted(stub_model, wav, torch.tensor([sr * 5] + [sr * 9] * 5))

    ref = batch(11, 1.0)
    for seed, scale, label in ((999, 5.0, "different content"),
                               (11, 50.0, "50x louder"),
                               (123, 0.01, "near-silent")):
        got = batch(seed, scale)
        for head in HEADS:
            assert ref[head][0] == got[head][0], (
                f"{head}: row 0 moved when its neighbours changed ({label})")


def test_the_uploaded_csv_is_identical_when_batch_membership_is_permuted(stub_model):
    """The same statement on the artifact itself. Batch size is held at 2
    throughout; only WHICH file shares each batch changes."""
    sr = AudioConfig().sample_rate
    g = torch.Generator().manual_seed(3)
    files = [torch.randn(int(sr * d), generator=g) * 0.1 for d in (4.5, 7.0, 5.25, 9.0)]
    ids = [f"f{i}" for i in range(len(files))]
    tmp = Path(tempfile.mkdtemp())

    def submission(order):
        rows = {}
        for start in range(0, len(order), 2):
            chunk = order[start:start + 2]
            width = max(len(files[i]) for i in chunk)
            wav = torch.zeros(len(chunk), width)
            for j, i in enumerate(chunk):
                wav[j, : len(files[i])] = files[i]
            got = _submitted(stub_model, wav,
                             torch.tensor([len(files[i]) for i in chunk]))
            for j, i in enumerate(chunk):
                rows[ids[i]] = {c: float(got[h][j]) for c, h in zip(PREDICTION_COLUMNS, HEADS)}
        preds = {c: [rows[i][c] for i in ids] for c in PREDICTION_COLUMNS}
        path = tmp / f"sub_{'-'.join(map(str, order))}.csv"
        write_submission(ids, preds, path, validate=False)
        return path.read_bytes()

    # 🔴 Same mutation as above: a batch-peak normalisation in Frontend.forward
    # turns these two byte strings different.
    assert submission([0, 1, 2, 3]) == submission([0, 3, 2, 1]), \
        "the uploaded CSV changed when batch membership was permuted"


def test_the_inference_batch_size_comes_from_the_config():
    """The mitigation for the batch-shape jitter. `predict`'s default and
    `runtime.batch_size` agree today by coincidence, not by wiring, so a
    submission run that takes the default while local validation passes another
    would disagree by ~7e-04 with nothing reporting it.

    🔴 MUTATION (observed): change `predict`'s default to 16 and this fails."""
    cfg = load_model_config("configs/a_shared_trunk.yaml")
    from training.validate import predict
    default = inspect.signature(predict).parameters["batch_size"].default
    assert default == cfg.runtime.batch_size, (
        f"predict() defaults to batch_size={default} while the shipped config says "
        f"{cfg.runtime.batch_size}: the submission path must pass "
        f"cfg.runtime.batch_size explicitly rather than rely on them agreeing")
