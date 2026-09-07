"""The assembly. Two properties matter most: A and B are the same class, and a
file's score does not depend on what is batched with it."""

import dataclasses

import pytest
import torch

from metrics.dacon import PREDICTION_COLUMNS
from models.config import load_model_config
from models.model import DeepVoiceNet, load_checkpoint, save_checkpoint

SR = 16_000


def _cfg(name):
    """A shipped stub config -- the real thing, not a mutation of it."""
    return load_model_config(f"configs/{name}.yaml")


def _model(name):
    return DeepVoiceNet(_cfg(name)).eval()


@pytest.fixture(params=["a_stub", "b_stub"])
def model(request):
    return _model(request.param)


# --------------------------------------------------------------------------- #
# shape and structure

def test_produces_every_submission_column(model):
    out = model(torch.randn(2, SR * 5))
    assert sorted(model.columns.values()) == sorted(PREDICTION_COLUMNS)
    for branch in model.cfg.branches:
        assert out[branch]["clip_logits"].shape == (2,)


def test_a_and_b_share_branch_structure():
    """docs/architecture/03: B strictly contains A."""
    a, b = _model("a_stub"), _model("b_stub")
    assert set(a.heads) == set(b.heads)
    assert len(a.frontends) == 1 and len(b.frontends) == 2
    assert b.n_parameters() > a.n_parameters()


def test_every_file_passes_through_every_branch():
    """The branches are outputs, not input types -- nothing is routed."""
    m = _model("b_stub")
    out = m(torch.randn(1, SR * 4))
    for branch in ("voice", "music", "file"):
        assert torch.isfinite(out[branch]["clip_logits"]).all()


@pytest.mark.parametrize("seconds", [4, 7, 60])
def test_handles_the_documented_duration_range(model, seconds):
    out = model(torch.randn(1, SR * seconds))
    assert out["file"]["frame_logits"].shape[1] == seconds * 50


def test_file_branch_aligns_two_frame_rates():
    """B's file branch reads both frontends, which disagree on fps."""
    cfg = _cfg("b_stub")
    fes = dict(cfg.frontends)
    fes["speech"] = dataclasses.replace(fes["speech"], fps=25.0)
    m = DeepVoiceNet(dataclasses.replace(cfg, frontends=fes)).eval()
    out = m(torch.randn(1, SR * 4))
    # align_to is `audio` at 50 fps, so the file branch runs on the audio grid.
    assert out["file"]["frame_logits"].shape[1] == 200
    assert out["voice"]["frame_logits"].shape[1] == 100        # speech grid, 25 fps


# --------------------------------------------------------------------------- #
# 🔴 rule 2.4 — per-file independence

def test_score_does_not_depend_on_the_batch(model):
    """A file scored alone and inside a batch of longer files must agree.

    Not bitwise: kernel selection varies with batch shape. Tolerance plus rank
    preservation is the real contract (docs/architecture/01 §1.4).
    """
    torch.manual_seed(0)
    solo = torch.randn(1, SR * 4)
    alone = model(solo, torch.tensor([SR * 4]))

    others = torch.randn(7, SR * 9) * 10                       # longer, louder
    batch = torch.cat([torch.nn.functional.pad(solo, (0, SR * 5)), others])
    lengths = torch.tensor([SR * 4] + [SR * 9] * 7)
    batched = model(batch, lengths)

    for branch in model.cfg.branches:
        a = alone[branch]["clip_logits"][0]
        b = batched[branch]["clip_logits"][0]
        assert torch.allclose(a, b, atol=1e-4), f"{branch}: {a.item()} vs {b.item()}"


def test_ranking_over_a_canned_set_is_batch_invariant(model):
    """Ranking is what EER reads, so it is the property that must hold."""
    torch.manual_seed(1)
    files = [torch.randn(1, SR * (4 + i)) for i in range(6)]
    solo = torch.stack([model(f)["file"]["clip_logits"][0] for f in files])

    width = max(f.shape[-1] for f in files)
    batch = torch.cat([torch.nn.functional.pad(f, (0, width - f.shape[-1])) for f in files])
    lengths = torch.tensor([f.shape[-1] for f in files])
    together = model(batch, lengths)["file"]["clip_logits"]

    assert torch.equal(solo.argsort(), together.argsort())


def test_padding_content_cannot_leak_into_a_score(model):
    """Same file, two different pad fillings -> identical score.

    🔴 Asserted on the **submitted probabilities**, not on clip_logits. An
    earlier version of this test checked clip_logits only and passed while the
    submission path was broken: attention already excludes padding, but the
    frame_max term did not, so the number we would have uploaded moved from
    0.519 to 0.847 depending on what shared the batch.
    """
    torch.manual_seed(2)
    x = torch.randn(1, SR * 4)
    lengths = torch.tensor([SR * 4])
    quiet = model.submission_probs(model(torch.nn.functional.pad(x, (0, SR * 4)), lengths))
    loud = torch.cat([x, torch.randn(1, SR * 4) * 50], dim=-1)
    noisy = model.submission_probs(model(loud, lengths))
    for column in quiet:
        assert torch.allclose(quiet[column], noisy[column], atol=1e-9), column


def test_submitted_probabilities_are_batch_invariant(model):
    """The end-to-end contract: what we upload must not move with the batch."""
    torch.manual_seed(3)
    solo = torch.randn(1, SR * 4)
    alone = model.submission_probs(model(solo, torch.tensor([SR * 4])))

    others = torch.randn(7, SR * 9) * 10
    batch = torch.cat([torch.nn.functional.pad(solo, (0, SR * 5)), others])
    lengths = torch.tensor([SR * 4] + [SR * 9] * 7)
    together = model.submission_probs(model(batch, lengths))

    for column in alone:
        assert torch.allclose(alone[column], together[column][:1], atol=1e-6), column


def test_frame_max_respects_the_mask_recorded_by_the_head():
    """The head carries its mask so callers cannot forget it."""
    m = _model("b_stub")
    out = m(torch.nn.functional.pad(torch.randn(1, SR * 4), (0, SR * 4)),
            torch.tensor([SR * 4]))
    assert out["file"]["mask"].shape == out["file"]["frame_logits"].shape
    assert out["file"]["mask"][0, :200].all()
    assert not out["file"]["mask"][0, 200:].any()


def test_deterministic_in_eval_mode(model):
    x = torch.randn(2, SR * 4)
    assert torch.equal(model(x)["file"]["clip_logits"], model(x)["file"]["clip_logits"])


# --------------------------------------------------------------------------- #
# checkpoints

def test_checkpoint_round_trip_carries_the_config(tmp_path):
    m = _model("b_stub")
    path = tmp_path / "model.pt"
    save_checkpoint(m, path)
    back = load_checkpoint(path)
    assert back.cfg == m.cfg
    x = torch.randn(1, SR * 4)
    assert torch.allclose(m(x)["file"]["clip_logits"], back(x)["file"]["clip_logits"])


def test_checkpoint_with_mismatched_weights_fails_loudly(tmp_path):
    """A strict load is only an assertion if it actually raises."""
    m = _model("b_stub")
    path = tmp_path / "model.pt"
    save_checkpoint(m, path)
    blob = torch.load(path, weights_only=False)
    blob["state_dict"].pop(next(iter(blob["state_dict"])))
    torch.save(blob, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(path)


def test_non_checkpoint_file_is_rejected(tmp_path):
    path = tmp_path / "junk.pt"
    torch.save({"weights": 1}, path)
    with pytest.raises(ValueError, match="not a DeepVoiceNet checkpoint"):
        load_checkpoint(path)


# --------------------------------------------------------------------------- #
# training-only appendages

def test_distill_head_is_aux_not_a_column():
    m = _model("b_stub")
    out = m(torch.randn(1, SR * 4))
    assert "distill_emb" in out["_aux"]
    assert "_aux" not in m.columns


def test_c_lite_separation_head_is_one_config_flip():
    cfg = _cfg("b_stub")
    cfg = dataclasses.replace(cfg, aux=dataclasses.replace(cfg.aux, separation_head=True))
    out = DeepVoiceNet(cfg).eval()(torch.randn(1, SR * 4))
    assert out["_aux"]["separation"].shape[-1] == 2 * cfg.aux.stft_bins


def test_a_has_no_aux_by_default():
    assert "_aux" not in _model("a_stub")(torch.randn(1, SR * 4))


# --------------------------------------------------------------------------- #
# 🔴 config/model divergence

def test_no_config_field_is_silently_ignored():
    """Every config field must be read by the model, or be explicitly listed as
    consumed elsewhere.

    This guard exists because an audit found 19 validated-but-ignored fields at
    once, including `freeze` (the frontend was fully trainable) and
    `file_head.mode` (all three settings produced the same output). A knob that
    validates and then does nothing is worse than a missing knob: it makes an
    ablation report a difference it never tested.
    """
    import pathlib
    import re

    from models.model import CONSUMED_ELSEWHERE

    src = "\n".join(
        (pathlib.Path("models") / f"{m}.py").read_text()
        for m in ("model", "frontends", "heads", "losses", "outputs", "utils", "audio"))

    cfg = _cfg("b_stub")

    def walk(obj, prefix=""):
        for f in dataclasses.fields(obj):
            value, name = getattr(obj, f.name), f"{prefix}{f.name}"
            if dataclasses.is_dataclass(value):
                yield from walk(value, name + ".")
            elif isinstance(value, dict) and value and dataclasses.is_dataclass(
                    next(iter(value.values()))):
                yield from walk(next(iter(value.values())), name + ".")
            else:
                yield name, f.name

    ignored = [full for full, leaf in walk(cfg)
               if full not in CONSUMED_ELSEWHERE
               and not re.search(rf"\.{re.escape(leaf)}\b", src)]
    assert not ignored, (
        "config fields neither read by the model nor listed in CONSUMED_ELSEWHERE: "
        f"{ignored}. Implement them, or document who consumes them.")


def test_consumed_elsewhere_has_no_stale_entries():
    """The allowlist must not outlive the thing it excuses."""
    from models.model import CONSUMED_ELSEWHERE
    cfg = _cfg("b_stub")
    valid = set()

    def walk(obj, prefix=""):
        for f in dataclasses.fields(obj):
            value, name = getattr(obj, f.name), f"{prefix}{f.name}"
            if dataclasses.is_dataclass(value):
                walk(value, name + ".")
            elif isinstance(value, dict) and value and dataclasses.is_dataclass(
                    next(iter(value.values()))):
                walk(next(iter(value.values())), name + ".")
            else:
                valid.add(name)

    walk(cfg)
    assert not (set(CONSUMED_ELSEWHERE) - valid), \
        f"CONSUMED_ELSEWHERE names fields that no longer exist: {set(CONSUMED_ELSEWHERE) - valid}"


def test_freeze_is_honoured_end_to_end():
    """The frontend must receive no gradient when the config says it is frozen."""
    model, _ = _model("b_stub"), None
    fe_params = [p for f in model.frontends.values() for p in f.parameters()
                 if "freq_pool" not in str(f)]
    out = model(torch.randn(2, SR * 4))
    out["file"]["clip_logits"].sum().backward()
    encoder = [p for f in model.frontends.values()
               for n, p in f.named_parameters() if not n.startswith("freq_pool")]
    assert not any(p.requires_grad for p in encoder)
    assert all(p.grad is None for p in encoder)


def test_all_three_file_head_modes_actually_differ():
    """G3 says build all three and compare -- so they must not be the same thing."""
    probs = {}
    for mode in ("learned", "noisy_or", "max"):
        cfg = _cfg("b_stub")
        cfg = dataclasses.replace(cfg, file_head=dataclasses.replace(cfg.file_head, mode=mode))
        torch.manual_seed(0)
        model = DeepVoiceNet(cfg).eval()
        torch.manual_seed(7)
        out = model(torch.randn(4, SR * 4))
        probs[mode] = model.submission_probs(out)["FILE_FAKE_PROB"]

    assert not torch.allclose(probs["learned"], probs["noisy_or"])
    assert not torch.allclose(probs["learned"], probs["max"])
    assert not torch.allclose(probs["noisy_or"], probs["max"])
    # noisy-OR is >= max by construction, for probabilities in [0, 1]
    assert (probs["noisy_or"] >= probs["max"] - 1e-9).all()


def test_tiling_is_refused_rather_than_silently_ignored():
    cfg = _cfg("b_stub")
    cfg = dataclasses.replace(
        cfg, segmentation=dataclasses.replace(cfg.segmentation, mode="tiling"))
    with pytest.raises(NotImplementedError, match="segmentation"):
        DeepVoiceNet(cfg)


def test_band_hz_is_actually_applied():
    """09 B7 / D9: the low-band member must be buildable, not just configurable."""
    cfg = _cfg("b_stub")
    cfg = dataclasses.replace(cfg, audio=dataclasses.replace(cfg.audio, band_hz=(0.0, 4000.0)))
    torch.manual_seed(0)
    banded = DeepVoiceNet(cfg).eval()
    torch.manual_seed(0)
    full = DeepVoiceNet(_cfg("b_stub")).eval()

    x = torch.randn(2, SR * 4)
    assert not torch.allclose(banded(x)["file"]["clip_logits"],
                              full(x)["file"]["clip_logits"])
    # ...and a signal that is already low-band should be nearly unaffected.
    t = torch.arange(SR * 4) / SR
    low = torch.sin(2 * torch.pi * 500 * t)[None].repeat(2, 1)
    assert torch.allclose(banded(low)["file"]["clip_logits"],
                          full(low)["file"]["clip_logits"], atol=1e-3)


def test_submission_probs_returns_five_float64_columns(model):
    probs = model.submission_probs(model(torch.randn(3, SR * 4)))
    assert sorted(probs) == sorted(PREDICTION_COLUMNS)
    for col, p in probs.items():
        assert p.dtype == torch.float64 and p.shape == (3,)
        assert (p > 0).all() and (p < 1).all()
