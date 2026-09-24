"""XLS-R 300M, the speech trunk of configs/c_first_run.yaml (docs/training/07 §2).

Each invariant names the mutation that breaks it. Skipped when the snapshot is
absent; point DACON_XLSR_WEIGHTS at a directory holding
facebook/wav2vec2-xls-r-300m (config.json + pytorch_model.bin) to run them.

The GPU tests at the bottom measure a c_first_run training step and inference
throughput; run them with `-s` to see the numbers.
"""

import dataclasses
import os
import time

import pytest
import torch
from torch import nn

from models.config import AdapterConfig, AudioConfig, FreqPoolConfig, load_model_config
from models.frontends import LoRALinear, _apply_lora, build_frontend

WEIGHTS = os.environ.get("DACON_XLSR_WEIGHTS", "/data/project/private/dacon-weights/xlsr-300m")
BEATS = os.environ.get("DACON_BEATS_WEIGHTS", "/data/project/private/dacon-weights/beats")
_HAVE = os.path.exists(os.path.join(WEIGHTS, "config.json"))
_HAVE_BEATS = os.path.exists(os.path.join(BEATS, "BEATs_iter3_plus_AS2M.pt"))
needs_xlsr = pytest.mark.skipif(not _HAVE, reason=f"no XLS-R snapshot under {WEIGHTS}")
needs_both = pytest.mark.skipif(not (_HAVE and _HAVE_BEATS), reason="needs XLS-R and BEATs")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

SR = AudioConfig().sample_rate
SIX = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


def _cfg(**kw):
    cfg = load_model_config("configs/c_first_run.yaml")
    fe = dataclasses.replace(cfg.frontends["speech"], **{"weights": WEIGHTS, **kw})
    return fe, cfg.audio


@pytest.fixture(scope="module")
def xlsr():
    fe, audio = _cfg()
    return build_frontend(fe, audio).eval()


def _ragged(seed=0):
    """Two rows whose lengths are NOT multiples of the 320-sample hop."""
    g = torch.Generator().manual_seed(seed)
    n0, n1 = SR * 4 + 123, SR * 2 + 77
    wav = torch.zeros(2, n0)
    wav[0] = torch.randn(n0, generator=g) * 0.1
    wav[1, :n1] = torch.randn(n1, generator=g) * 0.3
    return wav, torch.tensor([n0, n1])


# --------------------------------------------------------------------------- #
# construction

@needs_xlsr
def test_builds_from_disk_with_no_network(monkeypatch):
    """The test server has no network. With the Hub switched off -- what
    script.py sets -- construction needs nothing but the directory."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    fe, audio = _cfg(layers=1, adapter=AdapterConfig(kind="none"))
    enc = build_frontend(fe, audio)
    assert enc.output_dim == 1024 and enc.fps == 50.0


@needs_xlsr
def test_matches_the_upstream_wav2vec2_forward_exactly(xlsr):
    """Our encoder loop replaces `Wav2Vec2Model.forward` (see the class
    docstring). It must compute the same function on valid frames: MUTATION --
    skip the pre-pos-conv zeroing, or drop the final LayerNorm, and this fails."""
    from transformers import Wav2Vec2Model
    ref = Wav2Vec2Model.from_pretrained(WEIGHTS, local_files_only=True, num_hidden_layers=12,
                                        attn_implementation="sdpa").eval()
    wav, lengths = _ragged()
    with torch.no_grad():
        ours, mask = xlsr(wav, lengths)
        x = xlsr._normalise(wav, lengths)
        am = (torch.arange(x.shape[1])[None] < lengths[:, None]).long()
        theirs = ref(x, attention_mask=am).last_hidden_state
    for i in range(2):
        n = int(mask[i].sum())
        assert torch.allclose(ours[i, :n], theirs[i, :n], atol=1e-5), i


@needs_xlsr
def test_normalisation_matches_the_feature_extractor_on_the_valid_prefix(xlsr):
    from transformers import Wav2Vec2FeatureExtractor
    fx = Wav2Vec2FeatureExtractor.from_pretrained(WEIGHTS, local_files_only=True)
    wav, lengths = _ragged()
    ours = xlsr._normalise(wav, lengths)
    for i in range(2):
        n = int(lengths[i])
        ref = torch.tensor(fx(wav[i, :n].numpy(), sampling_rate=SR).input_values[0])
        assert torch.allclose(ours[i, :n], ref, atol=1e-5)
        assert torch.all(ours[i, n:] == 0), "padding must stay zero after normalising"


@needs_xlsr
@pytest.mark.parametrize("kw, match", [
    ({"fps": 6.25}, "fps"),
    ({"freq_pool": FreqPoolConfig(kind="gem")}, "frequency axis"),
    ({"output_dim": 768}, "hidden_size"),
    ({"weights": None}, "StubFrontend"),
    ({"layers": 25}, "out of range"),
])
def test_a_wrong_config_is_refused(kw, match):
    fe, audio = _cfg(**kw)
    with pytest.raises(ValueError, match=match):
        build_frontend(fe, audio)


# --------------------------------------------------------------------------- #
# truncation, LoRA, freeze

@needs_xlsr
def test_truncation_deletes_layers_rather_than_skipping_them(xlsr):
    assert xlsr.n_layers == 12
    assert len(xlsr.model.encoder.layers) == 12, "unused blocks must not cost parameters"
    assert not any(".layers.12." in k for k in xlsr.state_dict())


@needs_xlsr
def test_lora_wraps_all_six_projections_in_every_kept_layer(xlsr):
    assert xlsr.n_lora == 12 * 6
    for layer in xlsr.model.encoder.layers:
        for mod in (layer.attention.q_proj, layer.attention.k_proj, layer.attention.v_proj,
                    layer.attention.out_proj, layer.feed_forward.intermediate_dense,
                    layer.feed_forward.output_dense):
            assert isinstance(mod, LoRALinear)


@needs_xlsr
def test_an_adapter_target_that_matches_nothing_is_an_error():
    """🔴 Per target, not in total: MUTATION -- raise only when the sum is zero
    (the old BEATs check) and `[q_proj, not_a_layer]` builds, half-adapted."""
    fe, audio = _cfg(layers=1, adapter=AdapterConfig(kind="lora",
                                                     targets=("q_proj", "not_a_layer")))
    with pytest.raises(ValueError, match="not_a_layer.*matched no nn.Linear"):
        build_frontend(fe, audio)


@needs_xlsr
def test_lora_is_the_identity_at_init_and_every_target_reaches_the_output(xlsr):
    """Zero-init `lora_b` leaves the checkpoint's function bitwise intact, and a
    nonzero `lora_b` on ANY of the six targets moves the output -- i.e. the
    forward calls the wrapped module rather than reading its `.weight`."""
    fe_none, audio = _cfg(adapter=AdapterConfig(kind="none"))
    plain = build_frontend(fe_none, audio).eval()
    wav, lengths = _ragged(1)
    with torch.no_grad():
        base = xlsr(wav, lengths)[0]
        assert torch.equal(base, plain(wav, lengths)[0])
        layer = xlsr.model.encoder.layers[5]
        mods = {"q_proj": layer.attention.q_proj, "k_proj": layer.attention.k_proj,
                "v_proj": layer.attention.v_proj, "out_proj": layer.attention.out_proj,
                "fc1": layer.feed_forward.intermediate_dense,
                "fc2": layer.feed_forward.output_dense}
        for name, mod in mods.items():
            mod.lora_b.normal_(0, 0.05)
            try:
                moved = (xlsr(wav, lengths)[0] - base).abs().max().item()
            finally:
                mod.lora_b.zero_()
            assert moved > 1e-3, f"{name}: a trained lora_b did not reach the output"


@needs_xlsr
def test_only_adapters_train_under_freeze_and_the_cnn_never_does(xlsr):
    trainable = {n for n, p in xlsr.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable), trainable
    assert not any(p.requires_grad for p in xlsr.model.feature_extractor.parameters())

    fe, audio = _cfg(layers=1, freeze=False)
    unfrozen = build_frontend(fe, audio)
    assert not any(p.requires_grad for p in unfrozen.model.feature_extractor.parameters())
    assert any(p.requires_grad for n, p in unfrozen.named_parameters()
               if ".layers.0." in n and "lora_" not in n)


# --------------------------------------------------------------------------- #
# rule 2.4 and framing

@needs_xlsr
def test_a_file_scores_the_same_alone_and_inside_a_padded_batch(xlsr):
    """MUTATION: take the normalisation statistics over the padded row, or
    drop the attention mask, and the short row moves by O(0.1)."""
    wav, lengths = _ragged(2)
    with torch.no_grad():
        batched, bmask = xlsr(wav, lengths)
        for i in range(2):
            n = int(lengths[i])
            solo, smask = xlsr(wav[i:i + 1, :n], lengths[i:i + 1])
            t = int(smask.sum())
            assert int(bmask[i].sum()) == t
            err = (batched[i, :t] - solo[0, :t]).abs().max().item()
            assert err < 1e-4, f"row {i}: {err}"


@needs_xlsr
def test_what_fills_the_padding_cannot_reach_the_features(xlsr):
    wav, lengths = _ragged(3)
    noisy = wav.clone()
    noisy[1, int(lengths[1]):] = torch.randn(wav.shape[1] - int(lengths[1])) * 5
    with torch.no_grad():
        a, m = xlsr(wav, lengths)
        b, _ = xlsr(noisy, lengths)
    n = int(m[1].sum())
    assert (a[1, :n] - b[1, :n]).abs().max().item() < 1e-5


@needs_xlsr
@pytest.mark.parametrize("n", [400, 719, SR * 4, SR * 4 + 123, SR * 7 + 319])
def test_frame_count_is_the_conv_stack_s_own_arithmetic(xlsr, n):
    """MUTATION: fall back to the nominal ceil(n / 320) and 4 s claims 200 frames
    for an encoder that emits 199 -- one frame made of padding, masked in."""
    with torch.no_grad():
        feats, mask = xlsr(torch.randn(1, n) * 0.1, torch.tensor([n]))
    assert feats.shape[1] == xlsr._frames_for(n) == int(mask.sum())
    assert xlsr._frames_for(torch.tensor([n])).item() == xlsr._frames_for(n)
    if n == SR * 4:
        assert feats.shape[1] == 199


@needs_xlsr
def test_audio_shorter_than_one_receptive_field_still_yields_a_frame(xlsr):
    with torch.no_grad():
        feats, mask = xlsr(torch.randn(1, 250) * 0.1, torch.tensor([250]))
    assert feats.shape[1] == 1 and bool(mask.all())
    assert torch.isfinite(feats).all()


@needs_xlsr
def test_no_hidden_generator_in_the_forward_pass(xlsr):
    """Upstream `Wav2Vec2EncoderStableLayerNorm` draws `torch.rand([])` per layer
    on every forward. MUTATION: call `self.model(...)` instead of our loop and the
    global RNG state moves."""
    xlsr.train()
    try:
        wav, lengths = _ragged(4)
        state = torch.get_rng_state()
        a = xlsr(wav, lengths)[0]
        assert torch.equal(state, torch.get_rng_state())
        assert torch.equal(a, xlsr(wav, lengths)[0])
    finally:
        xlsr.eval()


# --------------------------------------------------------------------------- #
# the shared LoRA helper

class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        from models.vendor.beats.modules import GLU_Linear
        self.fc1 = GLU_Linear(8, 16, "swish")
        self.fc2 = nn.Linear(16, 8)
        self.norm = nn.LayerNorm(8)


def test_a_glu_projection_is_adapted_through_its_inner_linear():
    """BEATs checkpoints with `activation_fn: glu` build `fc1` as GLU_Linear, not
    nn.Linear. The shipped checkpoint is gelu, but a GLU one must not silently
    skip `fc1`: its inner `.linear` is wrapped."""
    blk = _Block()
    x = torch.randn(2, 3, 8)
    before = blk.fc2(blk.fc1(x))
    assert _apply_lora(blk, ("fc1", "fc2"), 4, 8.0, 0.0) == 2
    assert isinstance(blk.fc1.linear, LoRALinear)
    assert torch.equal(blk.fc2(blk.fc1(x)), before)
    with torch.no_grad():
        blk.fc1.linear.lora_b.normal_()
    assert not torch.allclose(blk.fc2(blk.fc1(x)), before)


def test_a_target_that_is_neither_linear_nor_glu_is_refused():
    with pytest.raises(ValueError, match="not an nn.Linear"):
        _apply_lora(_Block(), ("norm",), 4, 8.0, 0.0)


# --------------------------------------------------------------------------- #
# the whole c_first_run model

def _c_first_run():
    from models.model import DeepVoiceNet
    cfg = load_model_config("configs/c_first_run.yaml")
    paths = {"audio": BEATS, "speech": WEIGHTS}
    cfg = dataclasses.replace(cfg, frontends={
        k: dataclasses.replace(v, weights=paths[k]) for k, v in cfg.frontends.items()})
    return DeepVoiceNet(cfg)


@needs_both
def test_c_first_run_builds_and_round_trips_through_a_checkpoint(tmp_path):
    from models.model import load_checkpoint, save_checkpoint, shipped_weights
    assert shipped_weights(tmp_path) is None
    model = _c_first_run().eval()
    assert model.frontends["audio"].n_layers == 12
    assert model.frontends["audio"].n_lora == 12 * 6
    assert model.frontends["speech"].n_lora == 12 * 6
    trainable = model.n_parameters(True)
    assert 0 < trainable < model.n_parameters() // 20
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.01)       # make adapters non-identity

    # 🔴 The stored config names the training machine's absolute paths, which do
    # not exist on the offline test server. Simulate that: point them nowhere.
    path = tmp_path / "scored.pt"
    save_checkpoint(model, path)
    blob = torch.load(path, weights_only=False)
    for fe in blob["config"]["frontends"].values():
        fe["weights"] = str(tmp_path / "gone" / fe["name"])
    torch.save(blob, path)
    with pytest.raises(OSError):                    # no override -> cannot even build
        load_checkpoint(path)
    print(f"\n[c_first_run] scored.pt {path.stat().st_size / 2**20:.0f} MiB")

    # script.py's layout: model/weights/<frontend>/ beside scored.pt.
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "audio").symlink_to(BEATS)
    (tmp_path / "weights" / "speech").symlink_to(WEIGHTS)
    shipped = shipped_weights(tmp_path)
    assert shipped == {"audio": str(tmp_path / "weights" / "audio"),
                       "speech": str(tmp_path / "weights" / "speech")}
    restored = load_checkpoint(path, weights=shipped)
    with pytest.raises(ValueError, match="not in the checkpoint"):
        load_checkpoint(path, weights={"nope": BEATS})
    wav, lengths = _ragged(5)
    with torch.no_grad():
        a = model.submission_probs(model(wav, lengths))
        b = restored.submission_probs(restored(wav, lengths))
    for col in a:
        assert torch.equal(a[col], b[col]), col


def _step_stats(model, b, seconds, dtype):
    from models.config import LossConfig
    from models.losses import multitask_loss
    dev = "cuda"
    n = SR * seconds
    wav = torch.randn(b, n, device=dev) * 0.1
    lengths = torch.tensor([n] + [n - 12_345] * (b - 1), device=dev)
    targets = {k: torch.randint(0, 2, (b,), device=dev).float()
               for k in ("voice_fake", "music_fake", "file_fake",
                         "voice_present", "music_present")}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(4):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=dtype):
            out = model(wav, lengths)
            loss, _ = multitask_loss(out, targets, model.cfg, LossConfig())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    assert torch.isfinite(loss)
    grads_reached = model.frontends["speech"].model.encoder.layers[0].attention.q_proj
    assert grads_reached.lora_b.abs().sum() > 0, "speech LoRA never updated"
    return torch.cuda.max_memory_allocated() / 2 ** 30, min(times[1:])


@needs_both
@needs_gpu
def test_c_first_run_trains_on_a_gpu_under_bf16():
    model = _c_first_run().cuda().train()
    for seconds in (30, 60):
        gib, sec = _step_stats(model, 2, seconds, torch.bfloat16)
        print(f"\n[c_first_run train] batch 2 x {seconds}s bf16: "
              f"peak {gib:.2f} GiB, {sec:.3f} s/step")


@needs_both
@needs_gpu
def test_c_first_run_inference_throughput_fp16():
    """Seconds of audio per second at batch 8 x 60 s, fp16 autocast."""
    model = _c_first_run().cuda().eval()
    b, n = 8, SR * 60
    wav = torch.randn(b, n, device="cuda") * 0.1
    lengths = torch.full((b,), n, device="cuda")
    times = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(4):
            torch.cuda.synchronize()
            t0 = time.time()
            probs = model.submission_probs(model(wav, lengths))
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    for v in probs.values():
        assert torch.isfinite(v).all()
    sec = min(times[1:])
    rate = b * 60 / sec
    l4_minutes = 1200 * 60 / (rate / 6) / 60
    print(f"\n[c_first_run infer] batch 8 x 60s fp16: {sec:.3f} s/batch, "
          f"{rate:.0f} s-audio/s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB; "
          f"L4 (/6) worst case 1200 x 60 s = {l4_minutes:.1f} min")
