"""BEATs is the first frontend with real pretrained weights, so it is the first
place where "the encoder is a function of its own row" stops being free.

Every invariant here is paired in the comments with the mutation that was
*observed* to break it, per docs/pipelines/05. The mutation for the padding
invariant is not hypothetical: encoding the padded batch in one pass -- the
obvious faster implementation -- moves a 4 s file's features by 0.398.

Skipped when the checkpoint is absent, so the suite stays runnable on a machine
that has never downloaded 361 MB. Point DACON_BEATS_WEIGHTS at the directory
holding BEATs_iter3_plus_AS2M.pt to run them.
"""

import dataclasses
import os

import pytest
import torch

from models.config import AudioConfig, load_model_config
from models.frontends import LoRALinear, build_frontend

WEIGHTS = os.environ.get("DACON_BEATS_WEIGHTS", "/data/project/private/dacon-weights/beats")
_HAVE = os.path.exists(os.path.join(WEIGHTS, "BEATs_iter3_plus_AS2M.pt"))
pytestmark = pytest.mark.skipif(not _HAVE, reason=f"no BEATs checkpoint under {WEIGHTS}")

SR = AudioConfig().sample_rate


def _cfg(**kw):
    cfg = load_model_config("configs/a_shared_trunk.yaml")
    fe = dataclasses.replace(cfg.frontends["audio"], **{"weights": WEIGHTS, **kw})
    return fe, cfg.audio


@pytest.fixture(scope="module")
def beats():
    fe, audio = _cfg()
    return build_frontend(fe, audio).eval()


# --------------------------------------------------------------------------- #
# rule 2.4 -- the reason _encode runs one row at a time

def test_a_file_scores_the_same_alone_and_inside_a_longer_louder_batch(beats):
    """🔴 MUTATION: replace `_encode`'s per-row loop with a single padded pass
    and this goes from 0.0 to 0.398. BEATs flattens the patch grid into one
    `T' * F'` token sequence, so a row's valid tokens are T'-strided runs rather
    than a prefix -- a padding mask over that sequence is easy to get wrong and
    easy to omit entirely."""
    torch.manual_seed(0)
    target = torch.randn(SR * 4) * 0.1

    solo, mask = beats(target[None, :], torch.tensor([SR * 4]))
    padded = torch.zeros(3, SR * 9)
    padded[0, : SR * 4] = target
    padded[1] = torch.randn(SR * 9)
    padded[2] = torch.randn(SR * 9) * 5.0            # much louder neighbour
    batched, _ = beats(padded, torch.tensor([SR * 4, SR * 9, SR * 9]))

    n = int(mask.sum())

    # 🔴 Split deliberately. The ENCODER is bitwise invariant -- that is the
    # property `_encode`'s per-row loop buys, and it is the one a mutation
    # destroys (0.0 -> 0.398). What survives is 3e-8 introduced downstream by
    # `FreqPool` running on a (B, F, 24, D) tensor here and (B, F, 56, D)
    # there: float32 kernel selection, the same residual PROGRESS.md already
    # records for `align_time`. Asserting only the pooled tensor would hide
    # which of the two we actually guarantee.
    grid_solo = beats._encode(target[None, :], torch.tensor([SR * 4]))
    grid_batch = beats._encode(padded, torch.tensor([SR * 4, SR * 9, SR * 9]))
    assert torch.equal(grid_solo[0, :, :n], grid_batch[0, :, :n]), "encoder must be exact"
    assert (solo[0, :n] - batched[0, :n]).abs().max() < 1e-7


def test_what_fills_the_padding_cannot_reach_the_features(beats):
    """The companion to the above: same batch shape, different garbage past
    `lengths`. Zeros are not a neutral filler for BEATs -- after the
    checkpoint's fbank normalisation a zero row is -1.18, i.e. loud silence."""
    torch.manual_seed(1)
    target = torch.randn(SR * 4) * 0.1
    lengths = torch.tensor([SR * 4, SR * 9])

    a = torch.zeros(2, SR * 9); a[0, : SR * 4] = target; a[1] = torch.randn(SR * 9)
    b = a.clone(); b[0, SR * 4:] = 3.7

    fa, mask = beats(a, lengths)
    fb, _ = beats(b, lengths)
    n = int(mask[0].sum())
    assert torch.equal(fa[0, :n], fb[0, :n])


# --------------------------------------------------------------------------- #
# framing

def test_frame_count_is_the_encoder_s_own_arithmetic_not_the_nominal_hop(beats):
    """🔴 MUTATION: drop `BEATsFrontend._frames_for` so the base class's nominal
    ceil(samples/hop) is used, and a 4 s file claims 25 frames where BEATs emits
    24 -- marking a frame valid that came from the padding."""
    for seconds in (4, 7, 60):
        feats, mask = beats(torch.randn(1, SR * seconds), torch.tensor([SR * seconds]))
        mel = 1 + (SR * seconds - 400) // 160
        assert feats.shape[1] == mel // 16
        assert int(mask.sum()) == feats.shape[1]


def test_mask_marks_exactly_each_row_s_own_frames(beats):
    lengths = torch.tensor([SR * 4, SR * 3, SR * 2])
    wav = torch.zeros(3, SR * 4)
    for i, n in enumerate(lengths):
        wav[i, :n] = torch.randn(int(n)) * 0.1
    _, mask = beats(wav, lengths)
    assert mask.sum(1).tolist() == [beats._frames_for(int(n)) for n in lengths]


@pytest.mark.parametrize("n", [1, 100, 399, 400, 1000, 2799])
def test_audio_shorter_than_one_patch_still_yields_a_frame(beats, n):
    """🔴 `n=1000` alone did not reach the interesting path.

    1000 samples still produce 4 mel frames; below 400 -- one 25 ms analysis
    window -- kaldi's fbank ASSERTS ("choose a window size 400 that is [2, 399]")
    rather than returning an empty tensor, so `_frames_for`'s promise of at least
    one frame was false and `_encode` raised. MUTATION: remove the zero-pad in
    `_fbank` and every n < 400 case here fails."""
    feats, mask = beats(torch.randn(1, n) * 0.1, torch.tensor([n]))
    assert feats.shape[1] == 1 and int(mask.sum()) == 1


# --------------------------------------------------------------------------- #
# the patch grid

def test_the_patch_grid_is_time_major(beats):
    """🔴 MUTATION: read the flattened tokens as `view(B, F', T', D)` instead of
    `view(B, T', F', D)` and frequency is transposed against time. It raises at
    most lengths because F'=8 and T' is ~24, but at T'=8 the shapes agree and it
    would train silently wrong -- so pin the ordering, not the shape."""
    C, T, F_ = 3, 4, 2
    x = torch.zeros(1, C, T, F_)
    for t in range(T):
        for f in range(F_):
            x[0, :, t, f] = t * 10 + f
    flat = x.reshape(1, C, -1).transpose(1, 2)       # upstream's two lines
    assert torch.equal(flat.view(1, T, F_, C)[0, :, :, 0], x[0, 0])


def test_freq_pool_is_required_because_beats_emits_a_grid():
    from models.config import FreqPoolConfig
    fe, audio = _cfg(freq_pool=FreqPoolConfig(kind="none"))
    with pytest.raises(ValueError, match="patch grid"):
        build_frontend(fe, audio)


# --------------------------------------------------------------------------- #
# configuration that must not be silently accepted

def test_a_wrong_fps_is_refused_rather_than_misaligning_evidence():
    """🔴 `fps` is what aligns two frontends onto a common time base. The shipped
    config said 50.0 -- copied from the wav2vec2 family -- where BEATs emits
    6.25. Nothing downstream would have failed; the file branch would simply
    have concatenated misaligned evidence."""
    fe, audio = _cfg(fps=50.0)
    with pytest.raises(ValueError, match="6.25"):
        build_frontend(fe, audio)


def test_n_freq_must_match_the_checkpoint_s_patch_size():
    fe, audio = _cfg(n_freq=4)
    with pytest.raises(ValueError, match="patch grid is 8 bands"):
        build_frontend(fe, audio)


def test_weightless_beats_is_refused_because_it_is_a_slower_stub():
    fe, audio = _cfg(weights=None)
    with pytest.raises(ValueError, match="StubFrontend"):
        build_frontend(fe, audio)


def test_an_adapter_that_matches_nothing_is_an_error_not_a_no_op():
    from models.config import AdapterConfig
    fe, audio = _cfg(adapter=AdapterConfig(kind="lora", targets=("not_a_layer",)))
    with pytest.raises(ValueError, match="matched no nn.Linear"):
        build_frontend(fe, audio)


# --------------------------------------------------------------------------- #
# truncation and LoRA

def test_truncation_deletes_layers_rather_than_skipping_them(beats):
    assert beats.n_layers == 9
    assert len(beats.encoder.layers) == 9, "unused blocks must not cost parameters"


def test_lora_wraps_both_targets_in_every_kept_layer(beats):
    assert beats.n_lora == 9 * 2                     # q_proj and v_proj per layer
    assert all(isinstance(l.self_attn.q_proj, LoRALinear) for l in beats.encoder.layers)


def test_lora_is_the_identity_at_initialisation(beats):
    """🔴 MUTATION: initialise `lora_b` with anything but zeros and this fails.
    Zero-init is what makes "load a pretrained encoder, then adapt it" true --
    an untrained adapter must not perturb the checkpoint at all."""
    fe_none, audio = _cfg(adapter=dataclasses.replace(
        load_model_config("configs/a_shared_trunk.yaml").frontends["audio"].adapter,
        kind="none"))
    plain = build_frontend(fe_none, audio).eval()
    plain.freq_pool.load_state_dict(beats.freq_pool.state_dict())

    torch.manual_seed(2)
    wav = torch.randn(1, SR * 4) * 0.1
    lengths = torch.tensor([SR * 4])
    assert torch.equal(beats(wav, lengths)[0], plain(wav, lengths)[0])


def test_only_our_own_parameters_train_under_freeze(beats):
    trainable = {n for n, p in beats.named_parameters() if p.requires_grad}
    assert trainable, "a frozen frontend with nothing trainable cannot be adapted"
    assert all("lora_" in n or n.startswith("freq_pool.") for n in trainable), trainable
    frozen = [n for n, p in beats.named_parameters() if not p.requires_grad]
    assert any(n.startswith("encoder.layers.") for n in frozen)


# --------------------------------------------------------------------------- #
# determinism

def test_no_hidden_generator_in_the_forward_pass(beats):
    """🔴 Upstream draws `np.random.random()` per layer for layerdrop. That is a
    generator outside our seeding, and training/AGENTS.md records the resume
    guarantee as bitwise with a complete state list. `_BEATsEncoderConfig`
    forces `encoder_layerdrop = 0`; MUTATION: restore the checkpoint's 0.05 and
    this fails in train() mode."""
    beats.train()
    try:
        wav, lengths = torch.randn(2, SR * 3) * 0.1, torch.tensor([SR * 3, SR * 3])
        assert torch.equal(beats(wav, lengths)[0], beats(wav, lengths)[0])
    finally:
        beats.eval()
    assert beats.encoder.layerdrop == 0.0


# --------------------------------------------------------------------------- #
# the c_first_run audio trunk: all 12 layers, LoRA on six projections

@pytest.fixture(scope="module")
def beats12():
    cfg = load_model_config("configs/c_first_run.yaml")
    fe = dataclasses.replace(cfg.frontends["audio"], weights=WEIGHTS)
    return build_frontend(fe, cfg.audio).eval()


def test_layers_null_keeps_all_twelve_and_lora_wraps_six_per_layer(beats12):
    assert beats12.n_layers == 12 and len(beats12.encoder.layers) == 12
    assert beats12.n_lora == 12 * 6
    for layer in beats12.encoder.layers:
        for mod in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
                    layer.self_attn.out_proj, layer.fc1, layer.fc2):
            # The shipped checkpoint is `activation_fn: gelu`, so fc1 is a plain
            # nn.Linear (a `glu` one would be GLU_Linear, adapted via `.linear`).
            assert isinstance(mod, LoRALinear) and isinstance(mod.base, torch.nn.Linear)


def test_every_lora_target_reaches_the_output(beats12):
    """🔴 LoRA wraps a *module*; it is inert if the forward reads `.weight`
    directly (e.g. through `F.multi_head_attention_forward`). The vendored
    `MultiheadAttention.forward` calls `self.q_proj(...)` etc. -- this proves it
    per target: MUTATION -- have the attention compute `F.linear(x,
    self.q_proj.weight, ...)` and the q_proj case stops moving the output."""
    torch.manual_seed(7)
    wav = torch.randn(1, SR * 3 + 111) * 0.1
    lengths = torch.tensor([SR * 3 + 111])
    layer = beats12.encoder.layers[4]
    mods = {"q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj, "out_proj": layer.self_attn.out_proj,
            "fc1": layer.fc1, "fc2": layer.fc2}
    with torch.no_grad():
        base = beats12(wav, lengths)[0]
        for name, mod in mods.items():
            mod.lora_b.normal_(0, 0.05)
            try:
                moved = (beats12(wav, lengths)[0] - base).abs().max().item()
            finally:
                mod.lora_b.zero_()
            assert moved > 1e-4, f"{name}: a trained lora_b did not reach the output"
        assert torch.equal(beats12(wav, lengths)[0], base)


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs a GPU"))])
def test_fbank_is_immune_to_fp16_autocast(beats, device):
    """🔴 fbank's mel matmul autocast to fp16 overflows at int16 scale and every
    output went NaN. The fbank must be bitwise the fp32 one under autocast."""
    wav = torch.randn(SR * 2, device=device) * 0.1
    beats.to(device)
    try:
        ref = beats._fbank(wav)
        with torch.autocast(device, dtype=torch.float16):
            got = beats._fbank(wav)
    finally:
        beats.to("cpu")
    assert torch.isfinite(got).all()
    assert torch.equal(got, ref)


def test_six_target_adapters_train_and_nothing_else_in_the_encoder_does(beats12):
    trainable = {n for n, p in beats12.named_parameters() if p.requires_grad}
    assert all("lora_" in n or n.startswith("freq_pool.") for n in trainable), trainable
    assert sum("lora_b" in n for n in trainable) == 12 * 6
