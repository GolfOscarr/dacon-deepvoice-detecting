"""The SED head is where the label semantics become arithmetic, so its
properties are worth asserting rather than assuming."""

import pytest
import torch

from models.config import FreqPoolConfig, SEDHeadConfig
from models.heads import FreqPool, SEDHead, frame_max


def _head(in_dim=32, **kw):
    torch.manual_seed(0)
    return SEDHead(in_dim, SEDHeadConfig(**kw)).eval()


# --------------------------------------------------------------------------- #
# FreqPool

def test_freq_pool_shapes():
    x = torch.randn(2, 8, 50, 32).abs()          # (B, F, T, D)
    for kind in ("gem", "mean", "max"):
        out = FreqPool(FreqPoolConfig(kind=kind))(x)
        assert out.shape == (2, 50, 32)


def test_freq_pool_rejects_missing_frequency_axis():
    """wav2vec2-family frontends emit (B, T, D) and must configure kind=none."""
    with pytest.raises(ValueError, match="B, F, T, D"):
        FreqPool(FreqPoolConfig(kind="gem"))(torch.randn(2, 50, 32))


def test_gem_interpolates_between_mean_and_max():
    """p=1 is exactly the mean, and larger p moves monotonically toward the max.

    ⚠️ GeM approaches the max *slowly*: over F bins it is bounded by
    `max * (1/F)^(1/p)`, so at F=16 even p=60 still sits ~4.5% below the max.
    Asserting near-equality to `amax` at any practical p is therefore wrong --
    the meaningful property is the ordering, not the limit.
    """
    x = torch.rand(4, 16, 20, 8) + 0.1
    mean, mx = x.mean(1), x.amax(1)

    assert torch.allclose(FreqPool(FreqPoolConfig(kind="gem", p_init=1.0))(x), mean, atol=1e-4)

    prev = mean
    for p_init in (2.0, 4.0, 8.0, 32.0):
        cur = FreqPool(FreqPoolConfig(kind="gem", p_init=p_init))(x)
        assert (cur >= prev - 1e-5).all(), f"GeM should be non-decreasing in p (p={p_init})"
        assert (cur <= mx + 1e-5).all(), f"GeM should never exceed the max (p={p_init})"
        prev = cur
    assert (prev > mean).all(), "large p should be strictly above the mean"


def test_gem_p_is_learnable_when_configured():
    assert FreqPool(FreqPoolConfig(kind="gem", learnable=True)).p.requires_grad
    assert not FreqPool(FreqPoolConfig(kind="gem", learnable=False)).p.requires_grad


def test_gem_survives_negative_features():
    """SSL features are not sign-constrained; a negative base with fractional p
    is NaN, so the clamp is load-bearing."""
    out = FreqPool(FreqPoolConfig(kind="gem", p_init=2.5))(torch.randn(2, 4, 6, 3))
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# SEDHead

def test_sed_head_shapes():
    out = _head()(torch.randn(3, 40, 32))
    assert out.clip_logits.shape == (3,)
    assert out.frame_logits.shape == (3, 40)
    assert out.attention.shape == (3, 40)


def test_attention_is_a_distribution_over_time():
    out = _head()(torch.randn(3, 40, 32))
    assert torch.allclose(out.attention.sum(-1), torch.ones(3), atol=1e-5)


def test_head_emits_logits_not_probabilities():
    """No sigmoid here -- blending happens in logit space (04 §3)."""
    out = _head()(torch.randn(8, 30, 32) * 5)
    assert (out.frame_logits < 0).any(), "logits should be signed"


def test_padding_gets_zero_attention():
    x = torch.randn(2, 20, 32)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[1, 12:] = False
    out = _head()(x, mask)
    assert out.attention[1, 12:].abs().max() == 0.0
    assert torch.allclose(out.attention.sum(-1), torch.ones(2), atol=1e-5)


def test_clip_logit_ignores_padded_content():
    """🔴 rule 2.4: a file's score must not depend on what is batched with it.

    If padding could win attention mass, the same file would score differently
    depending on the longest file in its batch.
    """
    head = _head()
    x = torch.randn(1, 15, 32)
    alone = head(x, torch.ones(1, 15, dtype=torch.bool))

    padded = torch.cat([x, torch.randn(1, 25, 32) * 100], dim=1)   # loud padding
    mask = torch.zeros(1, 40, dtype=torch.bool)
    mask[:, :15] = True
    with_pad = head(padded, mask)

    assert torch.allclose(alone.clip_logits, with_pad.clip_logits, atol=1e-5)


def test_mask_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="mask"):
        _head()(torch.randn(2, 20, 32), torch.ones(2, 19, dtype=torch.bool))


def test_frame_max_is_padding_safe():
    logits = torch.tensor([[1.0, 2.0, 99.0]])
    mask = torch.tensor([[True, True, False]])
    assert frame_max(logits, mask).item() == pytest.approx(2.0)
    assert frame_max(logits).item() == pytest.approx(99.0)


def test_head_is_deterministic_in_eval_mode():
    """script.py asserts a canned-input fingerprint; that needs determinism."""
    head, x = _head(), torch.randn(2, 25, 32)
    assert torch.equal(head(x).clip_logits, head(x).clip_logits)
