"""docs/training/07 §3: layer checkpointing changes memory, not numbers.

The first GPU smoke of the two-trunk model ran an H200 out of memory at batch
8: both trunks kept every layer's attention maps for the LoRA gradients.
`enable_layer_checkpointing` recomputes each layer in backward instead. These
tests hold it to "the same run": equal outputs and equal adapter gradients in
training mode (dropout on, its RNG restored by the recompute), parameter names
unchanged, and no effect in eval mode.
"""

import dataclasses
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.config import load_model_config
from models.frontends import build_frontend, enable_layer_checkpointing

BEATS = os.environ.get("DACON_BEATS_WEIGHTS", "/data/project/private/dacon-weights/beats")
XLSR = os.environ.get("DACON_XLSR_WEIGHTS", "/data/project/private/dacon-weights/xlsr-300m")


def test_generic_layers_give_equal_outputs_and_grads():
    torch.manual_seed(0)
    layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.3), nn.GELU())
                            for _ in range(3)]).train()
    x = torch.randn(4, 8)

    def run():
        torch.manual_seed(1)
        h = x
        for layer in layers:
            h = layer(h)
        loss = h.square().sum()
        grads = torch.autograd.grad(loss, list(layers.parameters()))
        return h.detach(), grads

    ref_out, ref_g = run()
    names = [n for n, _ in layers.named_parameters()]
    assert enable_layer_checkpointing(layers) == 3
    assert enable_layer_checkpointing(layers) == 0, "idempotent"
    assert [n for n, _ in layers.named_parameters()] == names
    out, g = run()
    assert torch.equal(out, ref_out)
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(g, ref_g))


def _frontend(name, weights):
    cfg = load_model_config("configs/c_first_run.yaml")
    fe = dataclasses.replace(cfg.frontends[name], weights=weights)
    return build_frontend(fe, cfg.audio)


@pytest.mark.parametrize("name,weights", [("audio", BEATS), ("speech", XLSR)])
def test_real_trunks_train_identically_with_checkpointing(name, weights):
    if not os.path.isdir(weights):
        pytest.skip(f"no weights under {weights}")
    torch.manual_seed(0)
    fe = _frontend(name, weights).train()
    wav = torch.randn(2, 16000 * 3)
    lengths = torch.tensor([16000 * 3, 16000 * 2])

    def run():
        torch.manual_seed(7)
        np.random.seed(7)                      # BEATs' layerdrop draws from numpy
        fe.zero_grad(set_to_none=True)
        feats, mask = fe(wav, lengths)
        (feats * mask.unsqueeze(-1)).square().mean().backward()
        grads = {n: p.grad.clone() for n, p in fe.named_parameters()
                 if p.requires_grad and p.grad is not None}
        return feats.detach(), grads

    ref_feats, ref_grads = run()
    assert ref_grads, "no trainable parameter received a gradient"
    assert fe.enable_grad_checkpointing() > 0
    feats, grads = run()
    assert torch.allclose(feats, ref_feats, atol=1e-5)
    assert grads.keys() == ref_grads.keys()
    for n in grads:
        assert torch.allclose(grads[n], ref_grads[n], atol=1e-5, rtol=1e-4), n
    fe.eval()
    with torch.no_grad():
        a, _ = fe(wav, lengths)
    assert torch.isfinite(a).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_bandpass_caps_the_cufft_plan_cache():
    """The leak the first smoke hit: one cuFFT plan per new length, ~70 MB of
    device memory each, outside PyTorch's allocator."""
    from models.audio import CUFFT_PLAN_CACHE_MAX, bandpass
    from models.config import AudioConfig
    cfg = AudioConfig(band_hz=(0.0, 7200.0))
    torch.backends.cuda.cufft_plan_cache.clear()
    for n in range(16000 * 5, 16000 * 5 + 40 * 17, 17):
        w = torch.randn(2, n, device="cuda")
        bandpass(w, cfg, torch.full((2,), n, device="cuda"))
    assert torch.backends.cuda.cufft_plan_cache.size <= CUFFT_PLAN_CACHE_MAX
