"""The shipped chain: per-file over the valid prefix (I14), DC removed, the
band applied with no high-pass, every config field a live knob, and the same
function on a rendered sample and on a batch.

Critical: I14 is mutation-tested -- a chain that reads the padded row must be
caught -- and every knob is shown to move the output.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from processing.config import (ProcessingConfig, dump_processing_config,
                               load_processing_config, processing_config_from_dict)
from processing.render import ManifestIndex, RenderConfig, render
from processing.ship import PreprocessStep, ShipConfig, ship, ship_sample
from training.spec import ComponentDraw, SampleSpec
from training.synthetic import synthetic_manifest, write_synthetic_corpus

REPO = Path(__file__).resolve().parents[1]
SR = 16_000


def _noise(shape, seed=0, scale=1.0):
    return torch.from_numpy(
        (scale * np.random.default_rng(seed).standard_normal(shape)).astype(np.float32))


def _tone(hz, n=SR, amp=0.5):
    t = torch.arange(n, dtype=torch.float64) / SR
    return (amp * torch.sin(2 * np.pi * hz * t)).float()


def _band_energy(x: torch.Tensor, lo: float, hi: float) -> float:
    f = torch.fft.rfftfreq(x.shape[-1], 1.0 / SR)
    spec = torch.fft.rfft(x.double()).abs() ** 2
    return float(spec[(f >= lo) & (f <= hi)].sum() / spec.sum())


# --------------------------------------------------------------------------- #
# I14 -- rule 2.4 over the whole chain


def _batch_invariance_gap(cfg: ShipConfig, n: int = 977, total: int = 3_000) -> float:
    """max |solo - in-batch| over the row's valid prefix. Must be exactly 0.
    The batch is hostile: loud garbage in the padding, louder longer rows."""
    row = _noise(n, seed=0) + 0.2
    solo = ship(row[None, :], cfg, torch.tensor([n]))[0, :n]
    batch = _noise((4, total), seed=1, scale=30.0)
    batch[0, :n] = row
    lengths = torch.tensor([n, total, total - 1, total // 2])
    inside = ship(batch, cfg, lengths)[0, :n]
    return float((solo - inside).abs().max())


def test_the_shipped_chain_is_batch_invariant():
    """I14 on the chain that ships, not on its steps one by one."""
    assert _batch_invariance_gap(ShipConfig()) == 0.0


def test_the_batch_invariance_check_can_fail():
    """Mutation: a step over the padded row must be caught by the check above."""
    from training.registries import Registry
    broken = Registry("preprocess", ("wav", "sample_rate", "lengths"), frozenset())

    @broken.register("dc_over_the_padded_row")
    def _bad(wav, sample_rate, lengths):
        return wav - wav.mean(dim=-1, keepdim=True)

    def bad_ship(wav, cfg, lengths=None):
        mono = wav if wav.dim() == 2 else wav.mean(dim=1)
        return broken.build("dc_over_the_padded_row")(mono, SR, lengths)

    row = _noise(977, seed=0) + 0.2
    solo = bad_ship(row[None, :], None, torch.tensor([977]))[0]
    batch = _noise((4, 3000), seed=1, scale=30.0)
    batch[0, :977] = row
    inside = bad_ship(batch, None, torch.tensor([977, 3000, 2999, 1500]))[0, :977]
    assert float((solo - inside).abs().max()) > 0.0


def test_padding_beyond_the_valid_prefix_carries_no_signal():
    """The band stage returns zeros in the padding; nothing there is a file's."""
    batch = _noise((2, 3000), seed=2)
    out = ship(batch, ShipConfig(), torch.tensor([1000, 3000]))
    assert torch.equal(out[0, 1000:], torch.zeros(2000))


# --------------------------------------------------------------------------- #
# SHIP-4 / SHIP-5 -- what the chain does


def test_dc_is_removed_over_the_valid_prefix():
    x = _noise((1, SR), seed=3) + 0.37
    out = ship(x, ShipConfig(band_hz=None))
    assert abs(float(out[0].mean())) < 1e-6


def test_without_the_step_dc_stays():
    x = _noise((1, SR), seed=3) + 0.37
    out = ship(x, ShipConfig(preprocess=(), band_hz=None))
    assert abs(float(out[0].mean()) - 0.37) < 0.02


def test_the_band_keeps_the_low_end_and_removes_above_the_edge():
    """D-11: no high-pass -- a 20 Hz tone survives. D-12: nothing above 7 200 Hz."""
    x = (_tone(20.0) + _tone(1000.0) + _tone(7600.0))[None]
    out = ship(x, ShipConfig())[0]
    assert _band_energy(out, 7300.0, 8000.0) < 1e-6
    assert _band_energy(out, 15.0, 25.0) > 0.3
    assert _band_energy(out, 990.0, 1010.0) > 0.3


def test_band_none_is_full_band():
    x = (_tone(7600.0))[None]
    out = ship(x, ShipConfig(band_hz=None))[0]
    assert _band_energy(out, 7300.0, 8000.0) > 0.99


def test_the_chain_declares_zero_group_delay():
    """A shifted chain would move the audio out from under the frame targets."""
    x = _noise((1, SR), seed=4)
    out = ship(x, ShipConfig(band_hz=None))[0]
    lag = int(torch.argmax(torch.from_numpy(np.correlate(out.numpy(), x[0].numpy(), "full")))) \
        - (SR - 1)
    assert lag == 0


def test_downmix_averages_the_channels():
    x = torch.stack([_tone(440.0, amp=0.4), _tone(440.0, amp=0.8)])[None]
    out = ship(x, ShipConfig(band_hz=None))[0]
    assert torch.allclose(out, _tone(440.0, amp=0.6), atol=1e-5)


# --------------------------------------------------------------------------- #
# every ShipConfig field is a knob

#: `field -> (config a, config b)`; the band must be legal at both sample rates.
_KNOBS = {
    "sample_rate": ({"sample_rate": 16_000, "band_hz": (0.0, 5000.0)},
                    {"sample_rate": 12_000, "band_hz": (0.0, 5000.0)}),
    "channels": ({"channels": "downmix"}, {"channels": "left"}),
    "preprocess": ({"preprocess": (PreprocessStep("dc_offset"),)}, {"preprocess": ()}),
    "band_hz": ({"band_hz": (0.0, 7200.0)}, {"band_hz": (0.0, 3400.0)}),
}


def test_the_knob_table_names_every_ship_config_field():
    assert set(_KNOBS) == {f.name for f in dataclasses.fields(ShipConfig)}


@pytest.mark.parametrize("field", sorted(_KNOBS))
def test_every_ship_config_field_changes_the_output(field):
    a, b = _KNOBS[field]
    x = torch.stack([_tone(440.0, amp=0.4) + 0.3, _tone(5000.0, amp=0.8)])[None]
    assert not torch.equal(ship(x, ShipConfig(**a)), ship(x, ShipConfig(**b))), field


def test_the_v1_ship_defaults_are_pinned():
    cfg = ShipConfig()
    assert cfg.sample_rate == 16_000 and cfg.channels == "downmix"
    assert cfg.preprocess == (PreprocessStep("dc_offset"),)
    assert cfg.band_hz == (0.0, 7200.0)


@pytest.mark.parametrize("bad", [
    {"channels": "stereo"},
    {"band_hz": (0.0, 9000.0)},
    {"band_hz": (300.0, 300.0)},
    {"band_hz": (-1.0, 7200.0)},
    {"preprocess": (PreprocessStep("dc_offset"), PreprocessStep("dc_offset"))},
    {"sample_rate": 0},
])
def test_malformed_ship_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        ShipConfig(**bad)


def test_an_unregistered_or_misparameterised_step_is_refused():
    with pytest.raises(ValueError, match="not registered"):
        PreprocessStep("loudness_normalise")
    with pytest.raises(ValueError, match="no parameter"):
        PreprocessStep("dc_offset", {"coeff": 0.97})
    PreprocessStep("pre_emphasis", {"coeff": 0.97})


# --------------------------------------------------------------------------- #
# YAML and the rendered sample


def test_the_v1_config_file_ships_the_defaults():
    cfg = load_processing_config(REPO / "configs" / "processing_v1.yaml")
    assert cfg.ship == ProcessingConfig().ship


def test_a_non_default_ship_section_round_trips():
    raw = {"ship": {"sample_rate": 16_000, "channels": "left",
                    "preprocess": ["dc_offset", {"name": "pre_emphasis", "coeff": 0.9}],
                    "band_hz": [0.0, 4000.0]}}
    cfg = processing_config_from_dict(raw)
    assert cfg.ship.preprocess == (PreprocessStep("dc_offset"),
                                   PreprocessStep("pre_emphasis", {"coeff": 0.9}))
    assert cfg.ship.band_hz == (0.0, 4000.0)
    assert processing_config_from_dict(dump_processing_config(cfg)) == cfg
    assert dump_processing_config(cfg)["ship"]["preprocess"] == raw["ship"]["preprocess"]


def test_a_rendered_sample_ships_to_one_mono_row(tmp_path):
    manifest = synthetic_manifest(n_per_pool=2, n_whole_file=2, seed=0,
                                  duration_range=(6.0, 7.0))
    write_synthetic_corpus(manifest, tmp_path, seed=0)
    voice = manifest[manifest.pool == "A"].file_id.iloc[0]
    spec = SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x", duration_s=5.0,
                      cell=1, render_mode="composed", structure="overlap",
                      components=(ComponentDraw(voice, "voice", 0.5, 5.0, 0.0, 0.0),),
                      normalize={"channels": "stereo"})
    r = render(spec, ManifestIndex.from_frame(manifest), RenderConfig(root=tmp_path))
    assert r.wav.shape[0] == 2
    out = ship_sample(r, ShipConfig())
    assert out.shape == (5 * SR,)
    # the same function, the same answer, inside a batch beside another file
    batch = torch.zeros((2, 2, 6 * SR))
    batch[0, :, :5 * SR] = r.wav
    batch[1] = _noise((2, 6 * SR), seed=9, scale=10.0)
    inside = ship(batch, ShipConfig(), torch.tensor([5 * SR, 6 * SR]))[0, :5 * SR]
    assert torch.equal(out, inside)
