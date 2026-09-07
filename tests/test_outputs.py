"""The output stage exists because of one measured result: saturating the
operating point took EER 0.0950 -> 0.3017. These tests are that result,
turned into assertions."""

import pytest
import torch

from models.config import AggregationConfig, OutputConfig
from models.outputs import aggregate_windows, blend_logits, to_probability


# --------------------------------------------------------------------------- #
# saturation

def test_softsign_survives_logits_that_saturate_a_sigmoid():
    """🔴 The whole reason this module exists."""
    z = torch.tensor([40.0, 60.0, 100.0, 400.0], dtype=torch.float64)
    sig = to_probability(z, OutputConfig(squash="sigmoid", clamp_eps=0.0))
    soft = to_probability(z, OutputConfig(squash="softsign", clamp_eps=0.0))
    assert len(sig.unique()) == 1, "sigmoid should collapse these to one value"
    assert len(soft.unique()) == 4, "softsign must keep them distinct"


def test_no_ties_across_a_wide_logit_range():
    z = torch.linspace(-500, 500, 4000, dtype=torch.float64)
    p = to_probability(z, OutputConfig(squash="softsign"))
    assert len(p.unique()) == len(p), "ties cost ranking, and rank fixes are forbidden"


def test_output_is_float64_and_in_range():
    p = to_probability(torch.randn(100) * 20, OutputConfig())
    assert p.dtype == torch.float64
    assert (p > 0).all() and (p < 1).all()


def test_softsign_is_order_preserving_at_scale_and_sigmoid_is_not():
    """EER reads only the ordering, so order preservation is the whole contract.

    Measured on 500 logits at scale 30: softsign keeps 500/500 distinct values
    and the exact ordering; sigmoid keeps 314/500 and loses the ordering to
    ties. That gap is why softsign is the default.
    """
    z = torch.randn(500, dtype=torch.float64) * 30
    soft = to_probability(z, OutputConfig(squash="softsign"))
    assert len(soft.unique()) == 500
    assert torch.equal(z.argsort(), soft.argsort())

    sig = to_probability(z, OutputConfig(squash="sigmoid"))
    assert len(sig.unique()) < 400, "sigmoid is expected to tie at this scale"


def test_logit_scale_widens_the_unsaturated_region():
    z = torch.tensor([40.0, 80.0], dtype=torch.float64)
    tight = to_probability(z, OutputConfig(squash="sigmoid", logit_scale=1.0, clamp_eps=0.0))
    loose = to_probability(z, OutputConfig(squash="sigmoid", logit_scale=0.1, clamp_eps=0.0))
    assert tight[0] == tight[1]
    assert loose[0] != loose[1]


def test_unknown_squash_raises():
    with pytest.raises(ValueError, match="squash"):
        to_probability(torch.zeros(3), OutputConfig(squash="relu"))


# --------------------------------------------------------------------------- #
# blending

def test_blend_is_in_logit_space_and_weighted():
    clip, fmax = torch.tensor([2.0]), torch.tensor([6.0])
    assert blend_logits(clip, fmax, 0.5).item() == pytest.approx(4.0)
    assert blend_logits(clip, fmax, 1.0).item() == pytest.approx(2.0)
    assert blend_logits(clip, fmax, 0.0).item() == pytest.approx(6.0)


def test_blend_rejects_a_weight_outside_the_unit_interval():
    with pytest.raises(ValueError, match="clip_weight"):
        blend_logits(torch.zeros(1), torch.zeros(1), 1.5)


def test_logit_blend_differs_from_averaging_two_sigmoids():
    """The construction we deliberately do not ship."""
    clip, fmax = torch.tensor([50.0], dtype=torch.float64), torch.tensor([-50.0], dtype=torch.float64)
    ours = to_probability(blend_logits(clip, fmax, 0.5), OutputConfig(squash="softsign"))
    theirs = 0.5 * torch.sigmoid(clip) + 0.5 * torch.sigmoid(fmax)
    assert ours.item() == pytest.approx(0.5)
    assert theirs.item() == pytest.approx(0.5)
    # ...but the two diverge once one side dominates, and the sigmoid version
    # has already lost the magnitude information by then.
    clip2 = torch.tensor([80.0], dtype=torch.float64)
    assert torch.sigmoid(clip2) == torch.sigmoid(clip)          # saturated, indistinguishable
    assert to_probability(clip2, OutputConfig(squash="softsign")) != \
           to_probability(clip, OutputConfig(squash="softsign"))


# --------------------------------------------------------------------------- #
# cross-window aggregation and the duration trap

def _rand_windows(n_windows, seed=0):
    torch.manual_seed(seed)
    return torch.randn(2000, n_windows)


def _duration_spread(cfg, counts=(1, 2, 4, 8, 12), n=20000, seed=0):
    """Mean file score as a function of window count, with the per-window
    distribution held identical. Any movement is pure duration bias."""
    torch.manual_seed(seed)
    means = [aggregate_windows(torch.randn(n, c), cfg).mean().item() for c in counts]
    return max(means) - min(means)


def test_order_statistics_all_carry_a_large_duration_bias():
    """🔴 Measured, and it corrects a guess in docs/architecture/04 §6.1.

    Every order-statistic aggregator inherits the same problem: the expected
    k-th largest of W samples grows with W, so long files score higher than
    short ones on identical content. top-k mean is only ~28% better than max,
    not the "mild" effect the doc first assumed.
    """
    max_spread = _duration_spread(AggregationConfig(kind="max"))
    topk_spread = _duration_spread(AggregationConfig(kind="topk_mean", k=3))
    quant_spread = _duration_spread(AggregationConfig(kind="quantile", quantile=0.9))

    assert max_spread > 1.4
    assert topk_spread > 1.0, "top-k mean does NOT solve this"
    assert quant_spread > 0.9, "nor does a fixed quantile"
    assert topk_spread < max_spread, "it is a modest improvement, not a fix"


def test_only_mean_is_duration_neutral_and_it_dilutes():
    assert _duration_spread(AggregationConfig(kind="mean")) < 0.05


def test_confidence_gated_is_mean_in_disguise_on_symmetric_scores():
    """Its low duration bias is an artifact, not a property.

    The gate needs >40% of windows above logit 1.386 (p=0.8). On roughly
    symmetric scores that essentially never fires, so it falls through to the
    plain mean -- and inherits mean's neutrality *and* mean's dilution. Recorded
    so nobody reads its spread as evidence it solves the problem.
    """
    torch.manual_seed(0)
    x = torch.randn(20_000, 8)
    gated = aggregate_windows(x, AggregationConfig(kind="confidence_gated"))
    plain = aggregate_windows(x, AggregationConfig(kind="mean"))
    assert torch.isclose(gated, plain, atol=1e-6).float().mean() > 0.95


@pytest.mark.parametrize("kind", ["max", "mean", "topk_mean", "quantile", "confidence_gated"])
def test_every_aggregator_returns_one_score_per_file(kind):
    out = aggregate_windows(torch.randn(5, 8), AggregationConfig(kind=kind))
    assert out.shape == (5,)
    assert torch.isfinite(out).all()


def test_aggregation_respects_the_window_mask():
    logits = torch.tensor([[1.0, 2.0, 99.0]])
    mask = torch.tensor([[True, True, False]])
    assert aggregate_windows(logits, AggregationConfig(kind="max"), mask).item() == pytest.approx(2.0)


def test_topk_does_not_average_padding_for_short_files():
    """A one-window file must not be diluted by k-1 invalid slots."""
    logits = torch.tensor([[5.0, -99.0, -99.0]])
    mask = torch.tensor([[True, False, False]])
    got = aggregate_windows(logits, AggregationConfig(kind="topk_mean", k=3), mask)
    assert got.item() == pytest.approx(5.0)


def test_file_with_no_valid_window_is_an_error():
    with pytest.raises(ValueError, match="at least one valid window"):
        aggregate_windows(torch.randn(2, 4), AggregationConfig(),
                          torch.zeros(2, 4, dtype=torch.bool))


def test_unknown_aggregation_raises():
    with pytest.raises(ValueError, match="unknown aggregation"):
        aggregate_windows(torch.randn(2, 4), AggregationConfig(kind="median"))
