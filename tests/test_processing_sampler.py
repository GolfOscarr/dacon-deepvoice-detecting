"""The processing sampler: the take/offset/tile rule, DOSS on every row kind,
the component floor, the config, and every knob moving the stream.

Critical: each invariant is mutation-tested -- shown failing on the config
value that should break it -- before it is trusted (docs/processing/03 §6).
"""

import dataclasses
import math
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import yaml

from processing.config import (SECTIONS, ConfigError, DrawConfig, ProcessingConfig,
                               dump_processing_config, load_processing_config,
                               processing_config_from_dict)
from processing.sampler import Sampler
from training.audit import run_audit
from training.sampler import REFERENCE_MIX, CellMix
from training.synthetic import synthetic_manifest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest():
    return synthetic_manifest(n_per_pool=200, n_whole_file=200, seed=0)


@pytest.fixture(scope="module")
def sampler(manifest):
    return Sampler(manifest)


def _durations(manifest) -> pd.Series:
    return manifest.set_index("file_id").duration_s


def _with_pool_duration(manifest, pool: str, seconds: float) -> pd.DataFrame:
    """The manifest with every row of ``pool`` declared exactly ``seconds`` long
    -- pool D of the real corpus is 10.00 s files (02 §2)."""
    df = manifest.copy()
    df.loc[df.pool == pool, "duration_s"] = seconds
    return df


# --------------------------------------------------------------------------- #
# D-3: the take lies strictly inside the file


def test_every_take_lies_strictly_inside_its_file(sampler, manifest):
    """``offset >= margin`` and ``offset + take <= file - margin`` for every
    component draw of every role and row kind."""
    dur = _durations(manifest)
    m = sampler.cfg.edge_margin_s
    n = 0
    for spec in sampler.epoch_specs(1500):
        for c in spec.components:
            n += 1
            assert c.source_offset_s >= m - 1e-9, (spec.sample_id, c)
            assert c.source_offset_s + c.duration_s <= dur[c.file_id] - m + 1e-9, \
                (spec.sample_id, c, dur[c.file_id])
    assert n > 1500


def test_a_ten_second_row_never_starts_at_its_onset(manifest):
    """The §6 step-1 test. Pool D is 10 s files; under the training sampler
    87 % of its draws began at sample 0 (02 §2). Here: none within 10 ms."""
    df = _with_pool_duration(manifest, "D", 10.0)
    s = Sampler(df)
    pool = df.set_index("file_id").pool
    early = Counter()
    seen = Counter()
    for spec in s.epoch_specs(3000):
        for c in spec.components:
            p = pool[c.file_id]
            seen[p] += 1
            early[p] += c.source_offset_s < 0.010
    assert seen["D"] > 500
    # equal across pools, and < 1 % (docs/processing/03 DRAW-3 Verify)
    assert all(early[p] == 0 for p in seen), dict(early)


def test_edge_margin_zero_re_exposes_the_onset(manifest):
    """Mutation: ``edge_margin_s = 0`` must make the test above fail."""
    df = _with_pool_duration(manifest, "D", 10.0)
    s = Sampler(df, DrawConfig(edge_margin_s=0.0))
    pool = df.set_index("file_id").pool
    early = sum(c.source_offset_s < 0.010
                for spec in s.epoch_specs(3000) for c in spec.components
                if pool[c.file_id] == "D")
    assert early > 0, "with no margin an offset of ~0 is drawn; if not, the " \
                      "margin test is not testing the margin"


def test_a_file_exactly_one_take_long_is_still_cropped_inside(manifest):
    """CompSpoof's 4.00 s clips: ``U(3, 8)`` capped at the file hit its onset
    78 % of the time under the training rule. With the margin the take shrinks
    to ``file - 2 * margin`` and the crop sits at ``margin`` exactly."""
    df = _with_pool_duration(manifest, "E", 4.0)
    s = Sampler(df)
    pool = df.set_index("file_id").pool
    takes = [c for spec in s.epoch_specs(2000) for c in spec.components
             if pool[c.file_id] == "E"]
    assert takes
    for c in takes:
        assert c.source_offset_s >= 0.5 - 1e-9
        assert c.source_offset_s + c.duration_s <= 3.5 + 1e-9
        assert c.duration_s <= 3.0 + 1e-9


# --------------------------------------------------------------------------- #
# D-4: tiled to the span, every role, no gap and no overlap


def _slots(spec):
    """``(role, file_id) -> sorted tiles`` for a composed spec."""
    out = {}
    for c in spec.components:
        out.setdefault((c.role, c.file_id), []).append(c)
    return {k: sorted(v, key=lambda c: c.target_start_s) for k, v in out.items()}


def test_every_component_is_tiled_contiguously_to_its_span(sampler):
    lo, hi = sampler.cfg.take_range_s
    seen_multi = 0
    for spec in sampler.epoch_specs(1500):
        if spec.render_mode != "composed":
            continue
        for (role, _), tiles in _slots(spec).items():
            seen_multi += len(tiles) > 1
            lengths = {round(c.duration_s, 9) for c in tiles}
            assert len(lengths) == 1, "tiles of one take are equal-length"
            assert tiles[0].duration_s <= hi + 1e-9
            for a, b in zip(tiles, tiles[1:]):
                assert b.target_start_s == pytest.approx(a.target_start_s + a.duration_s)
            assert {c.gain_db for c in tiles} == {tiles[0].gain_db}, \
                "one gain per component, shared by its tiles"
    assert seen_multi > 500


def test_the_tiles_cover_exactly_the_component_span(sampler):
    """Overlap: every component spans ``[lead, duration - tail]``. Sequential:
    each spans its slot. Nothing is silent between lead and tail (D-4, the
    52.8 % silence cue)."""
    for spec in sampler.epoch_specs(1500):
        if spec.render_mode != "composed":
            continue
        slots = _slots(spec)
        starts = sorted(min(c.target_start_s for c in t) for t in slots.values())
        ends = sorted(max(c.target_start_s + c.duration_s for c in t)
                      for t in slots.values())
        lead = starts[0]
        if spec.structure == "overlap":
            assert all(s == pytest.approx(lead) for s in starts)
            assert len(set(round(e, 6) for e in ends)) == 1
        else:
            for (a_end, b_start) in zip(ends[:-1], starts[1:]):
                assert b_start == pytest.approx(a_end)
        assert ends[-1] <= spec.duration_s + 1e-6


def test_the_join_count_is_a_function_of_span_and_take_only(sampler):
    """``n_tiles = ceil(span / take)`` with ``take <= take_hi`` -- so a longer
    timeline means more joins, and nothing about the file does."""
    lo, hi = sampler.cfg.take_range_s
    for spec in sampler.epoch_specs(600):
        if spec.render_mode != "composed":
            continue
        for tiles in _slots(spec).values():
            span = sum(c.duration_s for c in tiles)
            assert len(tiles) >= math.ceil(span / hi - 1e-9)


# --------------------------------------------------------------------------- #
# D-5: the component floor


def test_rows_at_the_floor_are_drawn_and_rows_below_it_are_not(manifest):
    df = manifest.copy()
    b = df.index[df.pool == "B"]
    at_floor, below = b[:20], b[20:40]
    df.loc[at_floor, "duration_s"] = 2.0
    df.loc[below, "duration_s"] = 1.9
    s = Sampler(df)
    assert s.n_dropped_short == 20
    drawn = Counter(c.file_id for spec in s.epoch_specs(3000) for c in spec.components)
    assert any(f in drawn for f in df.loc[at_floor, "file_id"]), \
        "a 2.0 s row is usable under the tile rule (D-5)"
    assert not any(f in drawn for f in df.loc[below, "file_id"])
    # and a 2.0 s row yields tiles no longer than its 1.0 s interior
    for spec in s.epoch_specs(3000):
        for c in spec.components:
            if c.file_id in set(df.loc[at_floor, "file_id"]):
                assert c.duration_s <= 1.0 + 1e-9


def test_a_floor_that_leaves_no_interior_is_rejected():
    with pytest.raises(ValueError, match="component_floor_s"):
        DrawConfig(component_floor_s=1.0, edge_margin_s=0.5)
    DrawConfig(component_floor_s=1.01, edge_margin_s=0.5)   # just enough


def test_a_corpus_of_only_short_components_fails_loudly(manifest):
    df = manifest.copy()
    df.loc[df.row_kind == "component", "duration_s"] = 1.0
    with pytest.raises(ValueError, match="shorter than component_floor_s"):
        Sampler(df)


# --------------------------------------------------------------------------- #
# D-16: DOSS reaches whole-file rows


def _whole_file_manifest(manifest):
    """Cell-8 whole-file rows split 3:1 between two generators, long enough to
    be drawn whole."""
    df = manifest.copy()
    w = df.index[(df.row_kind == "whole_file") & (df.cell == 8)]
    big, small = w[: (3 * len(w)) // 4], w[(3 * len(w)) // 4:]
    keys = ["artifact_family", "domain_key", "source_name"]
    df.loc[big, keys] = ["gen_big", "gen_big::big", "gen_big"]
    df.loc[small, keys] = ["gen_small", "gen_small::small", "gen_small"]
    df.loc[w, "duration_s"] = 120.0
    return df, len(big), len(small)


def _whole_file_family_counts(df, cap):
    fam = df.set_index("file_id").artifact_family
    s = Sampler(df, DrawConfig(f8=0.0, domain_cap=cap))
    return Counter(fam[spec.components[0].file_id]
                   for spec in s.epoch_specs(6000) if spec.render_mode == "whole_file")


def test_whole_file_draws_are_domain_capped(manifest):
    """Per-generator whole-file counts ∝ ``min(count, cap)``: with the cap at
    the small generator's size the two are drawn equally."""
    df, n_big, n_small = _whole_file_manifest(manifest)
    assert n_big > 2 * n_small
    counts = _whole_file_family_counts(df, cap=n_small)
    assert counts["gen_big"] + counts["gen_small"] > 200
    ratio = counts["gen_big"] / counts["gen_small"]
    assert 0.8 < ratio < 1.25, counts


def test_without_the_cap_the_big_generator_dominates_whole_file_draws(manifest):
    """Mutation: a cap above every count is the training sampler's uniform
    draw, and the 3:1 imbalance reaches the stream."""
    df, n_big, n_small = _whole_file_manifest(manifest)
    counts = _whole_file_family_counts(df, cap=10_000)
    ratio = counts["gen_big"] / counts["gen_small"]
    assert ratio > 2.0, counts


def test_whole_file_crops_lie_inside_the_file_too(manifest):
    df, _, _ = _whole_file_manifest(manifest)
    s = Sampler(df, DrawConfig(f8=0.0))
    dur = _durations(df)
    n = 0
    for spec in s.epoch_specs(2000):
        if spec.render_mode != "whole_file":
            continue
        n += 1
        (c,) = spec.components
        assert c.source_offset_s >= 0.5 - 1e-9
        assert c.source_offset_s + c.duration_s <= dur[c.file_id] - 0.5 + 1e-9
        assert spec.duration_s == c.duration_s
    assert n > 100


# --------------------------------------------------------------------------- #
# determinism and the audit


def test_same_key_gives_an_identical_spec(sampler):
    assert sampler.sample_spec(7, epoch=1, seed=3) == sampler.sample_spec(7, epoch=1, seed=3)


@pytest.mark.parametrize("kw", [{"sample_id": 8}, {"epoch": 2}, {"seed": 4}])
def test_changing_any_key_component_changes_the_stream(sampler, kw):
    base = dict(sample_id=7, epoch=1, seed=3)
    assert sampler.sample_spec(**base) != sampler.sample_spec(**{**base, **kw})


def test_the_drawn_stream_passes_the_training_audit(sampler, manifest):
    """The specs are ``training.spec.SampleSpec``s, so ``training.audit`` judges
    them unchanged -- and the tile rule must not have manufactured a
    metadata shortcut of its own (I1b)."""
    report = run_audit(sampler, n=2000, manifest=manifest)
    assert report.ok, str(report)


# --------------------------------------------------------------------------- #
# every DrawConfig field is a knob that moves the stream

#: `field -> (value a, value b, observable, extra config)`, exhaustive.
_KNOBS: dict[str, tuple] = {
    "cell_mix": (CellMix(), CellMix({**REFERENCE_MIX, 6: 0.095, 8: 0.125}), "cells", {}),
    "f8": (0.0, 1.0, "cell8_composed", {}),
    "single_composed_rate": (0.0, 1.0, "voice_only_composed", {}),
    "noise_composed_rate": (0.0, 1.0, "cell9_composed",
                            {"balance_marginal_composedness": False}),
    "balance_marginal_composedness": (True, False, "cell9_composed", {}),
    "domain_cap": (500, 1, "component_files", {}),
    "duration_range": ((4.0, 60.0), (4.0, 8.0), "durations", {}),
    "take_range_s": ((3.0, 8.0), (1.5, 1.5), "tile_lengths", {}),
    "edge_margin_s": (0.5, 0.0, "offsets", {}),
    "component_floor_s": (2.0, 40.0, "component_files", {}),
    "gain_db_range": ((-15.0, 15.0), (-1.0, 1.0), "gains", {}),
    "gain_db_mean": (-3.6, 6.0, "gains", {}),
    "gain_db_sigma": (4.0, 0.001, "gains", {}),
    "sequential_prob": (0.0, 1.0, "structures", {}),
    "crossfade_ms_range": ((10.0, 200.0), (300.0, 400.0), "crossfades", {}),
    "silence_lead_s": (3.0, 0.0, "starts", {}),
    "silence_tail_s": (1.0, 0.0, "ends", {}),
    "scheme_version": ("strategy-v1", "strategy-v2", "scheme", {}),
}

_NOT_A_STREAM_KNOB = {
    "allow_unsound_mix": "a validation hatch, not a draw parameter",
}


def _observables(manifest, n=400, **overrides):
    specs = list(Sampler(manifest, DrawConfig(**overrides)).epoch_specs(n))
    composed = [s for s in specs if s.render_mode == "composed"]
    return {
        "cells": Counter(s.cell for s in specs),
        "cell8_composed": sum(s.cell == 8 and s.render_mode == "composed" for s in specs),
        "cell9_composed": sum(s.cell == 9 and s.render_mode == "composed" for s in specs),
        "voice_only_composed": sum(s.stratum == "voice-only" and s.render_mode == "composed"
                                   for s in specs),
        "component_files": Counter(c.file_id for s in specs for c in s.components),
        "durations": tuple(round(s.duration_s, 6) for s in specs),
        "tile_lengths": tuple(round(c.duration_s, 6) for s in composed for c in s.components),
        "offsets": tuple(round(c.source_offset_s, 6) for s in composed for c in s.components),
        "gains": tuple(round(c.gain_db, 9) for s in specs for c in s.components),
        "structures": Counter(s.structure for s in composed),
        "crossfades": tuple(round(s.crossfade_ms, 9) for s in composed),
        "starts": tuple(round(c.target_start_s, 9) for s in composed for c in s.components),
        "ends": tuple(round(c.target_start_s + c.duration_s, 9)
                      for s in composed for c in s.components),
        "scheme": {s.scheme_version for s in specs},
    }


def test_the_knob_table_names_every_draw_config_field():
    named = set(_KNOBS) | set(_NOT_A_STREAM_KNOB)
    assert named == {f.name for f in dataclasses.fields(DrawConfig)}


@pytest.mark.parametrize("field", sorted(_KNOBS))
def test_every_draw_config_knob_changes_the_drawn_stream(field, manifest):
    a_value, b_value, key, extra = _KNOBS[field]
    a = _observables(manifest, **{**extra, field: a_value})
    b = _observables(manifest, **{**extra, field: b_value})
    assert a[key] != b[key], (
        f"DrawConfig.{field} = {a_value!r} and {b_value!r} produced the same "
        f"{key}: the knob validates and then does nothing")


#: The v1 values, pinned (docs/processing/03 §1). Changing one is a decision.
_V1_DEFAULTS = {
    "cell_mix": CellMix(),
    "f8": 1.0,
    "single_composed_rate": 0.0,
    "noise_composed_rate": 0.0,
    "balance_marginal_composedness": True,
    "domain_cap": 500,
    "duration_range": (4.0, 60.0),
    "take_range_s": (3.0, 8.0),
    "edge_margin_s": 0.5,
    "component_floor_s": 2.0,
    "gain_db_range": (-15.0, 15.0),
    "gain_db_mean": -3.6,
    "gain_db_sigma": 4.0,
    "sequential_prob": 0.25,
    "crossfade_ms_range": (10.0, 200.0),
    "silence_lead_s": 3.0,
    "silence_tail_s": 1.0,
    "scheme_version": "strategy-v1",
    "allow_unsound_mix": False,
}


def test_the_v1_defaults_are_pinned():
    cfg = DrawConfig()
    assert set(_V1_DEFAULTS) == {f.name for f in dataclasses.fields(DrawConfig)}, \
        "a new DrawConfig field arrived without a pinned default"
    for name, want in _V1_DEFAULTS.items():
        assert getattr(cfg, name) == want, name


@pytest.mark.parametrize("bad", [
    {"duration_range": (60.0, 4.0)},
    {"take_range_s": (8.0, 3.0)},
    {"take_range_s": (0.0, 3.0)},
    {"edge_margin_s": -0.1},
    {"domain_cap": 0},
    {"silence_lead_s": -1.0},
    {"sequential_prob": 1.5},
    {"crossfade_ms_range": (200.0, 10.0)},
    {"f8": 1.5},
])
def test_malformed_draw_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        DrawConfig(**bad)


def test_an_unsound_mix_must_be_asked_for():
    unsound = CellMix({1: 0.02, 2: 0.02, 3: 0.02, 4: 0.02, 5: 0.02,
                       6: 0.45, 7: 0.43, 8: 0.01, 9: 0.01})
    with pytest.raises(ValueError, match="C1|C3"):
        DrawConfig(cell_mix=unsound)
    DrawConfig(cell_mix=unsound, allow_unsound_mix=True)


# --------------------------------------------------------------------------- #
# YAML


def test_the_v1_config_file_names_every_field_and_equals_the_defaults():
    raw = yaml.safe_load((REPO / "configs" / "processing_v1.yaml").read_text())
    assert set(raw) == set(SECTIONS)
    for name, cls in SECTIONS.items():
        assert set(raw[name]) == {f.name for f in dataclasses.fields(cls)}, name
    assert load_processing_config(REPO / "configs" / "processing_v1.yaml") == ProcessingConfig()


def test_a_non_default_config_round_trips(tmp_path):
    alt = {"draw": {
        "cell_mix": {"p": {1: 0.060, 2: 0.130, 3: 0.060, 4: 0.135, 5: 0.155,
                           6: 0.095, 7: 0.125, 8: 0.125, 9: 0.115}},
        "f8": 0.5, "single_composed_rate": 0.5, "noise_composed_rate": 0.25,
        "balance_marginal_composedness": False, "domain_cap": 42,
        "duration_range": [5.0, 30.0], "take_range_s": [2.0, 4.0],
        "edge_margin_s": 0.25, "component_floor_s": 1.0,
        "gain_db_range": [-10.0, 10.0], "gain_db_mean": -1.0, "gain_db_sigma": 2.0,
        "sequential_prob": 0.75, "crossfade_ms_range": [5.0, 50.0],
        "silence_lead_s": 0.4, "silence_tail_s": 0.3,
        "scheme_version": "strategy-v2", "allow_unsound_mix": True,
    }}
    assert set(alt["draw"]) == {f.name for f in dataclasses.fields(DrawConfig)}
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump(alt, sort_keys=False))
    cfg = load_processing_config(path)
    assert cfg.draw != DrawConfig()
    for k, v in alt["draw"].items():
        got = getattr(cfg.draw, k)
        if k == "cell_mix":
            assert got == CellMix({int(i): p for i, p in v["p"].items()})
        elif isinstance(v, list):
            assert got == tuple(v)
        else:
            assert got == v, k
    assert processing_config_from_dict(dump_processing_config(cfg)) == cfg


def test_unknown_keys_and_sections_are_errors():
    with pytest.raises(ConfigError, match="unknown key"):
        processing_config_from_dict({"draw": {"take_range": [3.0, 8.0]}})
    with pytest.raises(ConfigError, match="unknown section"):
        processing_config_from_dict({"sampler": {}})
