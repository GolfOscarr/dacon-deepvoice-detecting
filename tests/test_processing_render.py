"""The processing renderer, against real files on disk: sample-exact placement,
a taper at every joint, no silence inside the components, merged frame
intervals, and ``render(spec) == render(spec)``.

Critical: each invariant is mutation-tested -- shown failing on a deliberately
broken spec or config -- before it is trusted on the real one.
"""

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from models.config import AudioConfig
from models.losses import TARGET_FOR_COLUMN
from processing.config import DrawConfig
from processing.render import (ManifestIndex, RenderConfig, frame_intervals_for,
                               placements_for, render)
from processing.sampler import Sampler
from training.spec import ComponentDraw, SampleSpec
from training.synthetic import synthetic_manifest, write_synthetic_corpus

REPO = Path(__file__).resolve().parents[1]
CORPUS = dict(n_per_pool=5, n_whole_file=6, seed=0, duration_range=(7.0, 10.0))
SR = AudioConfig().sample_rate


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    manifest = synthetic_manifest(**CORPUS)
    write_synthetic_corpus(manifest, root, seed=0)
    return root, manifest, ManifestIndex.from_frame(manifest)


@pytest.fixture(scope="module")
def cfg(corpus):
    return RenderConfig(root=corpus[0])


@pytest.fixture(scope="module")
def raw_cfg(corpus):
    """No level target and no clip: the placement and layer tests measure
    the composite as summed (06 D6 / D7 are tested on their own)."""
    return RenderConfig(root=corpus[0], target_rms_dbfs=None, tile_rms_dbfs=None, clip=False)


#: The placement tests look at the COMPOSED canvas -- the lead exactly silent,
#: nothing outside the intervals -- so the menus are off: additive noise and
#: the codec chain (REN-3/4) fill the lead by design.
PLACEMENT = dict(duration_range=(4.0, 12.0), augments=(), normalize_menu=None)


@pytest.fixture(scope="module")
def drawn(corpus):
    """Specs the processing sampler actually draws -- tiles, leads and all --
    short enough to render quickly."""
    _, manifest, _ = corpus
    s = Sampler(manifest, DrawConfig(**PLACEMENT))
    return [spec for spec in s.epoch_specs(40) if spec.render_mode == "composed"]


def _ids(manifest, pool):
    return manifest[manifest.pool == pool].file_id.tolist()


def _tiled(manifest, *, n_tiles=2, tile=2.5, crossfade_ms=100.0, lead=0.0,
           role="voice", cell=1, sample_id=0):
    """One component, ``n_tiles`` equal tiles from the same file, each but the
    last reading ``tile + crossfade`` so consecutive tiles overlap by the
    crossfade, as the sampler draws them (06 D8)."""
    fid = _ids(manifest, "A" if role == "voice" else "C")[0]
    xf = crossfade_ms / 1000.0
    comps = tuple(ComponentDraw(fid, role, 0.5 + 0.7 * i,
                                tile + (xf if i < n_tiles - 1 else 0.0), lead + i * tile, 0.0)
                  for i in range(n_tiles))
    return SampleSpec(sample_id=sample_id, epoch=0, seed=0, scheme_version="x",
                      duration_s=lead + n_tiles * tile, cell=cell,
                      render_mode="composed", structure="overlap",
                      components=comps, crossfade_ms=crossfade_ms)


# --------------------------------------------------------------------------- #
# I10 -- render(spec) == render(spec)


def test_rendering_twice_is_bitwise_identical(corpus, cfg, drawn):
    _, _, index = corpus
    for spec in drawn[:10]:
        assert torch.equal(render(spec, index, cfg).wav, render(spec, index, cfg).wav)


def test_rendering_in_another_process_is_bitwise_identical(corpus, cfg, drawn):
    """I10 across processes: ``hash()`` is salted per process, and a same-process
    check cannot see an RNG keyed on it."""
    root, _, index = corpus
    spec = drawn[3]
    spec = SampleSpec.from_dict({**spec.to_dict(),
                                 "transforms": [["gaussian_noise", {}], ["rawboost_ssi", {}]]})
    mine = hashlib.sha256(render(spec, index, cfg).wav.numpy().tobytes()).hexdigest()
    code = (
        "import json,sys,hashlib\n"
        "from pathlib import Path\n"
        "from processing.render import render, RenderConfig, ManifestIndex\n"
        "from training.spec import SampleSpec\n"
        "from training.synthetic import synthetic_manifest\n"
        f"m = synthetic_manifest(**{CORPUS!r})\n"
        "spec = SampleSpec.from_dict(json.loads(sys.argv[2]))\n"
        "w = render(spec, ManifestIndex.from_frame(m),\n"
        "           RenderConfig(root=Path(sys.argv[1]))).wav\n"
        "print(hashlib.sha256(w.numpy().tobytes()).hexdigest())\n")
    proc = subprocess.run([sys.executable, "-c", code, str(root),
                           json.dumps(spec.to_dict())],
                          cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == mine


def test_a_different_sample_id_renders_different_audio(corpus, cfg, drawn):
    _, _, index = corpus
    a, b = drawn[0], drawn[1]
    assert not torch.equal(render(a, index, cfg).wav[:, :SR],
                           render(b, index, cfg).wav[:, :SR])


# --------------------------------------------------------------------------- #
# sample-exact placement


def test_every_placement_is_on_the_grid_and_covers_exactly_its_draw(drawn):
    for spec in drawn:
        total = int(round(spec.duration_s * SR))
        for place in placements_for(spec, SR):
            draw = spec.components[place.index]
            assert place.start == int(round(draw.target_start_s * SR))
            assert place.end <= total
            assert place.n > 0


def test_consecutive_tiles_overlap_by_exactly_the_crossfade(drawn):
    """The sampler draws tiles as ``start + i * tile`` reading ``tile + xfade``;
    on the grid consecutive tiles of one slot overlap by the crossfade's
    samples (06 D8), to the rounding of one sample."""
    seen = 0
    for spec in drawn:
        xf = int(round(spec.crossfade_ms / 1000.0 * SR))
        by_key = {}
        for place in placements_for(spec, SR):
            d = spec.components[place.index]
            by_key.setdefault((d.role, d.slot), []).append(place)
        for places in by_key.values():
            places.sort(key=lambda p: p.start)
            for a, b in zip(places, places[1:]):
                seen += 1
                assert abs((a.end - b.start) - xf) <= 1, (spec.sample_id, a, b, xf)
    assert seen > 30


def test_the_rendered_length_is_the_timeline(corpus, cfg, drawn):
    _, _, index = corpus
    for spec in drawn[:10]:
        assert render(spec, index, cfg).wav.shape[-1] == int(round(spec.duration_s * SR))


# --------------------------------------------------------------------------- #
# D-4 -- a taper at every joint


def _rms_np(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def test_a_tile_joint_is_an_equal_power_crossfade_not_a_dip(corpus, raw_cfg):
    """06 D8 / 05 A5: the earlier taper faded both tiles to zero AT the joint
    (a 100 ms dip to silence every tile). Now the tiles overlap by the
    crossfade and fade with equal-power ramps: the level through the joint
    stays that of the tiles, and no sample in the overlap is zero."""
    _, manifest, index = corpus
    faded = render(_tiled(manifest, crossfade_ms=100.0), index, raw_cfg).wav.numpy()
    cut = render(_tiled(manifest, crossfade_ms=0.0), index, raw_cfg).wav.numpy()
    joint, xf = int(2.5 * SR), int(0.1 * SR)
    overlap = slice(joint, joint + xf)
    before = slice(joint - 5 * xf, joint - xf)
    ratio = _rms_np(faded[:, overlap]) / _rms_np(faded[:, before])
    assert 0.7 < ratio < 1.4, ratio
    assert float(np.abs(faded[:, overlap]).min(axis=0).max()) > 0.0
    assert not np.any(np.all(faded[:, overlap] == 0.0, axis=0))
    # away from the joint the two are the same audio
    assert np.allclose(faded[:, :joint - 10], cut[:, :joint - 10], atol=1e-6)
    # the hard cut is what a crossfade of 0 means, deliberately
    assert not np.allclose(faded[:, overlap], cut[:, overlap], atol=1e-6)


def test_the_crossfade_ramps_are_equal_power(raw_cfg):
    from processing.render import _ramps
    out, inn = _ramps(1600, raw_cfg.crossfade_shape)
    np.testing.assert_allclose(out ** 2 + inn ** 2, 1.0, atol=1e-6)
    assert out[0] == 1.0 and inn[0] == 0.0 and out[-1] == 0.0 and inn[-1] == 1.0
    assert np.all(np.diff(inn) >= 0) and np.all(np.diff(out) <= 0)


def test_every_joint_of_a_many_tile_component_keeps_its_level(corpus, raw_cfg):
    _, manifest, index = corpus
    spec = _tiled(manifest, n_tiles=4, tile=1.5, crossfade_ms=50.0)
    wav = render(spec, index, raw_cfg).wav.numpy()
    xf = int(0.05 * SR)
    body = _rms_np(wav[:, int(0.3 * SR):int(1.2 * SR)])
    for i in range(1, 4):
        joint = int(round(i * 1.5 * SR))
        assert 0.6 < _rms_np(wav[:, joint:joint + xf]) / body < 1.5, i
        assert not np.any(np.all(wav[:, joint:joint + xf] == 0.0, axis=0)), i


def test_the_first_tile_is_not_faded_in_and_the_last_not_out(corpus, cfg):
    """Only joints are tapered: a component's own onset and end are untouched,
    or the taper itself becomes the silence cue D-4 removes."""
    _, manifest, index = corpus
    lead = 1.0
    spec = _tiled(manifest, n_tiles=2, tile=2.0, crossfade_ms=200.0, lead=lead)
    wav = render(spec, index, cfg).wav
    start, end = int(lead * SR), int((lead + 4.0) * SR)
    k = int(0.2 * SR)
    head = float(wav[:, start:start + k].pow(2).mean())
    tail = float(wav[:, end - k:end].pow(2).mean())
    mid = float(wav[:, start + k: start + 2 * k].pow(2).mean())
    assert head > 0.5 * mid and tail > 0.5 * mid


def test_overlap_components_sharing_a_start_get_no_taper(corpus, cfg):
    _, manifest, index = corpus
    voice, music = _ids(manifest, "A")[0], _ids(manifest, "C")[0]
    spec = SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x",
                      duration_s=5.0, cell=5, render_mode="composed", structure="overlap",
                      components=(ComponentDraw(voice, "voice", 0.5, 5.0, 0.0, 0.0),
                                  ComponentDraw(music, "music", 0.5, 5.0, 0.0, 0.0)),
                      crossfade_ms=200.0)
    wav = render(spec, index, cfg).wav
    k = int(0.2 * SR)
    assert float(wav[:, :k].pow(2).mean()) > 0.5 * float(wav[:, k:2 * k].pow(2).mean())


def test_a_sequential_joint_between_components_is_a_crossfade_too(corpus, raw_cfg):
    """The last tile of a sequential slot reads the crossfade past its slot
    and overlaps the next slot's first tile; the joint is faded like any other."""
    _, manifest, index = corpus
    voice, music = _ids(manifest, "A")[0], _ids(manifest, "C")[0]

    def seq(xf_ms):
        xf = xf_ms / 1000.0
        return SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x",
                          duration_s=6.0, cell=5, render_mode="composed",
                          structure="sequential",
                          components=(ComponentDraw(voice, "voice", 0.5, 3.0 + xf, 0.0, 0.0,
                                                    slot=0),
                                      ComponentDraw(music, "music", 0.5, 3.0, 3.0, 0.0,
                                                    slot=1)),
                          crossfade_ms=xf_ms)
    faded = render(seq(200.0), index, raw_cfg).wav.numpy()
    cut = render(seq(0.0), index, raw_cfg).wav.numpy()
    joint, xf = 3 * SR, int(0.2 * SR)
    ratio = _rms_np(faded[:, joint:joint + xf]) / _rms_np(faded[:, joint - 5 * xf:joint - xf])
    assert 0.7 < ratio < 1.4, ratio
    assert not np.allclose(faded[:, joint:joint + xf], cut[:, joint:joint + xf], atol=1e-6)
    assert np.allclose(faded[:, joint + xf + 10:], cut[:, joint + xf + 10:], atol=1e-6)


def test_overlap_sums_the_components(corpus, raw_cfg):
    cfg = raw_cfg
    _, manifest, index = corpus
    voice, music = _ids(manifest, "A")[0], _ids(manifest, "C")[0]

    def spec(gv, gm):
        return SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x",
                          duration_s=5.0, cell=5, render_mode="composed",
                          structure="overlap",
                          components=(ComponentDraw(voice, "voice", 0.5, 5.0, 0.0, gv),
                                      ComponentDraw(music, "music", 0.5, 5.0, 0.0, gm)))
    both = render(spec(0.0, 0.0), index, cfg).wav
    v = render(spec(0.0, -400.0), index, cfg).wav
    m = render(spec(-400.0, 0.0), index, cfg).wav
    assert torch.allclose(both, v + m, atol=1e-6)


# --------------------------------------------------------------------------- #
# D-4 -- no silence inside the components; the lead and tail are exact


def _longest_zero_run(x: np.ndarray) -> int:
    best = run = 0
    for v in x:
        run = run + 1 if v == 0 else 0
        best = max(best, run)
    return best


def test_no_silence_run_inside_the_components_beyond_the_taper_endpoints(corpus, cfg, drawn):
    """The 52.8 % silence cue (02 §2) is gone: between lead and tail the canvas
    is never digitally silent for longer than a taper's two zero endpoints."""
    _, _, index = corpus
    for spec in drawn[:20]:
        wav = render(spec, index, cfg).wav.abs().max(dim=0).values.numpy()
        lead = min(c.target_start_s for c in spec.components)
        end = max(c.target_start_s + c.duration_s for c in spec.components)
        inner = wav[int(round(lead * SR)):int(round(end * SR))]
        assert _longest_zero_run(inner) <= 2, spec.sample_id
        assert np.all(wav[:int(round(lead * SR))] == 0)
        assert np.all(wav[int(round(end * SR)):] == 0)


def test_a_gap_in_the_spec_would_be_seen(corpus, cfg):
    """Mutation for the test above: a spec whose tiles do not abut renders a
    silent gap the check must find."""
    _, manifest, index = corpus
    fid = _ids(manifest, "A")[0]
    spec = SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x",
                      duration_s=6.0, cell=1, render_mode="composed", structure="overlap",
                      components=(ComponentDraw(fid, "voice", 0.5, 2.0, 0.0, 0.0),
                                  ComponentDraw(fid, "voice", 0.5, 2.0, 4.0, 0.0)))
    wav = render(spec, index, cfg).wav.abs().max(dim=0).values.numpy()
    assert _longest_zero_run(wav[:6 * SR]) >= 2 * SR - 2


# --------------------------------------------------------------------------- #
# frame intervals -- I13


def test_tiles_merge_into_one_interval_per_component(corpus):
    _, manifest, _ = corpus
    spec = _tiled(manifest, n_tiles=4, tile=1.5, lead=0.5)
    iv = frame_intervals_for(spec)
    assert iv["voice"] == ((0.5, 6.5, 0),)
    assert iv["file"] == ((0.5, 6.5, 0),)
    assert iv["music"] == ()


def test_frame_intervals_union_is_the_placed_span(drawn):
    for spec in drawn:
        iv = frame_intervals_for(spec)
        for role in ("voice", "music"):
            placed = [c for c in spec.components if c.role == role]
            if not placed:
                assert iv[role] == ()
                continue
            lo = min(c.target_start_s for c in placed)
            hi = max(c.target_start_s + c.duration_s for c in placed)
            assert iv[role][0][0] == pytest.approx(lo)
            assert iv[role][-1][1] == pytest.approx(hi)
            # tiles overlap by the crossfade, so the merged span is [lo, hi)
            covered = sum(e - s for s, e, _ in iv[role])
            assert covered == pytest.approx(hi - lo)


def test_frame_labels_follow_the_component(corpus):
    _, manifest, _ = corpus
    voice, music = _ids(manifest, "B")[0], _ids(manifest, "C")[0]
    spec = SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x",
                      duration_s=5.0, cell=7, render_mode="composed", structure="overlap",
                      components=(ComponentDraw(voice, "voice", 0.5, 5.0, 0.0, 0.0),
                                  ComponentDraw(music, "music", 0.5, 5.0, 0.0, 0.0)))
    iv = frame_intervals_for(spec)
    assert iv["voice"] == ((0.0, 5.0, 1),)
    assert iv["music"] == ((0.0, 5.0, 0),)
    assert sorted(iv["file"]) == [(0.0, 5.0, 0), (0.0, 5.0, 1)]


def test_frame_intervals_match_the_audio(corpus, cfg, drawn):
    """I13: outside every interval the canvas is zero; inside it is not."""
    _, _, index = corpus
    for spec in drawn[:10]:
        r = render(spec, index, cfg)
        wav = r.wav.abs().max(dim=0).values.numpy()
        mask = np.zeros(wav.shape[-1], dtype=bool)
        for s, e, _ in r.frame_intervals["file"]:
            mask[int(round(s * SR)):int(round(e * SR))] = True
        assert np.all(wav[~mask] == 0)
        assert wav[mask].mean() > 0


# --------------------------------------------------------------------------- #
# the rest of the contract


def test_targets_are_exactly_the_loss_keys(corpus, cfg, drawn):
    _, _, index = corpus
    r = render(drawn[0], index, cfg)
    assert set(r.targets) == set(TARGET_FOR_COLUMN.values())


def test_the_length_regime_guard_fires(corpus, cfg):
    _, manifest, index = corpus
    spec = _tiled(manifest, n_tiles=1, tile=2.0)          # 2 s < the 4 s floor
    with pytest.raises(ValueError, match="I12"):
        render(spec, index, cfg)
    render(spec, index, RenderConfig(root=cfg.root, check_duration=False))


def test_the_test_chain_is_applied_last(corpus, cfg):
    """A normalize draw reaches ``_normalize``: a flac round trip keeps the
    length, mono collapses the channels."""
    _, manifest, index = corpus
    spec = _tiled(manifest, n_tiles=2, tile=2.5)
    plain = render(spec, index, cfg).wav
    chained = render(SampleSpec.from_dict({**spec.to_dict(),
                                           "normalize": {"container": "flac",
                                                         "channels": "mono"}}),
                     index, cfg).wav
    assert chained.shape[0] == 1 and chained.shape[-1] == plain.shape[-1]


@pytest.fixture(scope="module")
def whole(corpus):
    """Whole-file specs of every cell the corpus scrapes whole."""
    _, manifest, _ = corpus
    s = Sampler(manifest, DrawConfig(f8=0.0, **PLACEMENT))
    out = {}
    for spec in s.epoch_specs(600):
        if spec.render_mode == "whole_file" and spec.cell not in out:
            out[spec.cell] = spec
    return out


def test_whole_file_specs_render_to_the_timeline_with_the_lead_silent(corpus, cfg, whole):
    _, _, index = corpus
    # the 6-row fixture scrapes three cells whole; 8 is the both-roles case
    assert 8 in whole and len(whole) >= 3, sorted(whole)
    for spec in whole.values():
        r = render(spec, index, cfg)
        assert r.wav.shape[-1] == int(round(spec.duration_s * SR))
        wav = r.wav.abs().max(dim=0).values.numpy()
        lead = min(c.target_start_s for c in spec.components)
        end = max(c.target_start_s + c.duration_s for c in spec.components)
        assert np.all(wav[:int(round(lead * SR))] == 0)
        assert _longest_zero_run(wav[int(round(lead * SR)):int(round(end * SR))]) <= 2
        assert torch.equal(r.wav, render(spec, index, cfg).wav)


def test_whole_file_frame_intervals_name_every_present_role_from_the_cell(whole):
    """A cell-8 row is voice AND music, fake on both; its tiles describe both
    branches at once, merged into one span each."""
    for cell, spec in whole.items():
        iv = frame_intervals_for(spec)
        lead = min(c.target_start_s for c in spec.components)
        end = max(c.target_start_s + c.duration_s for c in spec.components)
        for role in ("voice", "music"):
            present = getattr(spec, f"{role}_present")
            if present:
                assert iv[role] == ((pytest.approx(lead), pytest.approx(end),
                                     int(getattr(spec, f"{role}_fake") or 0)),), (cell, iv)
            else:
                assert iv[role] == (), (cell, iv)
        assert iv["file"] and all(lbl == spec.file_fake for _, _, lbl in iv["file"]), (cell, iv)
        if cell == 9:
            assert iv["file"] == ((pytest.approx(lead), pytest.approx(end), 0),)


def test_whole_file_frame_intervals_match_the_audio(corpus, cfg, whole):
    _, _, index = corpus
    for spec in whole.values():
        r = render(spec, index, cfg)
        wav = r.wav.abs().max(dim=0).values.numpy()
        mask = np.zeros(wav.shape[-1], dtype=bool)
        for s, e, _ in r.frame_intervals["file"]:
            mask[int(round(s * SR)):int(round(e * SR))] = True
        assert np.all(wav[~mask] == 0) and wav[mask].mean() > 0


def test_a_fully_drawn_spec_renders_through_the_augment_and_test_chains(corpus, cfg):
    """DRAW-6/7 -> REN-3/4: the v1 menus, as drawn, reach the renderer -- the
    length is the timeline, the channel draw is honoured, and the render is
    still bitwise reproducible with noise and a codec in the chain."""
    _, manifest, index = corpus
    s = Sampler(manifest, DrawConfig(duration_range=(4.0, 6.0)))
    specs = list(s.epoch_specs(12))
    assert any(spec.transforms for spec in specs)
    assert any(spec.normalize.get("container") == "mp3" for spec in specs)
    for spec in specs:
        r = render(spec, index, cfg)
        assert r.wav.shape[-1] == int(round(spec.duration_s * SR))
        if spec.normalize.get("channels") == "mono":
            assert r.wav.shape[0] == 1
        elif spec.normalize.get("channels") == "stereo":
            assert r.wav.shape[0] == 2
        assert torch.equal(r.wav, render(spec, index, cfg).wav)


# --------------------------------------------------------------------------- #
# DRAW-5 -- the noise layer at render


def _layered(manifest, snr_db, n_tiles=2, lead=0.5):
    voice = _ids(manifest, "A")[0]
    noise = _ids(manifest, "E")[0]
    tile = 2.0
    comps = [ComponentDraw(voice, "voice", 0.5 + 0.3 * i, tile, lead + i * tile, 0.0)
             for i in range(n_tiles)]
    comps += [ComponentDraw(noise, "noise", 0.5 + 0.3 * i, tile, lead + i * tile, 0.0,
                            snr_db=snr_db) for i in range(n_tiles)]
    return SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x",
                      duration_s=lead + n_tiles * tile + 0.5, cell=1, render_mode="composed",
                      structure="overlap", components=tuple(comps), crossfade_ms=0.0)


def _rms(x):
    return float(x.double().pow(2).mean().sqrt())


def test_a_layer_sits_at_its_snr_below_the_composite(corpus, raw_cfg):
    cfg = raw_cfg
    _, manifest, index = corpus
    for snr in (10.0, 20.0):
        spec = _layered(manifest, snr)
        dry = render(SampleSpec.from_dict({**spec.to_dict(), "components": [
            c for c in spec.to_dict()["components"] if c["snr_db"] is None]}), index, cfg).wav
        wet = render(spec, index, cfg).wav
        span = slice(int(0.5 * SR), int(4.5 * SR))
        noise = wet[:, span] - dry[:, span]
        measured = 20 * np.log10(_rms(dry[:, span]) / _rms(noise))
        assert measured == pytest.approx(snr, abs=0.05), (snr, measured)
        # the composite itself is untouched, and the lead stays silent
        assert float(wet[:, :int(0.5 * SR)].abs().max()) == 0.0


def test_a_layer_carries_no_frame_target(corpus):
    _, manifest, _ = corpus
    iv = frame_intervals_for(_layered(manifest, 20.0))
    assert iv["voice"] == ((0.5, 4.5, 0),) and iv["file"] == ((0.5, 4.5, 0),)


def test_a_layer_under_a_silent_composite_is_skipped(corpus, cfg):
    """No reference RMS, no layer -- rather than a divide by zero or a layer at
    full scale."""
    _, manifest, index = corpus
    voice, noise = _ids(manifest, "A")[0], _ids(manifest, "E")[0]
    spec = SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x", duration_s=5.0, cell=1,
                      render_mode="composed", structure="overlap",
                      components=(ComponentDraw(voice, "voice", 0.5, 4.0, 0.0, -400.0),
                                  ComponentDraw(noise, "noise", 0.5, 4.0, 0.0, 0.0, snr_db=20.0)))
    wav = render(spec, index, cfg).wav
    assert float(wav.abs().max()) < 1e-6


def test_drawn_specs_with_layers_render_reproducibly(corpus, cfg):
    _, manifest, index = corpus
    s = Sampler(manifest, DrawConfig(**{**PLACEMENT, "p_noise_layer": 1.0}))
    specs = [sp for sp in s.epoch_specs(8)]
    assert all(any(c.snr_db is not None for c in sp.components) for sp in specs)
    for sp in specs:
        r = render(sp, index, cfg)
        assert r.wav.shape[-1] == int(round(sp.duration_s * SR))
        assert torch.equal(r.wav, render(sp, index, cfg).wav)


def test_a_multichannel_piece_composes_with_a_stereo_one_at_two_channels(corpus, raw_cfg):
    cfg = raw_cfg
    """Pool E holds 8- and 30-channel array recordings. The composite is
    capped at two channels: the array piece keeps its leading channels, the
    stereo piece is untouched, mono is broadcast. Before the cap, `_to_channels`
    could not lift 2 to 8 and the composite raised."""
    from processing.render import _compose, MAX_CHANNELS
    _, manifest, _ = corpus
    spec = _tiled(manifest, n_tiles=1, tile=2.0)
    n = int(round(2.0 * SR))
    pieces = [np.random.default_rng(0).standard_normal((8, n)).astype(np.float32),
              np.random.default_rng(1).standard_normal((2, n)).astype(np.float32)]
    two = SampleSpec(**{**spec.to_dict(), "components": (
        spec.components[0], ComponentDraw(spec.components[0].file_id, "noise", 0.0, 2.0,
                                          spec.components[0].target_start_s, 0.0))})
    placements = placements_for(two, SR)
    out = _compose(two, placements, pieces, cfg, SR)
    assert out.shape[0] == MAX_CHANNELS == 2
    s = placements[0].start
    np.testing.assert_allclose(out[:, s:s + n], pieces[0][:2] + pieces[1], rtol=1e-6)


def test_an_invalid_crossfade_shape_is_rejected(cfg):
    with pytest.raises(ValueError, match="crossfade_shape"):
        RenderConfig(root=cfg.root, crossfade_shape="cosine")


# --------------------------------------------------------------------------- #
# 06 D6 / D7 / D11, 05 B6 / C1: the level, the clip, the render stream, the guard


def test_the_composite_is_scaled_to_the_level_target_before_the_augments(corpus, raw_cfg):
    """06 D7: a mixed sample is the sum of two components and was louder by
    construction (presence heads 0.67 / 0.78 after the shipped chain)."""
    _, manifest, index = corpus
    voice, music = _ids(manifest, "A")[0], _ids(manifest, "C")[0]
    one = SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x", duration_s=5.0, cell=1,
                     render_mode="composed", structure="overlap",
                     components=(ComponentDraw(voice, "voice", 0.5, 4.0, 0.5, 0.0),))
    two = SampleSpec(**{**one.to_dict(), "cell": 5, "components": (
        ComponentDraw(voice, "voice", 0.5, 4.0, 0.5, 0.0),
        ComponentDraw(music, "music", 0.5, 4.0, 0.5, 0.0, slot=1))})
    levelled = dataclasses.replace(raw_cfg, target_rms_dbfs=-23.0)
    span = slice(int(0.5 * SR), int(4.5 * SR))
    for spec in (one, two):
        wav = render(spec, index, levelled).wav.numpy()
        assert 20 * np.log10(_rms_np(wav[:, span])) == pytest.approx(-23.0, abs=0.05)
        assert np.all(wav[:, :int(0.5 * SR)] == 0.0), "the lead stays silent"
    raw_one = render(one, index, raw_cfg).wav.numpy()
    raw_two = render(two, index, raw_cfg).wav.numpy()
    assert _rms_np(raw_two[:, span]) > 1.2 * _rms_np(raw_one[:, span]), \
        "without the target the sum is louder -- the cue D7 removes"


def test_a_silent_composite_is_not_amplified_to_the_target(corpus, cfg):
    _, manifest, index = corpus
    voice = _ids(manifest, "A")[0]
    spec = SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x", duration_s=5.0, cell=1,
                      render_mode="composed", structure="overlap",
                      components=(ComponentDraw(voice, "voice", 0.5, 4.0, 0.0, -400.0),))
    assert float(render(spec, index, cfg).wav.abs().max()) < 1e-6


def test_over_range_samples_are_clipped_before_the_test_chain(corpus, raw_cfg):
    """06 D6 / 05 A4: a third of samples exceeded 1.0 and the containers
    disagreed about it. Clipped, every leg sees what a wav on disk holds."""
    _, manifest, index = corpus
    voice = _ids(manifest, "A")[0]
    loud = SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x", duration_s=5.0, cell=1,
                      render_mode="composed", structure="overlap",
                      components=(ComponentDraw(voice, "voice", 0.5, 4.0, 0.0, 40.0),))
    unclipped = render(loud, index, raw_cfg).wav
    assert float(unclipped.abs().max()) > 1.0, "the fixture must exceed full scale"
    clipped = render(loud, index, dataclasses.replace(raw_cfg, clip=True)).wav
    assert float(clipped.abs().max()) <= 1.0
    assert torch.equal(clipped, unclipped.clamp(-1.0, 1.0))


def test_the_augment_chain_draws_from_the_render_stream_not_the_draw_stream(corpus, cfg):
    """05 B6: rawboost's first draw was the sampler's first draw, so its SNR
    was an exact function of the sample's duration. The render stream is a
    different domain of the same key."""
    from training.spec import spec_rng
    _, manifest, index = corpus
    a, b = spec_rng(7, 0, 0), spec_rng(7, 0, 0, "render")
    assert a.random() != b.random()
    assert spec_rng(7, 0, 0, "render").random() == spec_rng(7, 0, 0, "render").random()
    spec = _tiled(manifest, crossfade_ms=50.0)
    spec = dataclasses.replace(
        spec, transforms=(("gaussian_noise", {"snr_db_range": (20.0, 20.0)}),))
    r1 = render(spec, index, cfg).wav
    r2 = render(spec, index, cfg).wav
    assert torch.equal(r1, r2)


def test_rendering_is_bitwise_identical_across_torch_thread_counts(corpus, cfg):
    """06 D11 / 05 B14: 20 of 40 hashes differed between 1 and 16 threads
    before the augment chain was pinned to one thread."""
    _, manifest, index = corpus
    spec = _tiled(manifest, crossfade_ms=50.0)
    spec = dataclasses.replace(spec, transforms=(
        ("pink_noise", {"snr_db": 15.0}), ("rawboost_ssi", {"snr_db": 20.0, "tilt_db": 3.0})))
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        one = render(spec, index, cfg).wav
        torch.set_num_threads(max(2, min(8, before)))
        many = render(spec, index, cfg).wav
    finally:
        torch.set_num_threads(before)
    assert torch.equal(one, many)
    assert torch.get_num_threads() == before, "the pin is restored"


def test_the_training_renderer_refuses_a_processing_spec(corpus):
    """05 C1: fed a processing spec the old renderer rendered different audio
    without a word (the layer at 0 dB, no crossfade). Now it raises."""
    from training.render import RenderConfig as OldConfig, render as old_render
    root, manifest, index = corpus
    spec = _layered(manifest, 20.0)
    with pytest.raises(ValueError, match="processing.render"):
        old_render(spec, index, OldConfig(root=root))
    slotted = dataclasses.replace(_tiled(manifest), components=tuple(
        dataclasses.replace(c, slot=1) for c in _tiled(manifest).components))
    with pytest.raises(ValueError, match="processing.render"):
        old_render(slotted, index, OldConfig(root=root))
    old_render(_tiled(manifest, crossfade_ms=0.0), index, OldConfig(root=root))   # plain: fine


def test_a_layer_sits_at_its_snr_with_crossfaded_tiles(corpus, raw_cfg):
    """05 B5: measured on the untapered tiles the layer landed 0.2-0.8 dB
    quiet; the reference and the layer are now both the tapered audio."""
    _, manifest, index = corpus
    voice, noise = _ids(manifest, "A")[0], _ids(manifest, "E")[0]
    xf = 0.2
    comps = [ComponentDraw(voice, "voice", 0.5 + 0.3 * i, 2.0 + (xf if i < 2 else 0.0),
                           0.5 + 2.0 * i, 0.0) for i in range(3)]
    comps += [ComponentDraw(noise, "noise", 0.5 + 0.3 * i, 2.0 + (xf if i < 2 else 0.0),
                            0.5 + 2.0 * i, 0.0, snr_db=15.0, slot=8) for i in range(3)]
    spec = SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x", duration_s=7.0,
                      cell=1, render_mode="composed", structure="overlap",
                      components=tuple(comps), crossfade_ms=200.0)
    dry = render(SampleSpec.from_dict({**spec.to_dict(), "components": [
        c for c in spec.to_dict()["components"] if c["snr_db"] is None]}), index, raw_cfg).wav
    wet = render(spec, index, raw_cfg).wav
    span = slice(int(0.5 * SR), int(6.5 * SR))
    measured = 20 * np.log10(_rms(dry[:, span]) / _rms(wet[:, span] - dry[:, span]))
    assert measured == pytest.approx(15.0, abs=0.05), measured
