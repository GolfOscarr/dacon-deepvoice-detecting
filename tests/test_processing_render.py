"""The processing renderer, against real files on disk: sample-exact placement,
a taper at every joint, no silence inside the components, merged frame
intervals, and ``render(spec) == render(spec)``.

Critical: each invariant is mutation-tested -- shown failing on a deliberately
broken spec or config -- before it is trusted on the real one.
"""

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
def drawn(corpus):
    """Specs the processing sampler actually draws -- tiles, leads and all --
    short enough to render quickly."""
    _, manifest, _ = corpus
    s = Sampler(manifest, DrawConfig(duration_range=(4.0, 12.0)))
    return [spec for spec in s.epoch_specs(40) if spec.render_mode == "composed"]


def _ids(manifest, pool):
    return manifest[manifest.pool == pool].file_id.tolist()


def _tiled(manifest, *, n_tiles=2, tile=2.5, crossfade_ms=100.0, lead=0.0,
           role="voice", cell=1, sample_id=0):
    """One component, ``n_tiles`` equal tiles from the same file, back to back."""
    fid = _ids(manifest, "A" if role == "voice" else "C")[0]
    comps = tuple(ComponentDraw(fid, role, 0.5 + 0.7 * i, tile, lead + i * tile, 0.0)
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


def test_abutting_tiles_land_on_adjacent_samples(drawn):
    """The sampler draws tiles as ``start + i * tile``; on the grid they must
    neither overlap nor leave a one-sample gap."""
    seen = 0
    for spec in drawn:
        by_key = {}
        for place in placements_for(spec, SR):
            d = spec.components[place.index]
            by_key.setdefault((d.role, d.file_id), []).append(place)
        for places in by_key.values():
            places.sort(key=lambda p: p.start)
            for a, b in zip(places, places[1:]):
                seen += 1
                assert b.start == a.end, (spec.sample_id, a, b)
    assert seen > 30


def test_the_rendered_length_is_the_timeline(corpus, cfg, drawn):
    _, _, index = corpus
    for spec in drawn[:10]:
        assert render(spec, index, cfg).wav.shape[-1] == int(round(spec.duration_s * SR))


# --------------------------------------------------------------------------- #
# D-4 -- a taper at every joint


def test_a_tile_joint_is_faded_rather_than_cut(corpus, cfg):
    _, manifest, index = corpus
    faded = render(_tiled(manifest, crossfade_ms=100.0), index, cfg).wav
    cut = render(_tiled(manifest, crossfade_ms=0.0), index, cfg).wav
    joint = int(2.5 * SR)
    window = slice(joint - 400, joint + 400)
    assert float(faded[:, window].abs().max()) < 0.1 * float(cut[:, window].abs().max())
    # the endpoints are exact: the last sample before and the first after are 0
    assert float(faded[:, joint - 1].abs().max()) == 0.0
    assert float(faded[:, joint].abs().max()) == 0.0
    # away from the joint the two are the same audio
    assert torch.allclose(faded[:, :joint - 2000], cut[:, :joint - 2000], atol=1e-6)


def test_every_joint_of_a_many_tile_component_is_tapered(corpus, cfg):
    _, manifest, index = corpus
    spec = _tiled(manifest, n_tiles=4, tile=1.5, crossfade_ms=50.0)
    wav = render(spec, index, cfg).wav
    for i in range(1, 4):
        joint = int(round(i * 1.5 * SR))
        assert float(wav[:, joint - 1].abs().max()) == 0.0, i
        assert float(wav[:, joint].abs().max()) == 0.0, i


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


def test_a_sequential_joint_between_components_is_tapered_too(corpus, cfg):
    _, manifest, index = corpus
    voice, music = _ids(manifest, "A")[0], _ids(manifest, "C")[0]

    def seq(xf):
        return SampleSpec(sample_id=0, epoch=0, seed=0, scheme_version="x",
                          duration_s=6.0, cell=5, render_mode="composed",
                          structure="sequential",
                          components=(ComponentDraw(voice, "voice", 0.5, 3.0, 0.0, 0.0),
                                      ComponentDraw(music, "music", 0.5, 3.0, 3.0, 0.0)),
                          crossfade_ms=xf)
    faded, cut = render(seq(200.0), index, cfg).wav, render(seq(0.0), index, cfg).wav
    joint = 3 * SR
    assert float(faded[:, joint - 400:joint + 400].abs().max()) < \
        float(cut[:, joint - 400:joint + 400].abs().max())


def test_overlap_sums_the_components(corpus, cfg):
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
            covered = sum(e - s for s, e, _ in iv[role])
            assert covered == pytest.approx(sum(c.duration_s for c in placed))


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


def test_the_interim_whole_file_spec_renders(corpus, cfg):
    _, manifest, index = corpus
    s = Sampler(manifest, DrawConfig(f8=0.0, duration_range=(4.0, 6.0)))
    whole = [spec for spec in s.epoch_specs(200) if spec.render_mode == "whole_file"]
    assert whole
    r = render(whole[0], index, cfg)
    assert r.wav.shape[-1] == int(round(whole[0].duration_s * SR))
    assert r.frame_intervals == {"voice": (), "music": (), "file": ()}


def test_an_invalid_crossfade_shape_is_rejected(cfg):
    with pytest.raises(ValueError, match="crossfade_shape"):
        RenderConfig(root=cfg.root, crossfade_shape="cosine")
