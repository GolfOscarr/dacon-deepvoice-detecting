"""OFF-6: the decode cache -- a slice read back is bit-identical to the same
slice of the quantised full-file resample, the renderer with a cache reads
only the cache, and the build is resumable and stereo-preserving."""

from pathlib import Path

import numpy as np
import pytest
import torch

from processing.cache import (build_cache, cache_path, decode_int16, encode_int16,
                              read_slice)
from processing.config import DrawConfig
from processing.render import DecodeError, ManifestIndex, RenderConfig, render
from processing.sampler import Sampler
from training.render import load_audio, resample_poly_to
from training.synthetic import synthetic_manifest, write_synthetic_corpus

SR = 16_000
CORPUS = dict(n_per_pool=4, n_whole_file=4, seed=0, duration_range=(6.0, 9.0))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    manifest = synthetic_manifest(**CORPUS)
    write_synthetic_corpus(manifest, root, seed=0)
    cache = tmp_path_factory.mktemp("cache16k")
    report = build_cache(manifest, root, cache, workers=4, skip_cells=())
    return root, manifest, cache, report


def test_the_build_writes_every_row_once_and_is_resumable(corpus):
    root, manifest, cache, report = corpus
    assert report["written"] == len(manifest) and report["failed"] == 0
    again = build_cache(manifest, root, cache, workers=2, skip_cells=())
    assert again["exists"] == len(manifest) and again["written"] == 0


def test_a_cached_slice_is_bit_identical_to_the_quantised_full_resample(corpus):
    root, manifest, cache, _ = corpus
    for row in manifest.head(6).to_dict(orient="records"):
        full = load_audio(root / row["path"], sample_rate=SR, resampler=resample_poly_to)
        q = encode_int16(full)
        start, dur = 1.25, 2.5
        got = read_slice(cache_path(cache, row["file_id"]), sample_rate=SR,
                         offset_s=start, duration_s=dur)
        a, n = int(round(start * SR)), int(round(dur * SR))
        assert got.shape == (full.shape[0], n)
        assert np.array_equal(encode_int16(got), q[:, a:a + n])
        assert np.array_equal(got, decode_int16(q[:, a:a + n]))


def test_a_cached_slice_is_close_to_a_fresh_decode_away_from_its_edges(corpus):
    """int16 (< 1 LSB) plus the resampler's edge transient: the two regimes
    agree inside the slice and are never mixed in one run."""
    root, manifest, cache, _ = corpus
    row = manifest.iloc[0]
    fresh = load_audio(root / row["path"], sample_rate=SR, offset_s=1.0, duration_s=2.0)
    cached = read_slice(cache_path(cache, row["file_id"]), sample_rate=SR,
                        offset_s=1.0, duration_s=2.0)
    inner = slice(200, -200)
    assert np.abs(fresh[:, inner] - cached[:, inner]).max() < 2.0 / 32768


def test_stereo_files_keep_their_channels_and_short_reads_are_refused(corpus):
    root, manifest, cache, _ = corpus
    stereo = manifest[manifest.orig_channels == 2].iloc[0]
    got = read_slice(cache_path(cache, stereo["file_id"]), sample_rate=SR, duration_s=1.0)
    assert got.shape[0] == 2
    with pytest.raises(DecodeError, match="short of"):
        read_slice(cache_path(cache, stereo["file_id"]), sample_rate=SR,
                   offset_s=float(stereo["duration_s"]) - 0.5, duration_s=3.0)


def test_the_renderer_with_a_cache_reads_only_the_cache(corpus):
    root, manifest, cache, _ = corpus
    index = ManifestIndex.from_frame(manifest)
    spec = next(iter(Sampler(manifest, DrawConfig(duration_range=(4.0, 6.0), augments=(),
                                                  normalize_menu=None)).epoch_specs(1)))
    with_cache = RenderConfig(root=root, cache_root=cache)
    a = render(spec, index, with_cache).wav
    assert torch.equal(a, render(spec, index, with_cache).wav)              # I10 in the regime
    b = render(spec, index, RenderConfig(root=root)).wav
    # the two regimes agree in RMS (int16 + the resampler's transient at each
    # tile edge, ~0.02 peak there) and are never bit-equal
    assert a.shape == b.shape and not torch.equal(a, b)
    assert float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()) < 0.02
    with pytest.raises(DecodeError, match="not in the cache"):
        render(spec, index, RenderConfig(root=root, cache_root=Path(cache) / "nowhere"))


def test_skipped_cells_are_not_written(tmp_path):
    manifest = synthetic_manifest(**CORPUS)
    root = write_synthetic_corpus(manifest, tmp_path / "c", seed=0)
    report = build_cache(manifest, root, tmp_path / "cache", workers=2, skip_cells=(8,))
    n8 = int((manifest.cell == 8).sum())
    assert report["files"] == len(manifest) - n8 and report["skipped_cells"] == [8]
    for fid in manifest.loc[manifest.cell == 8, "file_id"]:
        assert not cache_path(tmp_path / "cache", fid).exists()
