"""strategy-v6 music zero rule: measured inside the drawable span (edges excluded), on
all channels, so the same rule applies to every music piece on both sides."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "strategy"))
from v6_music_filters import zero_stats  # noqa: E402


def _put(root: Path, fid: str, a: np.ndarray) -> None:
    p = root / fid.split(":", 1)[0] / (fid.split(":", 1)[1] + ".npy")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, a)


def test_zero_run_inside_span_counts_edges_do_not(tmp_path):
    a = np.ones((1, 16000 * 10), dtype=np.int16)
    a[:, :4000] = 0                                  # 250 ms at the head: outside the span
    _put(tmp_path, "x:a", a)
    assert zero_stats("x:a", tmp_path) == (0.0, 0.0)
    a[:, 80000:80480] = 0                            # 30 ms in the middle
    _put(tmp_path, "x:b", a)
    run, frac = zero_stats("x:b", tmp_path)
    assert abs(run - 30.0) < 1e-6 and frac > 0


def test_a_zero_needs_every_channel(tmp_path):
    a = np.ones((2, 16000 * 5), dtype=np.int16)
    a[0, 20000:21000] = 0                            # one channel only
    _put(tmp_path, "x:c", a)
    assert zero_stats("x:c", tmp_path) == (0.0, 0.0)
