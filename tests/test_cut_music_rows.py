"""strategy-v6 music cut: one rule for both sides (>= 15 s rows -> 10 s pieces, a last
remainder kept only when >= 5 s); pieces never leave their parent's fold."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "strategy"))
from cut_music_rows import piece_id, pieces_of  # noqa: E402


def test_short_rows_stay_whole():
    assert pieces_of(10.0) == [] and pieces_of(10.242) == [] and pieces_of(14.99) == []


def test_cut_rule():
    assert pieces_of(30.0) == [(0.0, 10.0), (10.0, 10.0), (20.0, 10.0)]
    p = pieces_of(29.99)                                   # fma: 10, 10, 9.99
    assert [o for o, _ in p] == [0.0, 10.0, 20.0] and abs(p[-1][1] - 9.99) < 1e-9
    assert pieces_of(84.0) == [(k * 10.0, 10.0) for k in range(8)]   # 4 s remainder dropped
    assert pieces_of(15.0) == [(0.0, 10.0), (10.0, 5.0)]


def test_piece_ids_live_under_their_own_cache_prefix():
    assert piece_id("fma:fma_small/000/000002.mp3", 3) == "cut10:fma/fma_small/000/000002.mp3~03"
