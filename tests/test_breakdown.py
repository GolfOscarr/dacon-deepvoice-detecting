"""Tier-3 diagnostic slice tests.

The central property under test: cell and artifact_family are label-determining,
so a within-slice EER is undefined for them and a shared contrast pool is
required. Getting this wrong yields an all-NaN diagnostic table that looks like
a plumbing bug rather than a specification error.
"""

import numpy as np
import pandas as pd
import pytest

from metrics.breakdown import breakdown_table, by, t3_gap, worst_cell_eer, worst_family_eer
from metrics.dacon import eer

# cell -> (voice_present, music_present, voice_fake, music_fake)
CELLS = {
    1: (1, 0, 0, 0), 2: (1, 0, 1, 0), 5: (1, 1, 0, 0),
    6: (1, 1, 0, 1), 7: (1, 1, 1, 0), 8: (1, 1, 1, 1),
}


def make_df(n=3000, seed=0, dead_cell=None, dead_share=0.08):
    """A frame with controllable per-cell difficulty and a realistic label map.

    The dead cell is deliberately a small share of the corpus -- that is the case
    worth testing, because a cell large enough to move the pooled number would
    not be hidden by it.
    """
    rng = np.random.default_rng(seed)
    cells = list(CELLS)
    if dead_cell is None:
        p = None
    else:
        p = np.full(len(cells), (1 - dead_share) / (len(cells) - 1))
        p[cells.index(dead_cell)] = dead_share
    cell = rng.choice(cells, n, p=p)
    vp, mp, vf, mf = (np.array([CELLS[c][i] for c in cell]) for i in range(4))
    file_fake = ((vp & vf) | (mp & mf)).astype(int)

    margin = np.where(cell == dead_cell, 0.0, 2.5)
    score = rng.normal(0, 1, n) + margin * file_fake

    families = np.where(file_fake == 1,
                        rng.choice(["hifigan", "encodec", "diffusion"], n),
                        rng.choice(["libritts", "musdb"], n))
    return pd.DataFrame({
        "file_id": [f"f{i}" for i in range(n)],
        "cell": cell, "voice_present": vp, "music_present": mp,
        "voice_fake": vf, "music_fake": mf, "file_fake": file_fake,
        "FILE_FAKE_PROB": score, "VOICE_FAKE_PROB": score, "MUSIC_FAKE_PROB": score,
        "artifact_family": families,
        "fold": rng.integers(0, 5, n),
        "pair_id": np.where(rng.random(n) < 0.3, "p" + rng.integers(0, 200, n).astype(str), None),
    })


def test_cell_is_label_determining():
    """The premise: within a cell there is only one class, so within-slice EER is NaN."""
    df = make_df(seed=0)
    for _, group in df.groupby("cell"):
        assert group["file_fake"].nunique() == 1
    within = by(df, "cell", contrast="within")
    assert within["eer"].isna().all()


def test_shared_contrast_makes_cells_scorable():
    df = make_df(seed=1)
    table = by(df, "cell")                       # contrast="auto"
    assert (table["contrast"] == "shared").all()
    assert table["eer"].notna().all()
    assert (table["n_pool"] > table["n_slice"]).all()


def test_by_cell_finds_the_collapsed_cell():
    df = make_df(dead_cell=7, seed=1)
    table = by(df, "cell").set_index("cell")
    assert table.loc[7, "eer"] > 0.30            # near chance against the shared pool
    assert table.loc[2, "eer"] < 0.12            # healthy


def test_a_small_dead_cell_is_invisible_in_the_pooled_number():
    """Why pooled numbers are not reportable on their own.

    A collapsed cell at 2% of the corpus moves the pooled EER by about a point --
    inside the +/-1.2pt measurement noise at our validation size, so it cannot be
    told apart from ordinary variation. The cell-level number sits at chance and
    is unmistakable. The claim is not that pooling hides the cell entirely; it is
    that pooling degrades it to something indistinguishable from noise.
    """
    def pooled_of(df):
        return eer(df["file_fake"].to_numpy(), df["FILE_FAKE_PROB"].to_numpy())

    healthy = [pooled_of(make_df(n=6000, seed=s)) for s in range(3)]
    damaged = [make_df(n=6000, seed=s, dead_cell=7, dead_share=0.02) for s in range(3)]

    pooled_shift = np.mean([pooled_of(d) for d in damaged]) - np.mean(healthy)
    worst = np.mean([worst_cell_eer(d) for d in damaged])

    assert pooled_shift < 0.02, f"pooled moved {pooled_shift:.3f} -- not a hidden failure"
    assert worst > 0.45, f"the dead cell should sit near chance, got {worst:.3f}"


def test_fold_is_label_crossing_and_scored_within():
    df = make_df(seed=3)
    table = by(df, "fold")
    assert (table["contrast"] == "within").all()
    assert table["eer"].notna().all()


def test_thin_slices_are_flagged_not_dropped():
    df = make_df(seed=3)
    df.loc[df.index[:5], "cell"] = 99             # a 5-row slice
    table = by(df, "cell", min_n=100).set_index("cell")
    assert bool(table.loc[99, "thin"]) is True
    assert 99 in table.index                      # kept, not dropped


def test_masked_pools_respected_in_slices():
    """Cells 1 and 2 have no music present, so the music head sees nothing there."""
    df = make_df(seed=4)
    music = by(df, "cell", head="music").set_index("cell")
    for c in (1, 2):
        assert music.loc[c, "n_slice"] == 0


def test_worst_family_and_breakdown_table_shape():
    df = make_df(seed=5)
    assert 0.0 <= worst_family_eer(df) <= 1.0
    table = breakdown_table(df)
    assert set(table["key"]) == {"cell", "artifact_family", "fold"}
    assert set(table["head"]) == {"file", "voice", "music"}
    assert set(table["contrast"]) <= {"within", "shared"}


def test_t3_gap_flags_corpus_identity_leakage():
    df = make_df(seed=6)
    # unpaired rows made trivially separable: pooled looks great, the matched
    # controls do not -- exactly the VG4 failure
    unpaired = df["pair_id"].isna()
    df.loc[unpaired, "VOICE_FAKE_PROB"] = df.loc[unpaired, "voice_fake"] * 50.0
    out = t3_gap(df, head="voice")
    assert out["t3_pair_eer"] > out["pooled_eer"] + 0.10
    assert out["passes_vg4"] is False


def test_t3_gap_passes_when_pairs_track_the_pool():
    df = make_df(seed=7)
    out = t3_gap(df, head="voice")
    assert out["passes_vg4"] is True
    assert out["n_pairs"] > 0


def test_by_rejects_unknown_key_and_contrast():
    with pytest.raises(KeyError):
        by(make_df(), "nope")
    with pytest.raises(ValueError, match="contrast"):
        by(make_df(), "cell", contrast="sideways")
