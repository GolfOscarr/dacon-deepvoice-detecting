"""G-EDA3 / G-EDA4 / G-EDA7 and the G4 / G7 tables over built tables
(processing/gates.py). Each gate is paired with the mutation that must trip it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from processing.gates import (FAIL, PASS, g4_table, g7_pack, g_eda3, g_eda4, g_eda7,
                              g_eda7_ledger, run_gates)
from training.synthetic import synthetic_manifest


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    m = synthetic_manifest().copy()
    m["corpus"] = m["source_name"].str.split("/").str[0]
    m["reassigned_from"] = None
    return m


def _files(manifest: pd.DataFrame, extra_dropped: pd.DataFrame | None = None) -> pd.DataFrame:
    """The EDA file table: every probed file, i.e. the survivors plus the dropped."""
    cols = ["file_id", "source_name", "pool", "cell", "duration_s"]
    f = manifest[cols].copy()
    f["source_name"] = manifest["corpus"]
    if extra_dropped is not None:
        f = pd.concat([f, extra_dropped[cols]], ignore_index=True)
    return f


def _verdict(files: pd.DataFrame, drop_mask: pd.Series, name: str = "usable_duration"
             ) -> pd.DataFrame:
    return pd.DataFrame({"file_id": files["file_id"].values, "filter": name,
                         "verdict": np.where(drop_mask.values, "drop", "keep"),
                         "reason": "", "threshold_version": f"{name}:v1"})


# --------------------------------------------------------------------------- #
# G-EDA3


def test_g_eda3_passes_on_the_synthetic_corpus_and_names_a_starved_side(manifest):
    """The synthetic corpus has 3 real publishers per pool; the real floor is 6."""
    assert g_eda3(manifest, floor=3).verdict == PASS
    assert g_eda3(manifest).verdict == FAIL
    starved = manifest.copy()
    d = starved["pool"] == "D"
    starved.loc[d, "source_name"] = "one-publisher"
    starved.loc[d, "artifact_family"] = "one-family"
    res = g_eda3(starved, floor=3)
    assert res.verdict == FAIL and "music/fake (D) 1 atoms, 1 families" in res.detail


def test_g_eda3_reads_the_stricter_of_atoms_and_families_on_a_fake_side(manifest):
    """Many publisher atoms over one generator family is still one family."""
    m = manifest.copy()
    m.loc[m["pool"] == "B", "artifact_family"] = "one-family"
    assert g_eda3(m, floor=3).verdict == FAIL


# --------------------------------------------------------------------------- #
# G-EDA4


def _folds(manifest: pd.DataFrame) -> pd.DataFrame:
    f = manifest[["file_id", "pair_id", "dup_group"]].copy()
    f["slice"] = "train_val"
    f["fold"] = 0
    return f


def test_g_eda4_passes_when_every_set_sits_in_one_fold_and_trips_on_a_straddle(manifest):
    folds = _folds(manifest)
    assert g_eda4(folds).verdict == PASS
    pair = folds[folds["pair_id"].notna()]["pair_id"].iloc[0]
    rows = folds.index[folds["pair_id"] == pair]
    assert len(rows) >= 2
    folds.loc[rows[0], "fold"] = 1
    res = g_eda4(folds)
    assert res.verdict == FAIL and "pair_id: 1 of" in res.detail


def test_g_eda4_counts_a_probe_versus_fold_split_as_a_straddle(manifest):
    folds = _folds(manifest)
    pair = folds[folds["pair_id"].notna()]["pair_id"].iloc[0]
    rows = folds.index[folds["pair_id"] == pair]
    folds.loc[rows[0], ["slice", "fold"]] = ["probe", None]
    assert g_eda4(folds).verdict == FAIL


# --------------------------------------------------------------------------- #
# G-EDA7 and G4


def test_g_eda7_ledger_reads_labels_from_pool_and_cell_and_rates_are_per_role(manifest):
    files = _files(manifest)
    drop = files["pool"].eq("B")            # a filter that only ever drops fake voice
    ledger = g_eda7_ledger(_verdict(files, drop), files)
    voice = ledger[ledger["role"] == "voice"].set_index("file_fake")
    assert voice.at[True, "rate"] == 1.0 and voice.at[False, "rate"] == 0.0
    music = ledger[ledger["role"] == "music"]
    assert (music["rate"] == 0.0).all() and set(music["file_fake"]) == {True, False}
    assert "file" in set(ledger["role"]), "whole-file rows are a role of their own"
    labelled = files["pool"].isin(list("ABCDE")) | pd.to_numeric(
        files["cell"], errors="coerce").isin([5, 8])
    assert ledger["n"].sum() == labelled.sum()


def test_g_eda7_is_symmetric_at_equal_rates_and_fires_past_the_tolerance(manifest):
    files = _files(manifest)
    rng = np.random.default_rng(0)
    even = pd.Series(rng.random(len(files)) < 0.2)
    res, _ = g_eda7(_verdict(files, even), files)
    assert res.verdict == PASS
    skew = pd.Series(files["pool"].eq("B").values & (rng.random(len(files)) < 0.5))
    res, ledger = g_eda7(_verdict(files, skew), files)
    assert res.verdict == FAIL and "asymmetric: usable_duration" in res.detail
    assert set(ledger["filter"]) == {"usable_duration"}


def test_g4_table_keys_whole_file_rows_by_cell_and_flags_a_five_percent_drop(manifest):
    files = _files(manifest)
    drop = files["pool"].eq("A")
    g4 = g4_table(_verdict(files, drop), files)
    units = set(g4["unit"])
    assert {"A", "B"} <= units and any(u.startswith("cell") for u in units)
    a = g4[g4["unit"] == "A"].iloc[0]
    assert a["rate"] == 1.0 and a["escalates"] and a["hours_dropped"] == pytest.approx(a["hours"])
    assert not g4[g4["unit"] == "B"]["escalates"].any()


# --------------------------------------------------------------------------- #
# G7


def test_g7_pack_counts_reassignments_per_corpus_and_samples_examples(manifest):
    m = manifest.copy()
    files = _files(manifest)
    c_rows = m.index[m["pool"] == "C"][:6]
    m.loc[c_rows, "reassigned_from"] = "C"
    m.loc[c_rows, "cell"] = 5
    table, examples = g7_pack(m, files, n_examples=4)
    assert table["reassigned"].sum() == 6
    assert (table["cell"] == 5).all() and (table["reassigned_from"] == "C").all()
    total_c = files[files["pool"] == "C"].groupby("source_name").size()
    for _, row in table.iterrows():
        assert row["rate"] == pytest.approx(row["reassigned"] / total_c[row["corpus"]])
    assert len(examples) == 4 and set(examples["file_id"]) <= set(m.loc[c_rows, "file_id"])


def test_run_gates_aggregates_to_the_worst_verdict(manifest):
    files = _files(manifest)
    rng = np.random.default_rng(1)
    out = run_gates(manifest, _verdict(files, pd.Series(rng.random(len(files)) < 0.1)),
                    _folds(manifest), files, min_atoms=3)
    assert set(out["gates"]["gate"]) == {"G-EDA3", "G-EDA4", "G-EDA7"}
    assert out["aggregate"] == PASS
    for key in ("g_eda7_ledger", "g4_table", "g7_pack", "g7_examples"):
        assert isinstance(out[key], pd.DataFrame)
