"""The manifest schema, and the two row kinds it exists to keep apart."""

import pandas as pd
import pytest

from training.manifest import REQUIRED_COLUMNS, load_manifest, validate_manifest
from training.synthetic import synthetic_manifest


@pytest.fixture(scope="module")
def df():
    return synthetic_manifest(n_per_pool=60, n_whole_file=40, seed=0)


def test_synthetic_manifest_validates(df):
    assert set(REQUIRED_COLUMNS) <= set(df.columns)
    assert len(df) == 60 * 5 + 40


def test_component_rows_have_a_pool_and_no_cell(df):
    """🔴 A component has no cell until the sampler composes one."""
    comp = df[df.row_kind == "component"]
    assert comp.pool.notna().all()
    assert comp.cell.isna().all()


def test_whole_file_rows_have_a_cell_and_no_pool(df):
    whole = df[df.row_kind == "whole_file"]
    assert whole.cell.notna().all()
    assert whole.pool.isna().all()


def test_cells_6_and_7_never_appear_as_whole_files(df):
    """They cannot be scraped -- the entire reason two fake heads exist.

    🔴 Asserted against VALIDATION, not just against the generator. The earlier
    version only checked `synthetic.py`'s own `whole_cells` tuple, so a real
    manifest carrying a cell-6 whole-file row would validate, the fold table
    would be built from it, and the failure would surface only inside
    `SampleSpec.__post_init__` at draw time -- long after the split was frozen.
    """
    assert not df.cell.isin([6, 7]).any(), "the generator must not emit them"

    whole_idx = df.index[df.row_kind == "whole_file"][0]
    for cell in (6, 7):
        bad = df.copy()
        bad.loc[whole_idx, "cell"] = cell
        bad.loc[whole_idx, "label_voice_present"] = 1
        bad.loc[whole_idx, "label_music_present"] = 1
        with pytest.raises(ValueError, match="cannot be whole_file"):
            validate_manifest(bad)


def test_rejects_a_whole_file_row_carrying_a_pool(df):
    """Validation kept the row kinds apart in only one direction.

    It required component rows to carry a pool, but never checked that
    whole_file rows do not -- so half the contract was unenforced.
    """
    whole_idx = df.index[df.row_kind == "whole_file"][0]
    bad = df.copy()
    bad.loc[whole_idx, "pool"] = "A"
    with pytest.raises(ValueError, match="pool = null"):
        validate_manifest(bad)


def test_fake_components_carry_the_split_and_capping_keys(df):
    fake = df[(df.row_kind == "component") & df.pool.isin(["B", "D"])]
    assert fake.artifact_family.notna().all(), "artifact_family is the split key"
    assert fake.domain_key.notna().all(), "domain_key is the DOSS capping key"


def test_real_components_have_no_artifact_family(df):
    """Family disjointness constrains fake components only; real files have none."""
    real = df[(df.row_kind == "component") & df.pool.isin(["A", "C", "E"])]
    assert real.artifact_family.isna().all()


def test_the_synthetic_corpus_is_imbalanced_on_purpose(df):
    """A balanced generator would make the DOSS audit vacuous."""
    counts = df.domain_key.value_counts()
    assert counts.max() >= 2 * counts.min(), "domains should be unevenly sized"


# --------------------------------------------------------------------------- #
# what validation must reject

def _broken(df, **assign):
    bad = df.copy()
    for column, value in assign.items():
        bad.loc[bad.index[0], column] = value
    return bad


def test_rejects_a_component_row_carrying_a_cell(df):
    comp_idx = df.index[df.row_kind == "component"][0]
    bad = df.copy()
    bad.loc[comp_idx, "cell"] = 5
    with pytest.raises(ValueError, match="cell = null"):
        validate_manifest(bad)


def test_rejects_a_whole_file_row_without_a_cell(df):
    whole_idx = df.index[df.row_kind == "whole_file"][0]
    bad = df.copy()
    bad.loc[whole_idx, "cell"] = None
    with pytest.raises(ValueError, match="must carry a cell"):
        validate_manifest(bad)


def test_rejects_labels_that_disagree_with_the_cell(df):
    """Two sources of truth for a label is one too many."""
    whole_idx = df.index[df.row_kind == "whole_file"][0]
    bad = df.copy()
    bad.loc[whole_idx, "cell"] = 5                      # implies presence (1, 1)
    bad.loc[whole_idx, "label_voice_present"] = 0
    bad.loc[whole_idx, "label_music_present"] = 0
    with pytest.raises(ValueError, match="implies presence"):
        validate_manifest(bad)


def test_rejects_a_fake_component_without_artifact_family(df):
    fake_idx = df.index[(df.row_kind == "component") & df.pool.isin(["B", "D"])][0]
    bad = df.copy()
    bad.loc[fake_idx, "artifact_family"] = None
    with pytest.raises(ValueError, match="artifact_family"):
        validate_manifest(bad)


def test_rejects_mixed_scheme_versions(df):
    """Results across scheme versions are never compared."""
    bad = _broken(df, scheme_version="other")
    with pytest.raises(ValueError, match="scheme_version"):
        validate_manifest(bad)


@pytest.mark.parametrize("column,value,match", [
    ("row_kind", "chunk", "row_kind"),
    ("slice", "holdout", "slice"),
    ("label_confidence", "guessed", "label_confidence"),
    ("duration_s", 0.0, "duration_s"),
])
def test_rejects_out_of_domain_values(df, column, value, match):
    with pytest.raises(ValueError, match=match):
        validate_manifest(_broken(df, **{column: value}))


def test_rejects_duplicate_file_ids(df):
    bad = pd.concat([df, df.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate file_id"):
        validate_manifest(bad)


def test_rejects_a_missing_column(df):
    with pytest.raises(ValueError, match="missing column"):
        validate_manifest(df.drop(columns=["domain_key"]))


def test_round_trips_through_parquet(df, tmp_path):
    """Nullable Int64 columns must survive; the fold table depends on it."""
    path = tmp_path / "manifest.parquet"
    df.to_parquet(path)
    back = load_manifest(path)
    assert back.cell.dtype == df.cell.dtype == "Int64"
    assert len(back) == len(df)
