"""The manifest schema, and the two row kinds it exists to keep apart."""

import numpy as np
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


def test_the_generator_emits_component_rows_with_a_pool_and_no_cell(df):
    """🔴 A component has no cell until the sampler composes one.

    ⚠️ This is a statement about `synthetic.py`, not about `validate_manifest`.
    The *enforcement* is asserted separately, once per direction, in
    `test_rejects_a_component_row_carrying_a_cell` and
    `test_rejects_a_component_row_without_a_pool` -- a generator that happens to
    produce well-formed rows proves nothing about what validation would reject.
    """
    comp = df[df.row_kind == "component"]
    assert comp.pool.notna().all()
    assert comp.cell.isna().all()


def test_the_generator_emits_whole_file_rows_with_a_cell_and_no_pool(df):
    """The mirror of the above, and equally a generator test.

    Enforcement lives in `test_rejects_a_whole_file_row_without_a_cell` and
    `test_rejects_a_whole_file_row_carrying_a_pool`.
    """
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
    """A balanced generator would make the DOSS audit vacuous.

    🔴 Measured *per fake pool* and against the UNIFORM share, not against the
    smallest domain. `max >= 2 * min` is met by almost any random assignment and
    says nothing about there being a head to flatten; the generator draws family
    sizes from a Zipf, which puts ~2.3x the uniform share on the top family and
    costs ~1.3 of the 8 available effective domains.

    ⚠️ This asserts the corpus has an imbalance. Whether that imbalance is large
    enough for the *shipped* `domain_cap` to bind is a different question, and it
    is asserted where it belongs, on a production-sized corpus:
    `tests/test_sampler.py::test_doss_flattens_over_represented_domains`.
    """
    for pool in ("B", "D"):
        counts = df.loc[(df.row_kind == "component") & (df.pool == pool),
                        "domain_key"].value_counts()
        n_domains = len(counts)
        assert n_domains >= 8, f"{pool}: {n_domains} domains is too few to be a corpus"

        share = counts / counts.sum()
        uniform = 1.0 / n_domains
        assert share.max() >= 2.0 * uniform, (
            f"{pool}: top domain holds {share.max():.3f}, barely above the uniform "
            f"{uniform:.3f} -- nothing for DOSS to flatten")
        assert counts.max() >= 5 * counts.min(), (
            f"{pool}: head/tail ratio {counts.max() / counts.min():.1f} is too flat")

        n_eff = float(np.exp(-(share * np.log(share)).sum()))
        assert n_eff <= 0.875 * n_domains, (
            f"{pool}: {n_eff:.2f} effective domains out of {n_domains} is near-uniform")


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


def test_rejects_a_component_row_without_a_pool(df):
    """The fourth direction: a component row must carry the pool it came from."""
    comp_idx = df.index[df.row_kind == "component"][0]
    bad = df.copy()
    bad.loc[comp_idx, "pool"] = None
    with pytest.raises(ValueError, match="must carry a pool"):
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
