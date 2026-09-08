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
    bad.loc[whole_idx, "cell"] = 5                      # implies (1, 1, 0, 0)
    bad.loc[whole_idx, "label_voice_present"] = 0
    bad.loc[whole_idx, "label_music_present"] = 0
    bad.loc[whole_idx, "label_voice_fake"] = 0
    bad.loc[whole_idx, "label_music_fake"] = 0
    with pytest.raises(ValueError, match="label_voice_present"):
        validate_manifest(bad)


# --------------------------------------------------------------------------- #
# the fake labels, which nothing checked

def test_rejects_a_whole_file_row_whose_fake_labels_disagree_with_its_cell(df):
    """Mutation: only the *presence* half was ever checked.

    The fake half is not inert. `folds._group_facts` builds the per-head
    artifact-family sets from `label_voice_fake` / `label_music_fake` and from
    nothing else, and `audit.py`'s I4 explicitly skips whole_file rows -- so a
    mislabelled one reached the fold builder unchallenged from both directions.
    """
    whole = df[df.row_kind == "whole_file"]
    fake_voice_idx = whole.index[whole.cell.isin([2, 8])][0]
    bad = df.copy()
    bad.loc[fake_voice_idx, "label_voice_fake"] = 0
    with pytest.raises(ValueError, match="label_voice_fake"):
        validate_manifest(bad)


def test_rejects_a_whole_file_row_carrying_an_absent_components_fake_label(df):
    """The null direction: cell 1 has no music, so its music_fake must be null.

    `None` in CELL_TABLE means "the metric ignores this label", and writing a 0
    there asserts something about a component that is not present.
    """
    whole = df[df.row_kind == "whole_file"]
    idx = whole.index[whole.cell == 1][0]
    bad = df.copy()
    bad.loc[idx, "label_music_fake"] = 0
    with pytest.raises(ValueError, match="label_music_fake"):
        validate_manifest(bad)


@pytest.mark.parametrize("pool,column,value", [
    ("B", "label_voice_fake", 0),      # a fake voice pool row claiming to be real
    ("A", "label_voice_fake", 1),      # and the reverse
    ("D", "label_music_fake", 0),
    ("A", "label_music_fake", 0),      # a label for a component that is absent
    ("E", "label_voice_present", 1),   # noise has neither component
])
def test_rejects_component_labels_that_disagree_with_the_pool(df, pool, column,
                                                              value):
    """A component's labels are fixed by its pool -- the only thing it has.

    Nothing checked them at all. Zeroing `label_voice_fake` across a manifest
    validated cleanly, took the voice artifact-family count 24 -> 0 in
    `folds._group_facts`, and `build_folds` went on to emit a full table.
    """
    idx = df.index[(df.row_kind == "component") & (df.pool == pool)][0]
    bad = df.copy()
    bad.loc[idx, column] = value
    with pytest.raises(ValueError, match=column):
        validate_manifest(bad)


def test_zeroing_a_fake_label_across_the_manifest_is_now_refused(df):
    """The whole-corpus version, which is what a real mislabelling looks like.

    One bad row is a typo; a column-wide default is a schema mistake, and it is
    the one that emptied the family sets while every check stayed green.
    """
    bad = df.copy()
    bad["label_voice_fake"] = 0
    with pytest.raises(ValueError, match="label_voice_fake"):
        validate_manifest(bad)


# --------------------------------------------------------------------------- #
# the two remaining value constraints, both already asserted in comments

def test_rejects_a_fake_component_without_a_domain_key(df):
    """Critical: a null domain_key does not opt a row out of DOSS, it renames it.

    `Sampler._doss_weights` does `domain_key.fillna("__real__")`, so every fake
    component missing one collapses into a single synthetic domain sharing one
    cap -- the DOSS failure (6.4k h naive 3.29% EER vs 0.2k h balanced 2.77%)
    with a manifest that validates.
    """
    fake_idx = df.index[(df.row_kind == "component") & df.pool.isin(["B", "D"])][0]
    bad = df.copy()
    bad.loc[fake_idx, "domain_key"] = None
    with pytest.raises(ValueError, match="domain_key"):
        validate_manifest(bad)


def test_the_null_domain_key_collapse_is_the_reason(df):
    """The mechanism the check exists for, measured rather than asserted.

    Without this, `test_rejects_a_fake_component_without_a_domain_key` would be
    a schema rule with no stated consequence.
    """
    import pandas as pd

    fake = df[(df.row_kind == "component") & (df.pool == "B")]
    assert fake.domain_key.nunique() >= 8, fake.domain_key.nunique()
    collapsed = fake.domain_key.where(pd.Series(False, index=fake.index))
    assert collapsed.fillna("__real__").nunique() == 1, (
        "a null domain_key resolves to one bucket, not to no bucket")


@pytest.mark.parametrize("kind", ["component", "whole_file"])
def test_rejects_a_real_row_carrying_an_artifact_family(df, kind):
    """`grouping_atoms` unions on artifact_family whatever the labels say.

    So a stray family on a real row binds that real file into a generator
    family's union-find group -- and therefore into its fold, decided before
    any label-aware code runs. Both row kinds: a real whole_file row has no
    pool, so its fake status comes from its cell.
    """
    if kind == "component":
        idx = df.index[(df.row_kind == "component") & df.pool.isin(["A", "C", "E"])][0]
    else:
        whole = df[df.row_kind == "whole_file"]
        idx = whole.index[whole.cell.isin([1, 3, 5, 9])][0]
    bad = df.copy()
    bad.loc[idx, "artifact_family"] = "hifigan"
    with pytest.raises(ValueError, match="artifact_family"):
        validate_manifest(bad)


def test_a_stray_family_really_would_move_the_real_file(df):
    """The consequence, measured on `grouping_atoms` with validation bypassed.

    A schema rule whose cost is only asserted in a docstring is the thing this
    repo keeps finding; this runs the union-find both ways.
    """
    from training.folds import grouping_atoms

    real_idx = df.index[(df.row_kind == "component") & (df.pool == "A")][0]
    fam = df.loc[df.pool == "B", "artifact_family"].iloc[0]
    fake_ids = set(df.loc[df.artifact_family == fam, "file_id"].astype(str))

    before = grouping_atoms(df)
    real_id = str(df.loc[real_idx, "file_id"])
    assert before[real_id] not in {before[f] for f in fake_ids}

    stray = df.copy()
    stray.loc[real_idx, "artifact_family"] = fam
    after = grouping_atoms(stray)
    assert after[real_id] in {after[f] for f in fake_ids}, (
        "the stray family must be what pulls the real file into the group")


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
