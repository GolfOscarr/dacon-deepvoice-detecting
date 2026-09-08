"""Proof that every VG1 assertion in `training.foldcheck` can actually fail.

The same rule as `tests/test_audit.py`: a green gate that cannot go red is
worse than no gate. A review of this repo found three audit checks that could
not fail and one printing PASS for something implemented nowhere. So every
assertion in `check_split_integrity` has a mutation test below that breaks the
constraint and requires the assertion to fire -- and the two that cannot be
computed on this table (A8/A9) are asserted to report SKIPPED, never PASS.

Caveat: under the rotating scheme every mutation has to be *isolated*. Moving
one row to another fold breaks every grouping key on it at once, so each test
nulls the four keys it is not testing. Otherwise five assertions fire together
and none of them is shown to be doing its own work.

The tables under test come from `training.folds.build_folds`, which is where a
real one comes from; `tests/test_folds.py` tests the builder itself.
"""

import pandas as pd
import pytest

from training.audit import AuditReport
from training.foldcheck import check_split_integrity, validate_folds
from training.folds import FoldConfig, apply_folds, build_folds
from training.sampler import Sampler
from training.synthetic import synthetic_manifest


#: 24 families and 8 real corpora per role. Caveat: deliberately wider than the
#: default corpus on **both** axes: under rotation the fold count is bounded by
#: the real-source count as well as the family count, and
#: `test_too_few_real_sources_is_fatal` is the test that says so.
WIDE = dict(n_per_pool=240, n_whole_file=200, seed=0, n_families=24, n_sources=8)
N_FOLDS = 5
OTHER_KEYS = {"artifact_family": None, "source_name": None,
              "speaker_ref_id": None, "pair_id": None, "dup_group": None}


@pytest.fixture(scope="module")
def manifest():
    return synthetic_manifest(**WIDE)


@pytest.fixture(scope="module")
def plan(manifest):
    return build_folds(manifest, FoldConfig(assigned_at="2026-01-01T00:00:00+00:00"))


@pytest.fixture(scope="module")
def folds(plan):
    return plan.frame


def _shadowed(folds, n=6):
    """Pair `n` rotating files to others *in the same fold*, as S-a re-renders.

    The manifest has no column saying "this is a telephone re-render", which is
    why `build_folds` takes the pairing as an argument; this builds one.
    """
    rot = folds[folds["slice"] == "train_val"]
    out = {}
    for k in sorted(rot["fold"].unique()):
        ids = rot.loc[rot["fold"] == k, "file_id"].tolist()
        for a, b in zip(ids[:2], ids[2:4]):
            out[a] = b
        if len(out) >= n:
            break
    return dict(list(out.items())[:n])


# --------------------------------------------------------------------------- #
# the reference split

def test_the_reference_split_passes_every_vg1_assertion(folds):
    report = check_split_integrity(folds, run_scheme_version="synthetic-v1")
    assert report.ok, str(report)

# --------------------------------------------------------------------------- #
# A8 / A9 live elsewhere, and really do

def test_a8_a9_report_skipped_and_never_pass(folds):
    """Critical: A skipped check must not read as a pass (`AuditReport.SKIP`)."""
    report = check_split_integrity(folds)
    assert "A8_A9_eval_size_floors" in report.skipped
    assert "SKIP  A8_A9" in str(report)
    assert "val_specs" in report.skipped["A8_A9_eval_size_floors"]


def test_a8_a9_are_implemented_where_they_belong(manifest, folds):
    """...and the module they point at really does implement them.

    Not a redirection to a check that exists nowhere: draw an eval stream and
    require `I7_eval_size_floors` to *fail* on a stream too small to meet the
    1,200-per-class floor. If it passed here, the pointer would be a lie.
    """
    from training.audit import run_audit

    sampler = Sampler(apply_folds(manifest, folds, fold=0), slice_="val")
    report = run_audit(sampler, n=800, manifest=manifest, eval_floors=True)
    passed, why = report.results["I7_eval_size_floors"]
    assert not passed and not why.startswith(AuditReport.SKIP), why
    assert "1200" in why.replace(",", "")

# --------------------------------------------------------------------------- #
# Critical: mutation tests -- every VG1 assertion must be able to fail

def _mutate(folds, file_id, **changes):
    out = folds.copy()
    for col, value in changes.items():
        out.loc[out.file_id == file_id, col] = value
    return out


def _isolated(folds, file_id, keep: str, **changes):
    """Move one row, nulling every grouping key except `keep`.

    Without this a single mutation trips all five assertions at once and none of
    them is shown to be doing its own work.
    """
    nulls = {k: v for k, v in OTHER_KEYS.items() if k != keep}
    return _mutate(folds, file_id, **nulls, **changes)


def _pick(folds, key, fold=None):
    """A rotating row whose `key` is shared with at least one other row."""
    rot = folds[(folds["slice"] == "train_val") & folds[key].notna()]
    if fold is not None:
        rot = rot[rot["fold"] == fold]
    sizes = folds.groupby(key).size()
    shared = rot[rot[key].isin(set(sizes[sizes > 1].index))]
    assert len(shared), f"no shared {key} to mutate"
    return shared.iloc[0]


def test_A1_catches_a_family_validated_in_two_folds(folds):
    """A family in two folds is VAL in one and TRAIN in the other -- so in each
    of those folds it is on both sides at once."""
    victim = _pick(folds, "artifact_family")
    other = (int(victim.fold) + 1) % N_FOLDS
    broken = _isolated(folds, victim.file_id, "artifact_family", fold=other)
    report = check_split_integrity(broken)
    assert "A1_family_in_one_cell" in report.failures
    assert victim.artifact_family in report.failures["A1_family_in_one_cell"]
    assert not (report.failures.keys() - {"A1_family_in_one_cell"})


def test_A1_catches_a_family_that_is_both_sealed_and_rotating(folds):
    victim = folds[folds["slice"] == "probe"].iloc[0]
    rotating_family = _pick(folds, "artifact_family").artifact_family
    broken = _mutate(folds, victim.file_id, artifact_family=rotating_family)
    assert "A1_family_in_one_cell" in check_split_integrity(broken).failures


def test_A2_catches_a_source_on_both_sides_of_a_fold(folds):
    """No MUSDB18 track or LibriTTS speaker on both sides."""
    victim = _pick(folds, "source_name")
    other = (int(victim.fold) + 1) % N_FOLDS
    broken = _isolated(folds, victim.file_id, "source_name", fold=other)
    report = check_split_integrity(broken)
    assert "A2_source_disjoint" in report.failures
    assert victim.source_name in report.failures["A2_source_disjoint"]
    assert "A1_family_in_one_cell" not in report.failures


def test_A3_catches_a_speaker_reappearing_under_another_source(folds):
    """Critical: the falsifiable form. Keyed on (source, speaker) instead, A3 would be
    entailed by A2 and could never fail on its own.

    The real leak is a LibriTTS speaker reappearing under a resynthesis corpus.
    """
    victim = _pick(folds, "speaker_ref_id")
    elsewhere = folds[(folds["slice"] == "train_val")
                      & (folds["fold"] != victim.fold)
                      & folds.speaker_ref_id.notna()].iloc[0]
    broken = _isolated(folds, elsewhere.file_id, "speaker_ref_id",
                       speaker_ref_id=victim.speaker_ref_id)
    report = check_split_integrity(broken)
    assert "A3_speaker_disjoint" in report.failures
    assert victim.speaker_ref_id in report.failures["A3_speaker_disjoint"]
    assert "A2_source_disjoint" not in report.failures


def test_A4_catches_a_split_twin_pair(folds):
    """A real file and its resynthesized twin must land in the same fold."""
    victim = _pick(folds, "pair_id")
    other = (int(victim.fold) + 1) % N_FOLDS
    broken = _isolated(folds, victim.file_id, "pair_id", fold=other)
    report = check_split_integrity(broken)
    assert "A4_pair_shares_a_cell" in report.failures
    assert victim.pair_id in report.failures["A4_pair_shares_a_cell"]
    assert not (report.failures.keys() - {"A4_pair_shares_a_cell"})


def test_A5_catches_a_dup_group_spanning_folds(folds):
    # Caveat: built, not found. The synthetic corpus's dup groups happen to be
    # mostly singletons, and a dup_group with one member cannot span anything
    # -- picking whatever the generator emitted would have made this test
    # unable to fail. The assertion is about a *manifest* property, so the test
    # states it.
    rot = folds[folds["slice"] == "train_val"]
    a = rot[rot["fold"] == 0].iloc[0]
    b = rot[rot["fold"] == 1].iloc[0]
    broken = _isolated(folds, a.file_id, "dup_group", dup_group="near_dupes")
    broken = _isolated(broken, b.file_id, "dup_group", dup_group="near_dupes")
    report = check_split_integrity(broken)
    assert "A5_dup_group_in_one_cell" in report.failures
    assert "near_dupes" in report.failures["A5_dup_group_in_one_cell"]
    assert not (report.failures.keys() - {"A5_dup_group_in_one_cell"})


def test_A6_catches_an_empty_probe_that_A1_would_pass_vacuously(folds):
    """Critical: the clause A1 does not carry, and the main thing A6 carries under
    rotation: a builder that sealed nothing leaves A1 green over an empty slice."""
    broken = folds.copy()
    sealed = broken["slice"] == "probe"
    broken.loc[sealed, "slice"] = "train_val"
    broken.loc[sealed, "fold"] = 0
    report = check_split_integrity(broken)
    assert "A6_probe_sealed" in report.failures
    assert "PROBE IS EMPTY" in report.failures["A6_probe_sealed"]
    assert "A1_family_in_one_cell" not in report.failures


def test_A7_catches_a_shadow_row_paired_across_folds(manifest, folds):
    """The VAL->SHADOW drop is attributable to the channel only if the pair is
    real *and* in this fold. A re-render of a file that is TRAIN here measures
    a training generator."""
    out = build_folds(manifest, FoldConfig(assigned_at="x"),
                      shadow_of=_shadowed(folds)).frame
    victim = out[out.shadow_kind == "a"].iloc[0]
    elsewhere = out[(out["slice"] == "train_val")
                    & (out["fold"] != victim.fold)].iloc[0]
    broken = _mutate(out, victim.file_id, shadow_of=elsewhere.file_id)
    report = check_split_integrity(broken)
    assert "A7_shadow_a_points_at_its_fold" in report.failures
    assert elsewhere.file_id in report.failures["A7_shadow_a_points_at_its_fold"]


def test_A7_catches_a_dangling_shadow_of(manifest, folds):
    out = build_folds(manifest, FoldConfig(assigned_at="x"),
                      shadow_of=_shadowed(folds)).frame
    victim = out[out.shadow_kind == "a"].iloc[0]
    broken = _mutate(out, victim.file_id, shadow_of="not_a_real_file")
    report = check_split_integrity(broken)
    assert "A7_shadow_a_points_at_its_fold" in report.failures


def test_A7_is_skipped_rather_than_passed_when_there_are_no_shadow_rows(folds):
    assert "A7_shadow_a_points_at_its_fold" in check_split_integrity(folds).skipped


def test_A10_catches_a_scheme_version_the_run_did_not_use(folds):
    """Results across scheme versions are never compared."""
    assert "A10_scheme_version_matches" in check_split_integrity(
        folds, run_scheme_version="v2").failures
    assert check_split_integrity(folds, run_scheme_version="synthetic-v1").ok


def test_A10_is_skipped_rather_than_passed_without_a_run_version(folds):
    assert "A10_scheme_version_matches" in check_split_integrity(folds).skipped


def test_raise_for_status_fires_on_a_broken_split(folds):
    broken = folds.copy()
    sealed = broken["slice"] == "probe"
    broken.loc[sealed, "slice"] = "train_val"
    broken.loc[sealed, "fold"] = 0
    with pytest.raises(AssertionError, match="A6_probe_sealed"):
        check_split_integrity(broken).raise_for_status()

# --------------------------------------------------------------------------- #
# schema

@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.drop(columns=["dup_group"]), "missing column"),
    (lambda d: d.assign(**{"slice": "holdout"}), "unknown slice"),
    (lambda d: d.assign(shadow_kind="c"), "shadow_kind must be"),
    (lambda d: d.assign(scheme_version=["v1"] + ["v2"] * (len(d) - 1)),
     "mixes scheme_version"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "duplicate file_id"),
])
def test_validate_folds_rejects_a_malformed_table(folds, mutate, match):
    with pytest.raises(ValueError, match=match):
        validate_folds(mutate(folds))


def test_validate_folds_rejects_a_fold_on_a_probe_row(folds):
    broken = folds.copy()
    broken.loc[broken.index[broken["slice"] == "probe"][0], "fold"] = 2
    with pytest.raises(ValueError, match="fold must be null on probe"):
        validate_folds(broken)


def test_validate_folds_rejects_a_rotating_row_without_a_fold(folds):
    broken = folds.copy()
    broken.loc[broken.index[broken["slice"] == "train_val"][0], "fold"] = pd.NA
    with pytest.raises(ValueError, match="needs a fold"):
        validate_folds(broken)


def test_validate_folds_rejects_a_gap_in_the_fold_numbering(folds):
    """A fold nobody validates on is a silently smaller k."""
    broken = folds.copy()
    broken.loc[broken["fold"] == 3, "fold"] = 9
    with pytest.raises(ValueError, match="folds must be 0..k-1"):
        validate_folds(broken)


def test_validate_folds_rejects_a_cell_on_a_component_row(folds):
    broken = folds.copy()
    broken.loc[broken.index[broken.row_kind == "component"][0], "cell"] = 5
    with pytest.raises(ValueError, match="cell must be non-null exactly"):
        validate_folds(broken)


def test_the_manifest_and_fold_slice_vocabularies_are_not_the_same_thing():
    """Critical: two constants, two vocabularies, and they must not converge.

    `MANIFEST_SLICES` is the materialised view a sampler draws from -- `val` is
    a real value there. `FOLD_SLICES` is the stored column in `folds.parquet`,
    where the same rows are `train_val` and `val` does not exist until
    `apply_folds` resolves it against a fold number.

    Both were named `SLICES` and both were exported. `training/folds.py`
    imported one and then redeclared the other four lines later, shadowing the
    import, so its own docstring's claim to re-export the constant was false and
    the two could drift apart without a single caller changing. Renaming is the
    fix; this asserts the distinction the names now carry, because a later
    "tidy-up" that unified them would silently make `val` a legal fold slice.
    """
    from training.foldcheck import FOLD_SLICES
    from training.manifest import MANIFEST_SLICES

    assert "val" in MANIFEST_SLICES and "val" not in FOLD_SLICES
    assert "train_val" in FOLD_SLICES and "train_val" not in MANIFEST_SLICES
    assert set(MANIFEST_SLICES) != set(FOLD_SLICES)
