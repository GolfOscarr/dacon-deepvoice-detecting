"""The fold builder, and proof that every VG1 assertion can actually fail.

🔴 The same rule as `tests/test_audit.py`: a green gate that cannot go red is
worse than no gate. A review of this repo found three audit checks that could
not fail and one printing PASS for something implemented nowhere. So every
assertion in `check_split_integrity` has a mutation test below that breaks the
constraint and requires the assertion to fire -- and the two that cannot be
computed on this table (A8/A9) are asserted to report SKIPPED, never PASS.
"""

import pandas as pd
import pytest

from training.audit import AuditReport
from training.folds import (FOLD_COLUMNS, PARTITIONED_SLICES, FoldConfig,
                            FoldInfeasible, apply_folds, build_folds,
                            check_split_integrity, grouping_atoms, load_folds,
                            validate_folds)
from training.sampler import Sampler
from training.synthetic import synthetic_manifest

#: Wide enough that a 5-fold has >=1 family per validation fold on both heads.
#: ⚠️ Deliberately not the default corpus: at eight families per pool a 5-fold is
#: infeasible, and `test_too_few_families_is_fatal` is the test that says so.
WIDE = dict(n_per_pool=200, n_whole_file=200, seed=0, n_families=24)


@pytest.fixture(scope="module")
def manifest():
    return synthetic_manifest(**WIDE)


@pytest.fixture(scope="module")
def plan(manifest):
    return build_folds(manifest, FoldConfig(assigned_at="2026-01-01T00:00:00+00:00"))


@pytest.fixture(scope="module")
def folds(plan):
    return plan.frame


def _shadowed(manifest, folds, n=6):
    """Turn `n` VAL rows into S-a re-renders of other VAL rows, in place.

    The manifest has no column saying "this is a telephone re-render", which is
    why `build_folds` takes the pairing as an argument; this builds one.
    """
    val = folds.loc[folds["slice"] == "val", "file_id"].tolist()
    return dict(zip(val[:n], val[n:2 * n]))


# --------------------------------------------------------------------------- #
# the table itself

def test_emitted_columns_are_exactly_the_spec(folds):
    """docs/validation/01 §3 lists the columns; drift there is silent."""
    assert list(folds.columns) == list(FOLD_COLUMNS)


def test_every_manifest_row_is_assigned_exactly_once(manifest, folds):
    assert len(folds) == len(manifest)
    assert set(folds.file_id) == set(manifest.file_id.astype(str))
    assert not folds.file_id.duplicated().any()
    assert set(folds["slice"]) <= {"train", "val", "shadow", "probe"}


def test_the_reference_split_passes_every_vg1_assertion(folds):
    report = check_split_integrity(folds, run_scheme_version="synthetic-v1")
    assert report.ok, str(report)


def test_a8_a9_report_skipped_and_never_pass(folds):
    """🔴 A skipped check must not read as a pass (`AuditReport.SKIP`).

    A8/A9 are statements about compositions and are evaluated against
    `val_specs.parquet`; a component row has no cell (docs/validation/04 §VG1).
    """
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

    sampler = Sampler(apply_folds(manifest, folds), slice_="val")
    report = run_audit(sampler, n=800, manifest=manifest, eval_floors=True)
    passed, why = report.results["I7_eval_size_floors"]
    assert not passed and not why.startswith(AuditReport.SKIP), why
    assert "1200" in why.replace(",", "") or "need 1200" in why.replace(",", "")


def test_family_shares_land_near_the_documented_targets(folds):
    """docs/validation/01 §2: ~65% / ~25% / ~10% of artifact families."""
    fam = {s: set(folds.loc[folds["slice"] == s, "artifact_family"].dropna())
           for s in PARTITIONED_SLICES}
    total = sum(len(v) for v in fam.values())
    assert total >= 40
    for s, want in (("train", 0.65), ("val", 0.25), ("probe", 0.10)):
        assert abs(len(fam[s]) / total - want) < 0.08, {k: len(v) for k, v in fam.items()}


def test_fold_is_null_outside_val_and_shadow(folds):
    """A TRAIN row is used by every fold, so a fold id on it would invite
    training on a fifth of the corpus. See the module docstring."""
    assert folds.loc[folds["slice"].isin(["train", "probe"]), "fold"].isna().all()
    assert folds.loc[folds["slice"] == "val", "fold"].notna().all()


def test_every_validation_fold_is_used_and_family_disjoint(folds):
    val = folds[folds["slice"] == "val"]
    assert sorted(val["fold"].unique()) == [0, 1, 2, 3, 4]
    by_fold = val.groupby("fold")["artifact_family"].agg(lambda s: set(s.dropna()))
    for a in range(5):
        for b in range(a + 1, 5):
            assert not (by_fold[a] & by_fold[b]), f"folds {a}/{b} share a family"


def test_the_builder_ignores_any_slice_already_on_the_manifest(manifest):
    """🔴 `build_folds` is the *source* of the assignment. Reading a stale one
    back would let a bad split reproduce itself forever."""
    poisoned = manifest.copy()
    poisoned["slice"] = "probe"
    poisoned["fold"] = pd.array([3] * len(poisoned), dtype="Int64")
    cfg = FoldConfig(assigned_at="x")
    assert build_folds(poisoned, cfg).frame.equals(build_folds(manifest, cfg).frame)


def test_the_split_is_deterministic(manifest):
    cfg = FoldConfig(assigned_at="x")
    assert build_folds(manifest, cfg).frame.equals(build_folds(manifest, cfg).frame)


def test_round_trip_through_parquet(plan, tmp_path):
    path = plan.write(tmp_path / "folds.parquet")
    back = load_folds(path)
    assert back.equals(plan.frame)
    # The caveats travel with the table rather than with whoever remembers them.
    side = (tmp_path / "folds.caveats.txt").read_text()
    assert side.count("\n") == len(plan.caveats)


# --------------------------------------------------------------------------- #
# the split is usable

def test_the_sampler_can_draw_from_every_slice(manifest, folds):
    """A slice with no real voice components cannot compose cells 1/5/6, and the
    sampler only discovers that at draw time -- long after the split is frozen."""
    joined = apply_folds(manifest, folds)
    for slice_ in PARTITIONED_SLICES:
        specs = list(Sampler(joined, slice_=slice_).epoch_specs(200))
        assert len(specs) == 200
        assert {s.file_fake for s in specs} == {0, 1}


def test_drawn_files_never_leave_their_slice(manifest, folds):
    joined = apply_folds(manifest, folds)
    allowed = set(joined.loc[joined["slice"] == "val", "file_id"].astype(str))
    for spec in Sampler(joined, slice_="val").epoch_specs(400):
        for c in spec.components:
            assert c.file_id in allowed


def test_apply_folds_refuses_a_mismatched_id_set(manifest, folds):
    with pytest.raises(ValueError, match="disagree on"):
        apply_folds(manifest, folds.iloc[:-1])


def test_apply_folds_refuses_a_scheme_version_mismatch(manifest, folds):
    with pytest.raises(ValueError, match="scheme_version mismatch"):
        apply_folds(manifest, folds.assign(scheme_version="v2"))


# --------------------------------------------------------------------------- #
# 🔴 grouping: the transitive closure, not five independent checks

def test_a_pair_id_binds_its_whole_source_and_family_together(manifest):
    """docs/validation/01 §1's five keys are all equivalence constraints, so a
    LibriTTS/HiFi-GAN twin binds the whole corpus to the whole family."""
    atoms = grouping_atoms(manifest)
    m = manifest.set_index(manifest.file_id.astype(str))
    twin = m[m.pair_id.notna() & (m.pool == "B")].iloc[0]
    partner = m[(m.pair_id == twin.pair_id) & (m.pool == "A")].iloc[0]
    assert atoms[twin.name] == atoms[partner.name]
    # ...and transitively, every row of both the family and the source.
    same = {atoms[f] for f in m[(m.artifact_family == twin.artifact_family)
                                | (m.source_name == partner.source_name)].index}
    assert same == {atoms[twin.name]}


def test_a_corpus_whose_twins_fuse_everything_is_reported_not_split(manifest):
    """🔴 The failure mode `training.synthetic._TWIN_COMBINATIONS` documents.

    Twinning every family to every real corpus collapses the grouping keys into
    one inseparable atom. The builder must say so, not quietly drop a key.
    """
    fused = manifest.copy()
    voice = fused.pool.isin(["A", "B"])
    fused.loc[voice, "pair_id"] = "everything_is_one_twin"
    with pytest.raises(FoldInfeasible) as e:
        build_folds(fused, FoldConfig(assigned_at="x"))
    assert "cannot fill 5" in str(e.value) or "no real" in str(e.value)


# --------------------------------------------------------------------------- #
# 🔴 failing loudly

def test_too_few_families_is_fatal_and_names_the_music_head(manifest):
    """docs/validation/01 §3: at 5 music families a 5-fold puts one family in
    each validation fold and PROBE cannot be carved out at all."""
    narrow = synthetic_manifest(n_per_pool=200, n_whole_file=200, seed=0)
    with pytest.raises(FoldInfeasible) as e:
        build_folds(narrow, FoldConfig(assigned_at="x"))
    assert "cannot fill 5 family-disjoint validation folds" in str(e.value)
    assert "docs/validation/01 §3" in str(e.value)
    # ...and it is not a lower VAL share away from passing.
    assert "do not lower the VAL share" in str(e.value)


def test_a_thin_head_gets_the_variance_caveat_not_silence(plan):
    """The music-head caveat travels with the table (docs/validation/01 §3)."""
    assert plan.caveats, "a 6-family VAL head over 5 folds must not be silent"
    music = [c for c in plan.caveats if c.startswith("music head")]
    assert music and "per validation fold" in music[0]
    assert "very high variance" in music[0]
    assert "CAVEAT" in str(plan)


def test_no_caveat_is_emitted_when_the_head_is_wide_enough(manifest):
    """⚠️ The caveat must be falsifiable too: a check that always fires is noise.

    Three folds over the same 6 VAL families is 2.0 per fold, which is the
    threshold, so the variance caveat must NOT appear.
    """
    plan = build_folds(manifest, FoldConfig(n_folds=3, assigned_at="x"))
    assert not [c for c in plan.caveats if "per validation fold" in c], plan.caveats


def test_an_empty_probe_must_be_asked_for(manifest):
    trimmed = manifest.copy()
    with pytest.raises(FoldInfeasible, match="allow_no_probe"):
        build_folds(trimmed, FoldConfig(shares={"train": 0.75, "val": 0.25,
                                                "probe": 0.0}, assigned_at="x"))
    plan = build_folds(trimmed, FoldConfig(shares={"train": 0.75, "val": 0.25,
                                                   "probe": 0.0},
                                           allow_no_probe=True,
                                           require_component_coverage=False,
                                           assigned_at="x"))
    assert any("PROBE" in c for c in plan.caveats)
    # 🔴 ...and VG1 A6 goes red for it. An empty sealed slice is not a pass.
    assert "A6_probe_sealed" in check_split_integrity(plan.frame).failures


def test_a_slice_the_sampler_cannot_draw_from_is_fatal(manifest):
    """Only one real voice source cannot cover three slices."""
    thin = manifest[~((manifest.pool == "A")
                      & (manifest.source_name != "libritts"))].copy()
    with pytest.raises(FoldInfeasible, match="cannot draw from"):
        build_folds(thin, FoldConfig(assigned_at="x"))


def test_shadow_rows_must_be_carved_out_of_val(manifest, folds):
    train_id = folds.loc[folds["slice"] == "train", "file_id"].iloc[0]
    val_id = folds.loc[folds["slice"] == "val", "file_id"].iloc[0]
    cfg = FoldConfig(assigned_at="x")
    with pytest.raises(FoldInfeasible, match="paired to a VAL file"):
        build_folds(manifest, cfg, shadow_of={val_id: train_id})
    with pytest.raises(FoldInfeasible, match="carved out of VAL"):
        build_folds(manifest, cfg, shadow_b=[train_id])
    with pytest.raises(FoldInfeasible, match="not in the manifest"):
        build_folds(manifest, cfg, shadow_b=["no_such_file"])


def test_an_impossible_configuration_is_rejected_at_construction():
    with pytest.raises(ValueError, match="must sum to 1"):
        FoldConfig(shares={"train": 0.6, "val": 0.2, "probe": 0.1})
    with pytest.raises(ValueError, match="nothing to validate on"):
        FoldConfig(shares={"train": 0.9, "val": 0.0, "probe": 0.1})
    with pytest.raises(ValueError, match="n_folds"):
        FoldConfig(n_folds=1)


# --------------------------------------------------------------------------- #
# SHADOW

def test_shadow_rows_carry_their_val_parents_fold(manifest, folds):
    plan = build_folds(manifest, FoldConfig(assigned_at="x"),
                       shadow_of=_shadowed(manifest, folds))
    out = plan.frame
    sa = out[out.shadow_kind == "a"]
    assert len(sa) == 6
    assert (sa["slice"] == "shadow").all()
    parent_fold = out.set_index("file_id")["fold"]
    for row in sa.itertuples():
        assert row.fold == parent_fold[row.shadow_of]
    assert check_split_integrity(out).ok, str(check_split_integrity(out))


def test_shadow_consumes_no_family_budget(manifest, folds):
    """docs/validation/01 §2: SHADOW is a condition axis, not a generator axis."""
    plan = build_folds(manifest, FoldConfig(assigned_at="x"),
                       shadow_of=_shadowed(manifest, folds))
    train_families = set(plan.frame.loc[plan.frame["slice"] == "train",
                                        "artifact_family"].dropna())
    base = set(folds.loc[folds["slice"] == "train", "artifact_family"].dropna())
    assert train_families == base


def test_s_b_rows_are_unpaired(manifest, folds):
    ids = folds.loc[folds["slice"] == "val", "file_id"].tolist()[:4]
    out = build_folds(manifest, FoldConfig(assigned_at="x"), shadow_b=ids).frame
    sb = out[out.shadow_kind == "b"]
    assert len(sb) == 4 and sb["shadow_of"].isna().all()


# --------------------------------------------------------------------------- #
# 🔴 mutation tests -- every VG1 assertion must be able to fail

def _mutate(folds, file_id, **changes):
    out = folds.copy()
    for col, value in changes.items():
        out.loc[out.file_id == file_id, col] = value
    return out


def _first(folds, slice_, **where):
    sub = folds[folds["slice"] == slice_]
    for col, value in where.items():
        sub = sub[sub[col] == value] if value is not None else sub[sub[col].notna()]
    return sub.iloc[0]


def test_A1_catches_a_family_spanning_train_and_val(folds):
    """The whole point of the split: hold out generators, not generator names."""
    victim = _first(folds, "train", artifact_family=None)
    broken = _mutate(folds, victim.file_id, **{"slice": "val", "fold": 0})
    report = check_split_integrity(broken)
    assert "A1_family_in_one_slice" in report.failures
    assert victim.artifact_family in report.failures["A1_family_in_one_slice"]


def test_A2_catches_a_source_spanning_train_and_val(folds):
    """No MUSDB18 track or LibriTTS speaker on both sides."""
    victim = _first(folds, "train", artifact_family=None)
    # Move the row to VAL but strip the family, so A1 stays green and only the
    # source-level statement fails -- A2 must not be carried by A1.
    broken = _mutate(folds, victim.file_id,
                     **{"slice": "val", "fold": 0, "artifact_family": None})
    report = check_split_integrity(broken)
    assert "A2_source_disjoint" in report.failures
    assert "A1_family_in_one_slice" not in report.failures


def test_A3_catches_a_speaker_reappearing_under_another_source(folds):
    """🔴 The falsifiable form. Keyed on (source, speaker) instead, A3 would be
    entailed by A2 and could never fail on its own -- see `check_split_integrity`.

    The real leak is a LibriTTS speaker reappearing under a resynthesis corpus.
    """
    speaker = _first(folds, "train", speaker_ref_id=None).speaker_ref_id
    victim = _first(folds, "val", artifact_family=None)
    broken = _mutate(folds, victim.file_id, speaker_ref_id=speaker,
                     source_name="a_different_corpus", artifact_family=None)
    report = check_split_integrity(broken)
    assert "A3_speaker_disjoint" in report.failures
    assert speaker in report.failures["A3_speaker_disjoint"]
    assert "A2_source_disjoint" not in report.failures


def test_A4_catches_a_split_twin_pair(folds):
    """A real file and its resynthesized twin must land in the same slice."""
    victim = _first(folds, "train", pair_id=None)
    broken = _mutate(folds, victim.file_id,
                     **{"slice": "probe", "artifact_family": None,
                        "source_name": "elsewhere", "speaker_ref_id": None})
    report = check_split_integrity(broken)
    assert "A4_pair_shares_a_slice" in report.failures
    assert victim.pair_id in report.failures["A4_pair_shares_a_slice"]


def test_A5_catches_a_dup_group_spanning_slices(folds):
    # ⚠️ A dup_group with a single member cannot span anything, so picking the
    # first one would have made this test unable to fail.
    sizes = folds.groupby("dup_group").size()
    shared = set(sizes[sizes > 1].index)
    train = folds[(folds["slice"] == "train") & folds.dup_group.isin(shared)]
    assert len(train), "no multi-member dup_group in train to mutate"
    victim = train.iloc[0]
    broken = _mutate(folds, victim.file_id,
                     **{"slice": "probe", "artifact_family": None,
                        "source_name": "elsewhere", "speaker_ref_id": None,
                        "pair_id": None})
    report = check_split_integrity(broken)
    assert "A5_dup_group_in_one_slice" in report.failures
    assert victim.dup_group in report.failures["A5_dup_group_in_one_slice"]


def test_A6_catches_a_probe_family_seen_in_training(folds):
    victim = _first(folds, "probe", artifact_family=None)
    train_family = _first(folds, "train", artifact_family=None).artifact_family
    broken = _mutate(folds, victim.file_id, artifact_family=train_family)
    report = check_split_integrity(broken)
    assert "A6_probe_sealed" in report.failures


def test_A6_catches_an_empty_probe_that_A1_would_pass_vacuously(folds):
    """🔴 The clause A1 does not carry. A builder that produced no PROBE at all
    leaves A1 green over an empty slice."""
    broken = folds.copy()
    broken.loc[broken["slice"] == "probe", "slice"] = "train"
    report = check_split_integrity(broken)
    assert "A6_probe_sealed" in report.failures
    assert "PROBE IS EMPTY" in report.failures["A6_probe_sealed"]
    assert "A1_family_in_one_slice" not in report.failures


def test_A7_catches_a_dangling_shadow_of(manifest, folds):
    """The VAL->SHADOW drop is only attributable to the channel if the pair is real."""
    out = build_folds(manifest, FoldConfig(assigned_at="x"),
                      shadow_of=_shadowed(manifest, folds)).frame
    victim = out[out.shadow_kind == "a"].iloc[0]
    broken = _mutate(out, victim.file_id, shadow_of="not_a_val_file")
    report = check_split_integrity(broken)
    assert "A7_shadow_a_points_at_val" in report.failures
    assert "not_a_val_file" in report.failures["A7_shadow_a_points_at_val"]


def test_A7_is_skipped_rather_than_passed_when_there_are_no_shadow_rows(folds):
    report = check_split_integrity(folds)
    assert "A7_shadow_a_points_at_val" in report.skipped


def test_A10_catches_a_scheme_version_the_run_did_not_use(folds):
    """Results across scheme versions are never compared."""
    report = check_split_integrity(folds, run_scheme_version="v2")
    assert "A10_scheme_version_matches" in report.failures
    assert check_split_integrity(folds, run_scheme_version="synthetic-v1").ok


def test_A10_is_skipped_rather_than_passed_without_a_run_version(folds):
    report = check_split_integrity(folds)
    assert "A10_scheme_version_matches" in report.skipped


def test_raise_for_status_fires_on_a_broken_split(folds):
    broken = folds.copy()
    broken.loc[broken["slice"] == "probe", "slice"] = "train"
    with pytest.raises(AssertionError, match="A6_probe_sealed"):
        check_split_integrity(broken).raise_for_status()


# --------------------------------------------------------------------------- #
# schema

@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.drop(columns=["dup_group"]), "missing column"),
    (lambda d: d.assign(**{"slice": "holdout"}), "unknown slice"),
    (lambda d: d.assign(shadow_kind="c"), "shadow_kind must be"),
    (lambda d: d.assign(scheme_version=["v1"] + ["v2"] * (len(d) - 1)), "mixes scheme_version"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "duplicate file_id"),
])
def test_validate_folds_rejects_a_malformed_table(folds, mutate, match):
    with pytest.raises(ValueError, match=match):
        validate_folds(mutate(folds))


def test_validate_folds_rejects_a_fold_on_a_train_row(folds):
    broken = folds.copy()
    idx = broken.index[broken["slice"] == "train"][0]
    broken.loc[idx, "fold"] = 2
    with pytest.raises(ValueError, match="fold must be null outside"):
        validate_folds(broken)


def test_validate_folds_rejects_a_val_row_without_a_fold(folds):
    broken = folds.copy()
    idx = broken.index[broken["slice"] == "val"][0]
    broken.loc[idx, "fold"] = pd.NA
    with pytest.raises(ValueError, match="needs a fold"):
        validate_folds(broken)


def test_validate_folds_rejects_a_cell_on_a_component_row(folds):
    broken = folds.copy()
    idx = broken.index[broken.row_kind == "component"][0]
    broken.loc[idx, "cell"] = 5
    with pytest.raises(ValueError, match="cell must be non-null exactly"):
        validate_folds(broken)
