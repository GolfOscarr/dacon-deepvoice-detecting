"""The fold builder, and proof that every VG1 assertion can actually fail.

🔴 The same rule as `tests/test_audit.py`: a green gate that cannot go red is
worse than no gate. A review of this repo found three audit checks that could not
fail and one printing PASS for something implemented nowhere. So every assertion
in `check_split_integrity` has a mutation test below that breaks the constraint
and requires the assertion to fire -- and the two that cannot be computed on this
table (A8/A9) are asserted to report SKIPPED, never PASS.

⚠️ Under the rotating scheme every mutation has to be *isolated*: moving one row
to another fold breaks every grouping key on it at once, so each test nulls the
four keys it is not testing. Otherwise five assertions fire together and none of
them is shown to be doing its own work.
"""

import pandas as pd
import pytest

from training.audit import AuditReport
from training.folds import (FOLD_COLUMNS, SLICES, FoldConfig, FoldInfeasible,
                            apply_folds, build_folds, check_split_integrity,
                            grouping_atoms, load_folds, validate_folds)
from training.sampler import Sampler
from training.synthetic import synthetic_manifest

#: 24 families and 8 real corpora per role. ⚠️ Deliberately wider than the
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
# the table itself

def test_emitted_columns_are_exactly_the_spec(folds):
    """docs/validation/01 §3 lists the columns; drift there is silent."""
    assert list(folds.columns) == list(FOLD_COLUMNS)


def test_every_manifest_row_is_assigned_exactly_once(manifest, folds):
    assert len(folds) == len(manifest)
    assert set(folds.file_id) == set(manifest.file_id.astype(str))
    assert not folds.file_id.duplicated().any()
    assert set(folds["slice"]) <= set(SLICES)


def test_the_reference_split_passes_every_vg1_assertion(folds):
    report = check_split_integrity(folds, run_scheme_version="synthetic-v1")
    assert report.ok, str(report)


# --------------------------------------------------------------------------- #
# 🔴 the rotation itself

def test_every_rotating_family_is_val_in_exactly_one_fold(manifest, folds):
    """The defining property of the scheme, measured on the materialized views
    rather than on the fold column it was built from."""
    seen = {}
    for k in range(N_FOLDS):
        view = apply_folds(manifest, folds, fold=k)
        for role in ("train", "val"):
            fams = set(view.loc[view["slice"] == role, "artifact_family"].dropna())
            for fam in fams:
                seen.setdefault(fam, {"train": 0, "val": 0})[role] += 1
    assert seen, "no families at all"
    for fam, counts in seen.items():
        assert counts["val"] == 1, f"{fam} is VAL in {counts['val']} folds"
        assert counts["train"] == N_FOLDS - 1, f"{fam}: {counts}"


def test_no_family_is_train_and_val_in_the_same_fold(manifest, folds):
    """What A1 exists to prevent, checked against the materialized view."""
    for k in range(N_FOLDS):
        view = apply_folds(manifest, folds, fold=k)
        train = set(view.loc[view["slice"] == "train", "artifact_family"].dropna())
        val = set(view.loc[view["slice"] == "val", "artifact_family"].dropna())
        assert not (train & val), f"fold {k}: {sorted(train & val)[:3]}"
        for key in ("source_name", "speaker_ref_id", "pair_id", "dup_group"):
            a = set(view.loc[view["slice"] == "train", key].dropna())
            b = set(view.loc[view["slice"] == "val", key].dropna())
            assert not (a & b), f"fold {k} {key}: {sorted(a & b)[:3]}"


def test_probe_is_never_trained_on_or_validated_on(manifest, folds):
    sealed = set(folds.loc[folds["slice"] == "probe", "artifact_family"].dropna())
    assert sealed, "nothing sealed"
    for k in range(N_FOLDS):
        view = apply_folds(manifest, folds, fold=k)
        rotating = set(view.loc[view["slice"].isin(["train", "val"]),
                                "artifact_family"].dropna())
        assert not (sealed & rotating), f"fold {k}: {sorted(sealed & rotating)[:3]}"
        assert (view["slice"] == "probe").sum() == (folds["slice"] == "probe").sum()


def test_fold_is_null_exactly_on_probe(folds):
    """A sealed family is never VAL, so "the fold it is validated in" does not
    exist for it."""
    assert folds.loc[folds["slice"] == "probe", "fold"].isna().all()
    assert folds.loc[folds["slice"] != "probe", "fold"].notna().all()


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
    assert load_folds(path).equals(plan.frame)
    # The caveats travel with the table rather than with whoever remembers them.
    side = (tmp_path / "folds.caveats.txt").read_text()
    assert side.count("\n") == len(plan.caveats)


# --------------------------------------------------------------------------- #
# A8 / A9 live elsewhere, and really do

def test_a8_a9_report_skipped_and_never_pass(folds):
    """🔴 A skipped check must not read as a pass (`AuditReport.SKIP`)."""
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
# the split is usable

def test_the_sampler_can_draw_from_every_side_of_every_fold(manifest, folds):
    """A VAL side with no real voice components cannot compose cells 1/5/6, and
    the sampler only discovers that at draw time -- long after the split is
    frozen. Under rotation this binds per fold, which is why
    `require_component_coverage` checks TRAIN and VAL separately for each k."""
    for k in range(N_FOLDS):
        view = apply_folds(manifest, folds, fold=k)
        for role in ("train", "val", "probe"):
            specs = list(Sampler(view, slice_=role).epoch_specs(150))
            assert len(specs) == 150
            assert {s.file_fake for s in specs} == {0, 1}, (k, role)


def test_drawn_files_never_cross_the_train_val_boundary(manifest, folds):
    view = apply_folds(manifest, folds, fold=2)
    allowed = set(view.loc[view["slice"] == "val", "file_id"].astype(str))
    for spec in Sampler(view, slice_="val").epoch_specs(400):
        for c in spec.components:
            assert c.file_id in allowed


def test_apply_folds_rejects_a_fold_that_does_not_exist(manifest, folds):
    with pytest.raises(ValueError, match="fold must be in 0..4"):
        apply_folds(manifest, folds, fold=5)


def test_apply_folds_refuses_a_mismatched_id_set(manifest, folds):
    with pytest.raises(ValueError, match="disagree on"):
        apply_folds(manifest, folds.iloc[:-1], fold=0)


def test_apply_folds_refuses_a_scheme_version_mismatch(manifest, folds):
    with pytest.raises(ValueError, match="scheme_version mismatch"):
        apply_folds(manifest, folds.assign(scheme_version="v2"), fold=0)


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
    same = {atoms[f] for f in m[(m.artifact_family == twin.artifact_family)
                                | (m.source_name == partner.source_name)].index}
    assert same == {atoms[twin.name]}


def test_a_corpus_whose_twins_fuse_everything_is_reported_not_split(manifest):
    """🔴 The failure mode `training.synthetic._TWIN_COMBINATIONS` documents.

    Twinning every family to every real corpus collapses the grouping keys into
    one inseparable atom. The builder must say so, not quietly drop a key.
    """
    fused = manifest.copy()
    fused.loc[fused.pool.isin(["A", "B"]), "pair_id"] = "everything_is_one_twin"
    with pytest.raises(FoldInfeasible) as e:
        build_folds(fused, FoldConfig(assigned_at="x"))
    assert "cannot fill 5 folds" in str(e.value) or "cannot draw from" in str(e.value)


# --------------------------------------------------------------------------- #
# 🔴 failing loudly

def test_too_few_families_is_fatal_and_names_the_family_floor(manifest):
    """A fold that validates on no family of its own validates nothing."""
    narrow = synthetic_manifest(n_per_pool=240, n_whole_file=200, seed=0,
                                n_families=4, n_sources=8)
    with pytest.raises(FoldInfeasible) as e:
        build_folds(narrow, FoldConfig(assigned_at="x"))
    assert "cannot fill 5 folds" in str(e.value)
    assert "docs/validation/01 §3" in str(e.value)
    assert "Do not shrink PROBE to make this pass" in str(e.value)


def test_too_few_real_sources_is_fatal_and_names_the_new_bound(manifest):
    """🔴 Under rotation the fold count is bounded by the number of real source
    corpora per role, not only by the family count: each of the k VAL sides needs
    its own real voice, real music and noise source, and PROBE needs one more.

    The default corpus has three per role, which is why `WIDE` sets `n_sources`.
    """
    thin = synthetic_manifest(n_per_pool=240, n_whole_file=200, seed=0,
                              n_families=24)
    with pytest.raises(FoldInfeasible) as e:
        build_folds(thin, FoldConfig(assigned_at="x"))
    assert "cannot draw from" in str(e.value)
    assert "real source corpora" in str(e.value)
    # ...and it is feasible at a fold count the corpus can actually support.
    assert build_folds(thin, FoldConfig(n_folds=2, assigned_at="x")).frame is not None


def test_a_thin_head_gets_the_variance_caveat_with_realized_counts(manifest):
    """docs/validation/01 §3: at 8 music families a 5-fold leaves 1-2 per
    validation fold. The caveat states the *realized* per-fold counts, not the
    mean -- a mean of 1.4 hides that some fold validates on exactly one."""
    eight = synthetic_manifest(n_per_pool=240, n_whole_file=200, seed=0,
                               n_sources=8)
    plan = build_folds(eight, FoldConfig(assigned_at="x"))
    music = [c for c in plan.caveats if c.startswith("music head")]
    assert music, plan.caveats
    assert "per validation fold" in music[0] and "very high variance" in music[0]
    assert "[1, " in music[0], music[0]        # the realized list, not just a mean
    assert "CAVEAT" in str(plan)


def test_no_variance_caveat_when_every_fold_validates_on_enough(plan):
    """⚠️ The caveat must be falsifiable too: one that always fires is noise."""
    assert not [c for c in plan.caveats if "per validation fold" in c], plan.caveats


def test_an_empty_probe_must_be_asked_for(manifest):
    cfg = dict(probe_share=0.0, assigned_at="x")
    with pytest.raises(FoldInfeasible, match="allow_no_probe"):
        build_folds(manifest, FoldConfig(**cfg))
    plan = build_folds(manifest, FoldConfig(allow_no_probe=True, **cfg))
    assert any("PROBE" in c for c in plan.caveats)
    # 🔴 ...and VG1 A6 goes red for it. An empty sealed slice is not a pass.
    assert "A6_probe_sealed" in check_split_integrity(plan.frame).failures


def test_shadow_rows_must_name_rotating_files(manifest, folds):
    probe_id = folds.loc[folds["slice"] == "probe", "file_id"].iloc[0]
    rot_id = folds.loc[folds["slice"] == "train_val", "file_id"].iloc[0]
    cfg = FoldConfig(assigned_at="x")
    with pytest.raises(FoldInfeasible, match="paired to a rotating"):
        build_folds(manifest, cfg, shadow_of={rot_id: probe_id})
    with pytest.raises(FoldInfeasible, match="not out of sealed PROBE"):
        build_folds(manifest, cfg, shadow_b=[probe_id])
    with pytest.raises(FoldInfeasible, match="not in the manifest"):
        build_folds(manifest, cfg, shadow_b=["no_such_file"])


def test_an_impossible_configuration_is_rejected_at_construction():
    with pytest.raises(ValueError, match="probe_share"):
        FoldConfig(probe_share=1.0)
    with pytest.raises(ValueError, match="n_folds"):
        FoldConfig(n_folds=1)


def test_the_val_family_share_is_derived_not_set():
    """Under rotation TRAIN/VAL shares are a consequence of `n_folds`, not knobs."""
    assert FoldConfig(n_folds=5).val_family_share == pytest.approx(0.18)
    assert FoldConfig(n_folds=4).val_family_share == pytest.approx(0.225)


# --------------------------------------------------------------------------- #
# SHADOW

def test_shadow_rows_carry_their_parents_fold(manifest, folds):
    plan = build_folds(manifest, FoldConfig(assigned_at="x"),
                       shadow_of=_shadowed(folds))
    out = plan.frame
    sa = out[out.shadow_kind == "a"]
    assert len(sa) == 6
    fold_of = out.set_index("file_id")["fold"]
    for row in sa.itertuples():
        assert row.fold == fold_of[row.shadow_of]
    assert check_split_integrity(out).ok, str(check_split_integrity(out))


def test_shadow_is_dropped_from_the_folds_it_does_not_belong_to(manifest, folds):
    """⚠️ SHADOW is "VAL generators × unseen acoustic conditions". On a fold
    where those generators are being trained on, the row measures nothing and
    training on it would retire the condition."""
    out = build_folds(manifest, FoldConfig(assigned_at="x"),
                      shadow_of=_shadowed(folds)).frame
    total = 0
    for k in range(N_FOLDS):
        view = apply_folds(manifest, out, fold=k)
        here = view[view["slice"] == "shadow"]
        total += len(here)
        assert (out.set_index("file_id").loc[here["file_id"], "fold"] == k).all()
        assert "shadow" not in set(view.loc[view["slice"] == "train", "slice"])
    assert total == int((out["slice"] == "shadow").sum())


def test_shadow_consumes_no_family_budget(manifest, folds):
    """docs/validation/01 §2: SHADOW is a condition axis, not a generator axis."""
    out = build_folds(manifest, FoldConfig(assigned_at="x"),
                      shadow_of=_shadowed(folds)).frame
    sealed = set(out.loc[out["slice"] == "probe", "artifact_family"].dropna())
    assert sealed == set(folds.loc[folds["slice"] == "probe",
                                   "artifact_family"].dropna())


def test_s_b_rows_are_unpaired(manifest, folds):
    ids = folds.loc[folds["slice"] == "train_val", "file_id"].tolist()[:4]
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
    """🔴 The falsifiable form. Keyed on (source, speaker) instead, A3 would be
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
    # ⚠️ Built, not found. The synthetic corpus's dup groups happen to be mostly
    # singletons, and a dup_group with one member cannot span anything -- picking
    # whatever the generator emitted would have made this test unable to fail.
    # The assertion is about a *manifest* property, so the test states it.
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
    """🔴 The clause A1 does not carry, and the main thing A6 carries under
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
