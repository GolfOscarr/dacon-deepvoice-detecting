"""The fold builder: the rotation, the grouping closure, and infeasibility.

`build_folds` is the source of every `(slice, fold)` assignment, so what is
asserted here is that the assignment it produces has the properties the scheme
claims -- a family VAL in exactly one fold, PROBE sealed, SHADOW carried with
its parent -- and that a corpus that cannot support the split is *reported*
rather than quietly downgraded.

The mutation tests that prove each VG1 assertion can fail live next door, in
`tests/test_foldcheck.py`; `check_split_integrity` is used here only as the
oracle for a table this module built.
"""

import pandas as pd
import pytest

from training.folds import (FOLD_COLUMNS, SLICES, FoldConfig, FoldInfeasible,
                            apply_folds, build_folds, check_split_integrity,
                            grouping_atoms, load_folds)
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
# the table itself

def test_emitted_columns_are_exactly_the_spec(folds):
    """docs/validation/01 §3 lists the columns; drift there is silent."""
    assert list(folds.columns) == list(FOLD_COLUMNS)


def test_every_manifest_row_is_assigned_exactly_once(manifest, folds):
    assert len(folds) == len(manifest)
    assert set(folds.file_id) == set(manifest.file_id.astype(str))
    assert not folds.file_id.duplicated().any()
    assert set(folds["slice"]) <= set(SLICES)


# --------------------------------------------------------------------------- #
# Critical: the rotation itself

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
    """Critical: `build_folds` is the *source* of the assignment. Reading a stale one
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
# Critical -- grouping is the transitive closure, not five independent checks

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
    """Critical: the failure mode `training.synthetic._TWIN_COMBINATIONS` documents.

    Twinning every family to every real corpus collapses the grouping keys into
    one inseparable atom. The builder must say so, not quietly drop a key.
    """
    fused = manifest.copy()
    fused.loc[fused.pool.isin(["A", "B"]), "pair_id"] = "everything_is_one_twin"
    with pytest.raises(FoldInfeasible) as e:
        build_folds(fused, FoldConfig(assigned_at="x"))
    assert "cannot fill 5 folds" in str(e.value) or "cannot draw from" in str(e.value)


# --------------------------------------------------------------------------- #
# Critical: failing loudly

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
    """Critical: Under rotation the fold count is bounded by the number of real source
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
    """Caveat: the caveat must be falsifiable too: one that always fires is noise."""
    assert not [c for c in plan.caveats if "per validation fold" in c], plan.caveats


def test_an_empty_probe_must_be_asked_for(manifest):
    cfg = dict(probe_share=0.0, assigned_at="x")
    with pytest.raises(FoldInfeasible, match="allow_no_probe"):
        build_folds(manifest, FoldConfig(**cfg))
    plan = build_folds(manifest, FoldConfig(allow_no_probe=True, **cfg))
    assert any("PROBE" in c for c in plan.caveats)
    # Critical: ...and VG1 A6 goes red for it. An empty sealed slice is not a
    # pass.
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
    """Caveat: SHADOW is "VAL generators × unseen acoustic conditions". On a fold
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
