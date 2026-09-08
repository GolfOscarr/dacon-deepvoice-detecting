"""`training.dataset` -- the sampler → render → collate path, end to end.

🔴 The two things this module decides, and which are therefore tested rather
than described: **the fold is resolved before anything is drawn** (a composed
sample's components can straddle folds, so a sample has no fold of its own), and
**the eval spec list is a value, not a seed**.

I17-I20 live here because they are contract tests *against the model*: a batch
this pipeline produces must be accepted by `DeepVoiceNet.forward` and produce a
finite `multitask_loss` on every branch, including the degenerate batch where no
row carries a component at all.
"""

import numpy as np
import pytest
import torch

from models.audio import prepare_waveform
from models.config import AudioConfig, LossConfig, load_model_config
from models.losses import TARGET_FOR_COLUMN, multitask_loss
from models.model import DeepVoiceNet
from training.collate import bucket_edges, bucket_of, collate, spec_durations
from training.dataset import (SpecDataset, eval_batches, fold_manifest,
                              frozen_eval_specs, training_batches)
from training.folds import FoldConfig, build_folds
from training.render import ManifestIndex, RenderConfig, render
from training.sampler import Sampler, SamplerConfig
from training.spec import ComponentDraw, SampleSpec
from training.synthetic import synthetic_manifest, write_synthetic_corpus

SR = 16_000
CORPUS = dict(n_per_pool=6, n_whole_file=6, seed=0, duration_range=(7.0, 9.0))
DRAW = SamplerConfig(duration_range=(4.0, 6.0))
STUBS = ("a_stub", "b_stub")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset-corpus")
    manifest = synthetic_manifest(**CORPUS)
    write_synthetic_corpus(manifest, root, seed=0)
    return root, manifest, ManifestIndex.from_frame(manifest)


@pytest.fixture(scope="module")
def dataset(corpus):
    root, manifest, index = corpus
    sampler = Sampler(manifest, DRAW, slice_="train")
    return SpecDataset.from_sampler(sampler, 8, index, RenderConfig(root=root))


@pytest.fixture(scope="module")
def batch(dataset):
    return collate([dataset[i] for i in range(len(dataset))])


@pytest.fixture(scope="module")
def model():
    return DeepVoiceNet(load_model_config("configs/b_stub.yaml")).eval()


# --------------------------------------------------------------------------- #
# The dataset protocol


def test_indexing_renders_the_spec_at_that_index(dataset):
    sample = dataset[3]
    assert sample.spec == dataset.specs[3]
    assert sample.wav.dim() == 2 and sample.sample_rate == SR


def test_rendering_the_same_index_twice_is_bitwise_identical(dataset):
    """The dataset must not add entropy of its own on top of `spec.rng`."""
    assert torch.equal(dataset[2].wav, dataset[2].wav)


# --------------------------------------------------------------------------- #
# The frozen eval set is a value, not a seed


def test_the_frozen_eval_set_is_the_same_rows_every_epoch(corpus):
    """🔴 The reason sampling is separated from rendering at all.

    A seed reproduces only against the same sampler, the same manifest and the
    same code -- all of which change during a competition. The eval set is
    carried as specs.
    """
    root, manifest, index = corpus
    sampler = Sampler(manifest, DRAW, slice_="train")
    specs = frozen_eval_specs(sampler, 6, seed=0)
    ds = SpecDataset.frozen(specs, index, RenderConfig(root=root), slice_="train",
                            fold=None)   # this corpus is unfolded; see below

    assert ds.is_frozen
    assert ds.specs == specs
    with pytest.raises(RuntimeError, match="frozen"):
        ds.set_epoch(1)


def test_the_freeze_can_actually_be_broken(corpus):
    """Mutation for the check above: the training dataset *does* redraw.

    If `set_epoch` were a no-op everywhere, the refusal above would be vacuous.
    """
    root, manifest, index = corpus
    sampler = Sampler(manifest, DRAW, slice_="train")
    ds = SpecDataset.from_sampler(sampler, 6, index, RenderConfig(root=root))
    before = ds.specs
    ds.set_epoch(1)
    assert ds.specs != before, "set_epoch did not redraw"
    assert [s.sample_id for s in ds.specs] == [s.sample_id for s in before], \
        "sample_id is the position in the epoch; only `epoch` should move"
    assert all(s.epoch == 1 for s in ds.specs)


def test_the_eval_plan_is_unbucketed_in_order_and_drops_nothing(corpus):
    """docs/pipelines/04 §3: the duration-vs-score check on the REAL class must
    be measured on unbucketed batches, and the frozen list gives that for free
    -- as long as nobody buckets it."""
    root, manifest, index = corpus
    sampler = Sampler(manifest, DRAW, slice_="train")
    ds = SpecDataset.frozen(frozen_eval_specs(sampler, 7, seed=0), index,
                            RenderConfig(root=root), slice_="train", fold=None)

    plan = eval_batches(ds, 3)
    assert [i for b in plan for i in b] == list(range(7)), \
        "the eval plan must be in order and complete"
    with pytest.raises(ValueError, match="unbucketed"):
        training_batches(ds, 3)


def test_an_omitted_slice_is_not_a_declaration_either(folded):
    """🔴 The `fold` trap, one argument over -- and this one drew blood.

    `slice_` names where the specs were **drawn from**, not the role they are
    used in. The old `"val"` default silently asserted a provenance the caller
    never stated, and when `training.loop` wired I5 in properly it reported **23
    file_ids outside the declared slice** for an eval set drawn from a
    `slice_="train"` sampler. The specs were right; the declaration was invented.
    """
    corpus, folds = folded
    resolved = fold_manifest(corpus, folds, fold=0)
    specs = list(Sampler(resolved, SamplerConfig(), slice_="train").epoch_specs(
        200, seed=0))
    index = ManifestIndex.from_frame(resolved)

    omitted = SpecDataset.frozen(specs, index, fold=None)
    assert omitted.slice_ is None, "an omitted slice must not invent one"
    why = omitted.audit(resolved).results["I5_split_safety"][1]
    assert why.startswith("SKIPPED"), why

    # Non-vacuity: naming the slice the specs were drawn from makes I5 run clean.
    stated = SpecDataset.frozen(specs, index, slice_="train", fold=None)
    ok, ran = stated.audit(resolved).results["I5_split_safety"]
    assert ok and not ran.startswith("SKIPPED"), ran

    # And the old default was not merely unstated, it was wrong: declaring "val"
    # for train-drawn specs is what produced the 23 outside-the-slice report.
    lied = SpecDataset.frozen(specs, index, slice_="val", fold=None)
    bad, why_bad = lied.audit(resolved).results["I5_split_safety"]
    assert not bad, "the mislabelled provenance must be caught, not tolerated"
    assert "outside" in why_bad


def test_the_eval_plan_refuses_a_training_dataset(corpus):
    """🔴 The mirror of `training_batches` refusing a frozen list.

    A training dataset's specs are redrawn by `set_epoch`, so an eval plan built
    over it goes stale silently: the indices stay valid and the length is
    unchanged, so the validation curve keeps plotting while the rows underneath
    it are a different sample. Nothing downstream can notice.
    """
    # ⚠️ Its own dataset, not the module fixture: this test calls `set_epoch`,
    # and mutating shared state would make every later test depend on ordering.
    root, manifest, index = corpus
    own = SpecDataset.from_sampler(Sampler(manifest, DRAW, slice_="train"), 6,
                                   index, RenderConfig(root=root))
    with pytest.raises(ValueError, match="redrawn by `set_epoch`"):
        eval_batches(own, 3)

    # Non-vacuity: the indices really do survive the redraw and come to mean
    # something else, which is what makes the silent version dangerous.
    before = own.specs
    own.set_epoch(own.epoch + 1)
    assert len(own.specs) == len(before) and own.specs != before
    assert [s.sample_id for s in own.specs] == [s.sample_id for s in before]


def test_the_training_plan_is_bucketed(dataset):
    """🔴 Asserts the four things a bucketer must satisfy, not just its shape.

    The first version checked `plan and all(len(b) == 2 for b in plan)`, which a
    plan that ignored the durations, ignored `seed`, or dropped half the stream
    would pass.
    """
    d = spec_durations(dataset.specs)
    plan = training_batches(dataset, 2, n_buckets=2, seed=0)
    edges = bucket_edges(d, 2)

    assert plan and all(len(b) == 2 for b in plan)             # full batches
    flat = [i for b in plan for i in b]
    assert len(flat) == len(set(flat)) and set(flat) <= set(range(len(dataset)))
    for b in plan:                                             # within one bucket
        assert len({bucket_of(d[i], edges) for i in b}) == 1
    assert training_batches(dataset, 2, n_buckets=2, seed=0) == plan   # deterministic


def test_the_training_plan_actually_reads_its_seed(dataset):
    """Mutation: `training_batches` dropping its `seed=` on the floor passed
    every other test, so every epoch would draw the identical batching."""
    plans = {tuple(tuple(b) for b in training_batches(dataset, 2, n_buckets=2,
                                                     seed=k))
             for k in range(6)}
    assert len(plans) > 1, "the seed changes nothing -- it is not being read"


def test_audit_measures_the_plan_the_optimiser_will_see(dataset, corpus):
    """Finding 5 at the dataset boundary: `audit(batches=)` reaches I9.

    `training/audit.py` says "audit the same order you train in", and M4 made
    that order bucketed -- so the argument has to exist and be forwarded.
    """
    _, manifest, _ = corpus
    plan = training_batches(dataset, 2, n_buckets=2, seed=0)

    why = dataset.audit(manifest, batches=plan, min_present=0).results[
        "I9_C2_present_count_floor"][1]
    assert f"{len(plan)} batch(es)" in why and "supplied plan" in why

    draw_order = dataset.audit(manifest, batch_size=2, min_present=0).results[
        "I9_C2_present_count_floor"][1]
    assert "draw order" in draw_order


def test_the_dropped_row_count_says_what_it_counted(corpus):
    """🔴 Report text, but the kind that goes quietly wrong.

    The two I9 branches drop rows for different reasons -- `drop_last`'s
    remainder in draw order, versus whatever a supplied plan never indexes -- and
    both used to print "trailing spec(s) excluded (drop_last)". For a
    `bucket_batches` plan those coincide; for a partial or overlapping plan they
    do not, and the number would mean something other than what it says.

    Neither branch's count was asserted anywhere before this, which is how the
    supplied-plan one was added wrong-and-green. Both are pinned here, on plans
    with a known remainder.
    """
    root, manifest, index = corpus
    sampler = Sampler(manifest, DRAW, slice_="train")
    specs = list(sampler.epoch_specs(10, seed=0))
    ds = SpecDataset.frozen(specs, index, RenderConfig(root=root),
                            slice_="train", fold=None)

    # draw order: 10 specs at batch_size 4 -> two batches, 2 left over.
    why = ds.audit(batch_size=4, min_present=0).results[
        "I9_C2_present_count_floor"][1]
    assert "2 trailing spec(s) excluded (drop_last)" in why, why

    # a supplied plan that covers 6 of 10 -> 4 never batched, and it must not
    # call them a drop_last remainder.
    partial = ds.audit(batches=[[0, 1, 2], [3, 4, 5]], min_present=0).results[
        "I9_C2_present_count_floor"][1]
    assert "4 spec(s) the plan never batches" in partial, partial
    assert "drop_last" not in partial

    # an overlapping plan is reported as such: a spec in two batches is stepped
    # on twice, and the dropped count is then not a remainder at all.
    overlap = ds.audit(batches=[[0, 1, 2], [2, 3, 4]], min_present=0).results[
        "I9_C2_present_count_floor"][1]
    assert "1 spec(s) batched more than once" in overlap, overlap

    # a full plan drops nothing and says nothing.
    full = ds.audit(batches=[list(range(5)), list(range(5, 10))],
                    min_present=0).results["I9_C2_present_count_floor"][1]
    assert "never batches" not in full and "drop_last" not in full


def test_the_audit_batch_size_is_not_a_silent_default(dataset, corpus):
    """`batch_size` used to be set by `kw.setdefault(...)` and read by nothing:
    changing 32 to 2 passed the whole suite."""
    _, manifest, _ = corpus
    for size in (2, 4):
        why = dataset.audit(manifest, batch_size=size, min_present=0).results[
            "I9_C2_present_count_floor"][1]
        assert f"batch_size={size}" in why, why


# --------------------------------------------------------------------------- #
# The fold is resolved before anything is drawn


@pytest.fixture(scope="module")
def folded():
    """A corpus big enough to split, and its fold-0 view."""
    corpus = synthetic_manifest(n_per_pool=240, n_whole_file=200,
                               n_families=24, n_sources=8)
    plan = build_folds(corpus, FoldConfig(n_folds=5))
    return corpus, plan.frame


@pytest.mark.parametrize("fold", range(5))
def test_a_composed_sample_may_straddle_families_but_never_the_fold(folded, fold):
    """🔴 The decision, made visible.

    "The fold of a composed sample" is undefined -- its components are drawn
    independently and nothing binds them to one family. So the dataset never
    asks: `fold_manifest(..., fold=k)` resolves TRAIN/VAL on the *manifest*, and
    the sampler then has nothing out-of-fold to draw from. Containment is
    structural; a rule like "the fold of the first component" would have been an
    accident of draw order.

    The test asserts both halves: components genuinely do span families (so the
    straddling case is real and not hypothetical), and no component lies outside
    the fold-resolved slice.
    """
    corpus, folds = folded
    train = fold_manifest(corpus, folds, fold=fold)
    sampler = Sampler(train, SamplerConfig(), slice_="train")
    specs = list(sampler.epoch_specs(400, seed=fold))

    # ⚠️ Swept over every fold, and that is not thoroughness for its own sake: a
    # single-fold version of this test passed against a `fold_manifest` that
    # ignored its `fold=` argument entirely and always resolved fold 0.
    held_out = set(train.loc[train["slice"] == "val", "file_id"].astype(str))
    assert held_out, f"fold {fold} holds nothing out"
    # 🔴 Cross-checked against the fold table, not against the frame under test.
    # An earlier version derived both sides from `fold_manifest`'s own output and
    # passed against an implementation that ignored `fold=` and always resolved
    # fold 0 -- the "adjacent quantity" pattern, one level up.
    expected = set(folds.loc[(folds["slice"] == "train_val")
                             & (folds["fold"] == fold), "file_id"].astype(str))
    assert held_out == expected, f"fold {fold} validated on the wrong rows"

    family = corpus.set_index("file_id").artifact_family.to_dict()
    spanning = sum(1 for s in specs if len(s.components) > 1
                   and len({family.get(c.file_id) for c in s.components}) > 1)
    assert spanning > 0, "no composed sample spans families; the case is untested"

    drawn = {c.file_id for s in specs for c in s.components}
    allowed = set(train.loc[train["slice"] == "train", "file_id"].astype(str))
    assert not drawn - allowed
    assert not drawn & held_out, f"fold {fold} drew its own validation rows"


def test_the_fold_containment_is_measured_by_I5_not_rebuilt(folded):
    """`SpecDataset.audit` forwards `slice_`/`fold`, which is what makes I5 run.

    docs/pipelines/05 §5: do not rebuild a check that exists. The dataset's job
    is to wire the arguments so the existing check can see the stream -- and
    called without a manifest, I5 must report SKIP, never PASS.

    ⚠️ On a frame that `fold_manifest` has already resolved, `fold=None` is the
    *honest* argument and I5 passes -- but the report has to say why, because the
    same argument on an unresolved frame means the fold clause never ran. The
    next test is that half.
    """
    corpus, folds = folded
    train = fold_manifest(corpus, folds, fold=0)
    sampler = Sampler(train, SamplerConfig(), slice_="train")
    ds = SpecDataset(sampler.epoch_specs(300, seed=0), ManifestIndex.from_frame(train),
                     slice_="train", fold=None, sampler=sampler, n=300)

    ok, why = ds.audit(train).results["I5_split_safety"]
    assert ok, why
    assert not why.startswith("SKIPPED")
    assert "vacuous" in why, \
        "a PASS with no fold given must say why the fold clause added nothing"

    blind = ds.audit().results["I5_split_safety"]
    assert blind[1].startswith("SKIPPED"), "a check with no manifest must not pass"


def test_an_unresolved_frame_without_a_fold_reports_skip_not_pass(folded):
    """🔴 The frozen eval set is the instrument this project trusts over the
    leaderboard, and it defaulted to `fold=None` -- so I5's fold clause did not
    run and the report said PASS. `AuditReport.SKIP` exists precisely for this,
    and the convention existing while being violated here is worse than the gap.

    An *unresolved* frame spans several folds, so `fold=None` leaves half of a
    two-clause invariant unevaluated. It must read SKIP.
    """
    corpus, folds = folded
    keyed = folds.set_index("file_id")
    unresolved = corpus.assign(
        slice=keyed["slice"].reindex(corpus.file_id.astype(str)).to_numpy(),
        fold=keyed["fold"].reindex(corpus.file_id.astype(str)).to_numpy())
    assert unresolved.loc[unresolved["slice"] == "train_val", "fold"].nunique() > 1

    resolved = fold_manifest(corpus, folds, fold=0)
    sampler = Sampler(resolved, SamplerConfig(), slice_="train")
    specs = list(sampler.epoch_specs(200, seed=0))

    blind = SpecDataset.frozen(specs, ManifestIndex.from_frame(resolved),
                               slice_="train_val", fold=None)
    passed, why = blind.audit(unresolved).results["I5_split_safety"]
    assert why.startswith("SKIPPED"), why
    assert "fold half of I5 did not run" in why

    # Non-vacuity: naming the fold makes the same check run and report.
    named = SpecDataset.frozen(specs, ManifestIndex.from_frame(resolved),
                               slice_="train_val", fold=0)
    _, ran = named.audit(unresolved).results["I5_split_safety"]
    assert not ran.startswith("SKIPPED") and "fold=0" in ran


def test_an_omitted_fold_is_not_the_same_as_a_stated_one(folded):
    """🔴 "I decided the clause is vacuous" and "I never thought about it" must
    not be the same argument.

    `None` is a *meaningful* value for `fold` -- on a `fold_manifest`-resolved
    frame the clause genuinely has nothing to add -- so a plain `None` default
    made the two indistinguishable, and the frozen eval set took the PASS. The
    sentinel splits them: stated `None` passes and says the clause was vacuous,
    an omitted `fold` reports SKIP no matter how the manifest looks.
    """
    corpus, folds = folded
    resolved = fold_manifest(corpus, folds, fold=0)
    specs = list(Sampler(resolved, SamplerConfig(), slice_="train").epoch_specs(200,
                                                                               seed=0))
    index = ManifestIndex.from_frame(resolved)

    omitted = SpecDataset.frozen(specs, index, slice_="train")
    stated = SpecDataset.frozen(specs, index, slice_="train", fold=None)
    assert omitted.fold == stated.fold is None, "the two must agree on the value"

    why_omitted = omitted.audit(resolved).results["I5_split_safety"][1]
    why_stated = stated.audit(resolved).results["I5_split_safety"][1]
    assert why_omitted.startswith("SKIPPED"), why_omitted
    assert "not a decision" in why_omitted
    assert not why_stated.startswith("SKIPPED"), why_stated
    assert "vacuous" in why_stated


def test_the_containment_check_can_fail(folded):
    """Mutation: splice one out-of-fold component in and I5 must go red."""
    corpus, folds = folded
    train = fold_manifest(corpus, folds, fold=0)
    sampler = Sampler(train, SamplerConfig(), slice_="train")
    specs = list(sampler.epoch_specs(300, seed=0))

    outsider = str(train.loc[train["slice"] == "val", "file_id"].iloc[0])
    victim = next(s for s in specs if s.components[0].role == "voice")
    leaked = SampleSpec(
        **{**victim.to_dict(),
           "components": (ComponentDraw(
               **{**victim.components[0].__dict__, "file_id": outsider}),
               *victim.components[1:])})
    specs[specs.index(victim)] = leaked

    ds = SpecDataset(specs, ManifestIndex.from_frame(train),
                     slice_="train", fold=None)
    ok, why = ds.audit(train).results["I5_split_safety"]
    assert not ok and outsider in why


# --------------------------------------------------------------------------- #
# I17-I20 -- contract tests against the model


@pytest.mark.parametrize("name", STUBS)
def test_a_collated_batch_is_accepted_by_the_model(batch, name):
    """I17, on both shipped stub configs."""
    cfg = load_model_config(f"configs/{name}.yaml")
    model = DeepVoiceNet(cfg).eval()
    wav = prepare_waveform(batch["wav"], cfg.audio)
    assert wav.dim() == 2 and wav.shape[0] == batch["wav"].shape[0]
    with torch.no_grad():
        out = model(wav, batch["lengths"])
    for branch in cfg.branches:
        assert out[branch]["clip_logits"].shape == (batch["wav"].shape[0],)
        assert torch.isfinite(out[branch]["clip_logits"]).all()


def test_the_batch_targets_are_exactly_the_loss_keys(batch):
    """I18. `multitask_loss` raises on a missing key, so this is the same
    statement one step earlier, where it is cheap."""
    assert set(batch["targets"]) == set(TARGET_FOR_COLUMN.values())


def test_the_missing_target_key_check_can_fail(batch, model):
    """Mutation for I18: drop a key and `multitask_loss` must refuse."""
    cfg = load_model_config("configs/b_stub.yaml")
    thin = {k: v for k, v in batch["targets"].items() if k != "music_present"}
    with torch.no_grad():
        out = model(prepare_waveform(batch["wav"], cfg.audio), batch["lengths"])
    with pytest.raises(KeyError, match="music_present"):
        multitask_loss(out, thin, cfg, LossConfig())


def test_every_branch_contributes_a_finite_loss(batch, model):
    """I19. A masking bug that zeroes a head is invisible in the total."""
    cfg = load_model_config("configs/b_stub.yaml")
    out = model(prepare_waveform(batch["wav"], cfg.audio), batch["lengths"])
    total, parts = multitask_loss(out, batch["targets"], cfg, LossConfig())

    assert torch.isfinite(total)
    assert set(parts) >= set(cfg.branches)
    for branch in cfg.branches:
        assert np.isfinite(parts[branch]), branch
        assert parts[branch] > 0.0, f"{branch} contributed nothing"


def test_a_batch_with_no_present_component_still_gives_a_finite_loss(corpus, model):
    """I20, at pipeline output rather than at the loss boundary.

    `tests/test_losses.py` covers `_masked_mean`'s zero-denominator path with
    hand-made tensors; this proves a batch the *pipeline* can actually produce
    -- every row cell 9, so `voice_present` and `music_present` are 0 throughout
    -- reaches the same place. It is also the DDP case: a head with no
    contributing sample has unused parameters, and a NaN here deadlocks a run at
    hour three rather than failing at minute one.
    """
    root, manifest, index = corpus
    cfg = load_model_config("configs/b_stub.yaml")
    noise_ids = manifest[manifest.pool == "E"].file_id.tolist()
    assert len(noise_ids) >= 2

    specs = [SampleSpec(
        sample_id=i, epoch=0, seed=0, scheme_version="synthetic-v1",
        duration_s=4.5, cell=9, render_mode="composed", structure="overlap",
        components=(ComponentDraw(str(fid), "noise", 0.0, 4.5, 0.0, 0.0),))
        for i, fid in enumerate(noise_ids[:2])]
    ds = SpecDataset.frozen(specs, index, RenderConfig(root=root), slice_="train",
                            fold=None)
    batch = collate([ds[i] for i in range(len(ds))])

    assert float(batch["targets"]["voice_present"].sum()) == 0.0
    assert float(batch["targets"]["music_present"].sum()) == 0.0

    out = model(prepare_waveform(batch["wav"], cfg.audio), batch["lengths"])
    total, parts = multitask_loss(out, batch["targets"], cfg, LossConfig())
    assert torch.isfinite(total), "a component-free batch produced a non-finite loss"
    for branch, value in parts.items():
        assert np.isfinite(value), branch


# --------------------------------------------------------------------------- #
# I12 over the dataset's own stream


def test_every_sample_the_dataset_yields_sits_in_the_length_regime(dataset):
    """I12 asserted on `AudioConfig.min_seconds` / `max_seconds` themselves.

    ⚠️ Not on an adjacent quantity: a version of this assertion written as
    `0 < duration` passed while 2,580 of 4,000 samples were under the 4 s floor.
    """
    audio = AudioConfig()
    for i in range(len(dataset)):
        duration = dataset[i].wav.shape[-1] / SR
        assert audio.min_seconds - 1e-6 <= duration <= audio.max_seconds + 1e-6, \
            f"sample {i} rendered {duration:.3f}s"
