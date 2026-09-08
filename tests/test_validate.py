"""`training.validate` -- scoring a fold, the VG gates and the leak tripwires.

Two shapes recur. The first is *never recompute a metric*: `evaluate` is proven
to go through the official estimator by breaking `metrics.dacon.roc_curve` and
watching it fail, rather than by comparing two numbers this repo produced.

The second is size dependence. VG5's B2 gate is a ratio against `n`, so a bf16
probability column passes it at n=400 -- a comfortable fixture size -- and fails
at the 1,200-per-class VAL floor. The assertions are therefore written at
`RESOLUTION_FLOOR` rather than at the size that is convenient, because this repo
has already shipped a padding defect that every test missed for exactly that
reason: every fixture used `lengths = SR * 4`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from loop_fixtures import (DRAW, _dataset, _fold_result, _metric_set, _model,
                           corpus, model_cfg)
from metrics.aggregate import fold_mean
from metrics.dacon import PREDICTION_COLUMNS, eer
from training.audit import AuditReport
from training.dataset import SpecDataset, fold_manifest, frozen_eval_specs
from training.folds import FoldConfig, build_folds
from training.loop import StageResult, stage_plan
from training.render import ManifestIndex
from training.sampler import Sampler
from training.synthetic import synthetic_manifest
from training.validate import (NOT_QUOTABLE, RESOLUTION_FLOOR, ValidationReport,
                               aggregate_folds, evaluate, generator_key,
                               leak_tripwires, measured_split_kind,
                               output_sanity, prediction_frame, run_gates,
                               validate_fold)


# --------------------------------------------------------------------------- #
# 1. Ranking resolution, and why eval is fp32


def _prob_frame(logits, dtype):
    p = torch.sigmoid(logits.to(dtype)).double().numpy()
    df = pd.DataFrame({"file_id": [f"s{i}" for i in range(len(p))]})
    for c in PREDICTION_COLUMNS:
        df[c] = p
    return df


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_a_small_n_pass_of_b2_never_reads_as_resolution_confirmed(seed):
    """B2 is a *ratio* against whatever `n` the caller passes, so on its own it
    has the shape of the defect it just caught: bf16 passes at n=400.

    **B2a** is the companion. Below the 1,200-row floor it SKIPs, so the pair can
    never come back "ranking resolution confirmed" on a set too small to see a
    precision-driven tie failure. `training.audit.RESOLVABLE_GAP` is the same
    pattern one layer down.
    """
    rng = torch.Generator().manual_seed(seed)
    small = _prob_frame(torch.randn(400, generator=rng) * 1.5, torch.bfloat16)

    report = output_sanity(small)
    # B2 itself passes at this size -- that is the whole problem ...
    assert report.results["VG5_B2_ranking_resolution"][0]
    # ... and B2a refuses to let that stand as evidence.
    assert "VG5_B2a_resolution_is_resolvable" in report.skipped
    assert "VG5_B2a_resolution_is_resolvable" not in report.ran

    big = _prob_frame(torch.randn(RESOLUTION_FLOOR, generator=rng) * 1.5,
                      torch.bfloat16)
    at_size = output_sanity(big)
    assert not at_size.results["VG5_B2_ranking_resolution"][0]
    assert "VG5_B2a_resolution_is_resolvable" in at_size.ran


def test_b2a_does_not_switch_b2_off_below_the_floor():
    """Caveat: the floor is about *precision-driven* ties only. A gross failure is
    visible at any n, and B2 must still fire on one -- otherwise the companion
    check would have quietly disabled the guard it exists to qualify."""
    n = 1000
    assert n < RESOLUTION_FLOOR
    rng = np.random.default_rng(0)
    clean = rng.random(n)
    saturated = np.where(clean <= 0.4, 0.0, np.where(clean >= 0.6, 1.0, clean))
    df = pd.DataFrame({"file_id": [f"s{i}" for i in range(n)]})
    for c in PREDICTION_COLUMNS:
        df[c] = saturated

    report = output_sanity(df)
    assert not report.results["VG5_B2_ranking_resolution"][0]
    assert "VG5_B2a_resolution_is_resolvable" in report.skipped


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_a_bf16_probability_column_loses_ranking_resolution_at_val_size(seed):
    """Why `LoopConfig.eval_precision` is fp32, measured at the size that
    matters.

    VG5's gate is `n_unique > 0.5 n`. At n=400 -- a comfortable fixture size --
    bf16 passes; at the 1,200-per-class VAL floor it does not. A test written at
    fixture size would have certified the wrong default, which is the same shape
    as the padding defect where every fixture used `lengths = SR * 4`.
    """
    rng = torch.Generator().manual_seed(seed)
    logits = torch.randn(RESOLUTION_FLOOR, generator=rng) * 1.5

    assert output_sanity(_prob_frame(logits, torch.float32)
                         ).results["VG5_B2_ranking_resolution"][0]
    assert not output_sanity(_prob_frame(logits, torch.bfloat16)
                             ).results["VG5_B2_ranking_resolution"][0]


# --------------------------------------------------------------------------- #
# 2. Validation goes through the official harness


@pytest.fixture(scope="module")
def scored(corpus, model_cfg):
    manifest, index, rcfg = corpus
    torch.manual_seed(0)
    model = _model(model_cfg)
    specs = frozen_eval_specs(Sampler(manifest, DRAW), 40, seed=11)
    ds = SpecDataset.frozen(specs, index, rcfg, slice_="train", fold=None)
    return model, ds, evaluate(model, ds, batch_size=8, fold=0)


def test_evaluate_goes_through_the_official_estimator(scored, monkeypatch):
    """Never recompute EER. Proven by breaking the official one and watching
    `evaluate` fail, rather than by comparing two numbers this module produced."""
    model, ds, _ = scored

    def refuse(*a, **kw):
        raise AssertionError("roc_curve was reached -- good")

    monkeypatch.setattr("metrics.dacon.roc_curve", refuse)
    with pytest.raises(AssertionError, match="roc_curve was reached"):
        evaluate(model, ds, batch_size=8, fold=0)


def test_the_loop_contains_no_roc_code_of_its_own():
    """The guard covers every module the scoring path was split across, or the
    split would have moved the code out from under it."""
    for module in ("training/loop.py", "training/validate.py",
                   "training/stages.py", "training/checkpoint.py"):
        source = Path(module).read_text()
        for forbidden in ("roc_curve", "roc_auc_score", "def eer("):
            assert forbidden not in source, (module, forbidden)


def test_the_report_carries_per_cell_and_per_generator_not_pooled_only(scored):
    """docs/validation: report per cell and per generator, never pooled only. A
    good pooled EER routinely hides a collapsed cell, and cells 6/7 are the whole
    reason the competition has two fake heads."""
    _, ds, report = scored
    assert isinstance(report, ValidationReport)
    drawn = {s.cell for s in ds.specs}
    assert set(report.per_cell[report.per_cell["head"] == "file"]["value"]) == drawn
    assert len(report.per_generator) > 0
    # Every slice records which contrast it used, so a shared-contrast row and
    # a within-slice row can never be silently averaged together.
    assert set(report.per_cell["contrast"]) <= {"shared", "within"}


def test_evaluate_refuses_a_redrawable_dataset(corpus, model_cfg):
    with pytest.raises(ValueError, match="frozen"):
        evaluate(_model(model_cfg), _dataset(corpus, n=4))


def test_prediction_truth_columns_come_from_the_cell(scored):
    _, ds, report = scored
    frame = report.predictions
    for spec, row in zip(ds.specs, frame.itertuples()):
        assert row.cell == spec.cell
        assert row.file_fake == spec.file_fake
        assert row.voice_present == spec.voice_present
        assert row.music_present == spec.music_present


def _preds_of(report, ids=None):
    """The mapping `predict` returns, rebuilt from a scored report's frame."""
    preds = {c: report.predictions[c].to_numpy() for c in PREDICTION_COLUMNS}
    preds["file_id"] = np.asarray(
        list(report.predictions["file_id"]) if ids is None else list(ids),
        dtype=object)
    return preds


def test_prediction_frame_refuses_a_mismatched_prediction_length(scored):
    _, ds, report = scored
    preds = {c: report.predictions[c].to_numpy()[:-1] for c in PREDICTION_COLUMNS}
    preds["file_id"] = report.predictions["file_id"].to_numpy()[:-1]
    with pytest.raises(ValueError, match="not in the eval set's order"):
        prediction_frame(ds.specs, preds, ds.index)


def test_a_permuted_prediction_pass_cannot_reach_the_frame(scored):
    """The join is positional, so a permutation silently scores every row
    against another row's truth. Measured on 64 rows before the ids travelled
    with the probabilities: reversing each eval batch moved `score` 0.4650 ->
    0.4919 and `eer_file` 0.6103 -> 0.4868, with every VG5 row green in both.

    The identity is what makes it visible, so the assertion is on the *verdict*
    of the join, not on any number the permutation happens to move.
    """
    _, ds, report = scored
    order = list(report.predictions["file_id"])

    # In order: accepted, and the frame it builds is the scored one.
    frame = prediction_frame(ds.specs, _preds_of(report), ds.index)
    assert list(frame["file_id"]) == order

    with pytest.raises(ValueError, match="not in the eval set's order"):
        prediction_frame(ds.specs, _preds_of(report, order[::-1]), ds.index)

    swapped = list(order)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match="first disagreement at row 0"):
        prediction_frame(ds.specs, _preds_of(report, swapped), ds.index)

    with pytest.raises(ValueError, match="no 'file_id'"):
        prediction_frame(ds.specs,
                         {c: report.predictions[c].to_numpy()
                          for c in PREDICTION_COLUMNS}, ds.index)


def test_evaluate_goes_red_when_the_eval_pass_reorders_its_batches(
        scored, monkeypatch):
    """The mutation the identity exists for: `predict` emitting its rows in
    another order than `dataset.specs`. Reversing the indices *within* each eval
    batch -- which `eval_batches` cannot do, but a rewrite of it could -- used to
    survive the whole suite with every gate green."""
    model, ds, _ = scored
    import training.validate as validate

    ordered = validate.eval_batches

    def reversed_batches(dataset, batch_size):
        for batch in ordered(dataset, batch_size):
            yield list(batch)[::-1]

    monkeypatch.setattr(validate, "eval_batches", reversed_batches)
    with pytest.raises(ValueError, match="not in the eval set's order"):
        evaluate(model, ds, batch_size=8, fold=0)


def test_generator_key_never_invents_a_family_for_a_real_row(corpus):
    """A real component carries no `artifact_family` at all. The key says
    `real:<source>` rather than borrowing one, because a borrowed family would
    put REAL rows into a generator's slice and make that slice's EER a fiction."""
    manifest, index, rcfg = corpus
    specs = list(Sampler(manifest, DRAW).epoch_specs(120, seed=3))
    families = set(manifest["artifact_family"].dropna().astype(str))
    for spec in specs:
        key = generator_key(spec, index)
        if spec.file_fake:
            assert not key.startswith("real:"), (spec.cell, key)
            assert set(key.split("+")) <= families
        else:
            assert key.startswith("real:") or key == "unknown", (spec.cell, key)


def _slices(rows):
    """A breakdown frame with `metrics.breakdown.by`'s columns, row by row."""
    return pd.DataFrame([
        {"value": v, "head": head, "eer": eer_, "contrast": "shared",
         "n_slice": 40 if thin else 300, "n_slice_fake": 20, "n_pool": 900,
         "thin": thin, "note": ""}
        for v, head, eer_, thin in rows])


def test_the_worst_slice_is_a_max_over_the_file_heads_usable_slices():
    """Every qualifier of the Tier-2 definition, one mutation each.

    P2 reads the worst *cell* and P3 the worst *family*, both on the file head
    and both over slices thick enough to mean anything. An unfiltered max
    reported a voice-head slice under a file-head name, a min reported the
    healthiest slice, and a thin slice is not evidence.
    """
    per_cell = _slices([(1, "file", 0.20, False), (2, "file", 0.44, False),
                        (3, "file", 0.98, True),        # thin: not evidence
                        (4, "voice", 0.91, False),      # another head entirely
                        (5, "file", float("nan"), False)])
    per_family = _slices([("hifigan", "file", 0.30, False),
                          ("suno_v3", "file", 0.51, False),
                          ("vits", "file", 0.99, True),
                          ("vits", "music", 0.95, False)])
    report = ValidationReport(0, _metric_set(0.10), per_cell, per_family,
                              pd.DataFrame({"file_id": ["s0"]}))

    assert report.worst_cell_eer == 0.44          # max, not 0.20 and not 0.98
    assert report.worst_family_eer == 0.51


def test_the_worst_slice_is_nan_when_nothing_is_usable():
    thin_only = _slices([(1, "file", 0.9, True), (2, "voice", 0.9, False)])
    report = ValidationReport(0, _metric_set(0.10), thin_only, thin_only,
                              pd.DataFrame({"file_id": ["s0"]}))
    assert np.isnan(report.worst_cell_eer)
    assert np.isnan(report.worst_family_eer)


def test_cells_and_families_are_scored_against_the_shared_contrast_pool(scored):
    """Why `_worst_slice` filters on the head and not on the contrast: both keys
    are label-determining, so their file-head rows can only be shared-contrast.
    A filter on a column that cannot vary would be a check nobody could see
    fail; this is the statement itself."""
    _, _, report = scored
    for table in (report.per_cell, report.per_generator):
        file_rows = table[table["head"] == "file"]
        assert len(file_rows)
        assert set(file_rows["contrast"]) == {"shared"}


# --------------------------------------------------------------------------- #
# 3. Aggregation


@pytest.mark.parametrize("seed", range(6))
def test_the_headline_is_the_mean_of_folds_and_pooling_raw_oof_is_wrong(seed):
    """Reproduces docs/validation/02 §4 at small scale.

    Five folds of one model with true EER 0.100, each fold's scores put through
    a *harmless monotone* rescale -- which changes no fold's own EER at all.
    Mean-of-folds recovers 0.100; pooling the raw scores does not, and the error
    is one-sided because the drift is independent of the label.
    """
    rng = np.random.default_rng(seed)
    per_fold, pooled = [], []
    for k in range(5):
        n = 600
        y = np.repeat([0, 1], n)
        s = np.concatenate([rng.normal(0.0, 1.0, n),
                            rng.normal(2.563, 1.0, n)])       # true EER ~= 0.10
        a, b = 1.0 + 0.8 * k, 3.0 * k                          # monotone per fold
        per_fold.append(eer(y, s))
        pooled.append(pd.DataFrame({"y": y, "s": a * s + b}))

    mean_of_folds = float(np.mean(per_fold))
    pooled_frame = pd.concat(pooled, ignore_index=True)
    pooled_eer = eer(pooled_frame["y"].to_numpy(), pooled_frame["s"].to_numpy())

    # The rescale is monotone *within* a fold, so it cannot move a per-fold EER.
    assert abs(mean_of_folds - 0.100) < 0.02, mean_of_folds
    # Pooled is worse, materially, and never better.
    assert pooled_eer > mean_of_folds + 0.03, (pooled_eer, mean_of_folds)


def test_aggregate_folds_uses_fold_mean_and_not_a_reimplementation():
    metrics = [_metric_set(0.10 + 0.01 * k) for k in range(5)]
    results = [_fold_result(k, m) for k, m in enumerate(metrics)]
    report = aggregate_folds(results)
    reference = fold_mean(metrics, fold_ids=list(range(5)))
    assert report.score_mean == reference.mean.score
    assert report.score_sd == reference.score_sd
    assert report.sd_is_a_measurement


def test_a_single_fold_run_is_first_class_but_says_its_sd_is_not_a_measurement():
    """The full 5-fold sweep is often unaffordable and Replay is fold 0 by
    definition. What a single fold does not give is `Score_sd`, which is P4's
    input and the primary-source E5 tiebreaker -- and `fold_mean` returns 0.0
    there, which
    reads as "perfectly stable" unless something says otherwise."""
    report = aggregate_folds([_fold_result(0, _metric_set(0.12))])
    assert report.aggregate.n_folds == 1
    assert report.score_sd == 0.0
    assert not report.sd_is_a_measurement
    assert any("nothing to vary" in c for c in report.caveats)
    assert report.quotable


def test_a_fold_below_the_pool_floor_is_excluded_and_recorded():
    big = [_fold_result(k, _metric_set(0.10, n=2000)) for k in range(4)]
    thin = _fold_result(4, _metric_set(0.40, n=200))
    report = aggregate_folds([*big, thin], min_pool=1200)
    assert report.aggregate.excluded_folds == (4,)
    assert any("min_pool" in c for c in report.caveats)
    assert abs(report.score_mean - aggregate_folds(big).score_mean) < 1e-12


def test_a_run_with_a_red_gate_is_not_quotable():
    red = AuditReport({"VG1_A1": (False, "a family is in two folds")})
    report = aggregate_folds([_fold_result(0, _metric_set(0.10), gates=red)])
    assert not report.quotable
    assert "NOT QUOTABLE" in str(report)
    assert report.as_ledger_row()["quotable"] is False


#: The columns of docs/validation/03 §5's schema this module is the source of.
#: The rest -- exp_id, git_sha, hypothesis, runtime, SHADOW -- belong to the
#: experiment runner.
LEDGER_COLUMNS = {
    "score_mean", "score_sd", "ads", "cps",
    "eer_file", "eer_voice", "eer_music", "auc_vp", "auc_mp",
    "per_cell_eer", "per_family_eer", "per_fold_score",
    "worst_cell_eer", "worst_family_eer",
    "vg1", "vg2", "vg3", "vg4", "vg5", "vg6",
}


def test_the_ledger_row_carries_every_schema_column_this_module_can_fill():
    """docs/validation/03 §5 names `vg1..vg6`, `per_cell_eer` and
    `per_family_eer`; none were emitted, though `per_fold_score` from the same
    schema line was. A ledger filter reads the row, not `__str__`."""
    row = aggregate_folds([_fold_result(0, _metric_set(0.10))]).as_ledger_row()
    assert LEDGER_COLUMNS <= set(row), LEDGER_COLUMNS - set(row)


def test_a_skipped_gate_reaches_the_ledger_as_na_and_never_as_pass():
    """The SKIP convention, in the column a machine reads. `quotable` cannot
    carry it -- VG3 skips on every run, so a SKIP that blocked would make
    nothing quotable -- so the tri-state is where the shortfall lands.

    Fail beats na beats pass: VG1 has one green sub-check and one skipped one
    here, and reporting it `pass` would be `quotable`'s problem one column on.
    """
    gates = AuditReport({
        "VG1_A1_family_in_one_cell": (True, "no family in two folds"),
        "VG1_A8A9_eval_size_floors": (True, AuditReport.SKIP + "no eval specs"),
        "VG3_adversarial_validation": (True, AuditReport.SKIP + "not implemented"),
        "VG4_corpus_identity": (True, "gate <= 0.10"),
        "VG5_B2_ranking_resolution": (False, "241 unique of 400"),
    })
    report = aggregate_folds([_fold_result(0, _metric_set(0.10), gates=gates)])
    status = report.gate_status()

    assert status["vg1"] == "na"          # one sub-check skipped: not a pass
    assert status["vg2"] == "na"          # never ran at all
    assert status["vg3"] == "na"
    assert status["vg4"] == "pass"
    assert status["vg5"] == "fail"        # red beats everything
    assert status["vg6"] == "na"
    row = report.as_ledger_row()
    assert {k: row[k] for k in status} == status


def test_the_ledger_reports_the_gates_that_actually_ran_as_pass():
    """The other direction: a gate that ran green must read `pass`, or `na`
    would be a synonym for "we do not fill this column" and stop meaning
    anything."""
    gates = AuditReport({f"VG{n}_x": (True, "green") for n in range(1, 7)})
    status = aggregate_folds(
        [_fold_result(0, _metric_set(0.10), gates=gates)]).gate_status()
    assert set(status.values()) == {"pass"}


def test_the_per_slice_ledger_columns_mean_over_folds_on_the_file_head():
    """`per_cell_eer` and `per_family_eer` use `worst_cell_eer`'s qualifiers --
    file head, thin excluded -- so a ledger row cannot list a cell under one
    definition and exclude it from the worst under another."""
    def report(fold, cell_eer):
        per_cell = _slices([(1, "file", cell_eer, False),
                            (2, "file", 0.30, False),
                            (3, "file", 0.99, True),
                            (1, "voice", 0.95, False)])
        per_family = _slices([("hifigan", "file", 0.40, False)])
        return _fold_result(fold, _metric_set(0.10), per_cell=per_cell,
                            per_family=per_family)

    row = aggregate_folds([report(0, 0.20), report(1, 0.40)]).as_ledger_row()
    assert json.loads(row["per_cell_eer"]) == pytest.approx({"1": 0.30, "2": 0.30})
    assert json.loads(row["per_family_eer"]) == pytest.approx({"hifigan": 0.40})
    assert row["worst_cell_eer"] == 0.40           # fold 1's cell 1
    assert row["worst_family_eer"] == 0.40


def test_a_fold_with_no_usable_slice_does_not_poison_the_worst_columns():
    """`max` over a sequence containing NaN answers whatever the fold order is."""
    empty = _slices([(1, "file", 0.9, True)])
    good = _slices([(1, "file", 0.35, False)])
    rows = [_fold_result(0, _metric_set(0.10), per_cell=empty, per_family=empty),
            _fold_result(1, _metric_set(0.10), per_cell=good, per_family=good)]
    for order in (rows, rows[::-1]):
        row = aggregate_folds(list(order)).as_ledger_row()
        assert row["worst_cell_eer"] == 0.35
        assert row["worst_family_eer"] == 0.35


def test_a_truncated_run_is_not_quotable(model_cfg):
    """Two docstrings used to contradict each other: `LoopConfig.max_steps` says
    a truncated run "is not quotable", `RunReport.quotable` said only a red gate
    voids a result, and the flag agreed with the second. The existing test
    asserted the caveat *string* contained "not quotable" and never looked at
    the column a Replay-vs-Full ledger filter reads.

    Asserted here on the flag, through the real `StageResult.caveats` rather
    than an invented string, because that wiring is the claim.
    """
    plan = stage_plan("joint", model_cfg)
    truncated = StageResult("joint", plan, steps=3, passes_done=1, truncated=True)
    full = StageResult("joint", plan, steps=3, passes_done=1)

    green = [_fold_result(0, _metric_set(0.10))]
    stopped = aggregate_folds(green, caveats=truncated.caveats)
    assert not stopped.quotable
    assert stopped.as_ledger_row()["quotable"] is False
    assert stopped.blocking_caveats and "truncated" in stopped.blocking_caveats[0]
    assert "NOT QUOTABLE" in str(stopped)
    # Every gate is still green: this is not a gate failure being renamed.
    assert set(stopped.gate_status().values()) == {"na"}

    assert aggregate_folds(green, caveats=full.caveats).quotable


def test_only_a_caveat_that_says_not_quotable_voids_the_run():
    """The marker is a phrase, so the boundary is worth pinning: an ordinary
    caveat -- the single-fold one, which every Replay run carries -- must not
    void a result, or `quotable` would go the way of the SKIP."""
    single = aggregate_folds([_fold_result(0, _metric_set(0.12))])
    assert single.caveats and single.quotable
    assert single.blocking_caveats == ()

    voided = aggregate_folds([_fold_result(0, _metric_set(0.12))],
                             caveats=[f"subsampled VAL: {NOT_QUOTABLE}"])
    assert not voided.quotable


def test_a_skipped_gate_does_not_block_but_is_reported():
    """VG3 is skipped on every run today. Treating a SKIP as a failure would
    make nothing quotable and the distinction would stop being read -- but a
    silent SKIP is the `I7` defect, which printed PASS for a check that existed
    nowhere."""
    skipped = AuditReport({"VG3": (True, AuditReport.SKIP + "not implemented")})
    report = aggregate_folds([_fold_result(0, _metric_set(0.10), gates=skipped)])
    assert report.quotable
    assert "VG3" in report.skipped_gates()
    assert "SKIP VG3" in str(report)
    assert "VG3" not in skipped.ran


# --------------------------------------------------------------------------- #
# 4. Leak tripwires


def test_the_music_tripwire_fires_below_three_percent():
    """Published cross-generator music detection is 46.4% EER. A local 0.5%
    on an unseen-generator split is evidence of a leak, not of success."""
    report = leak_tripwires(_metric_set(0.10, eer_music=0.005),
                            "generator_disjoint", "8 VAL families")
    passed, why = report.results["L1_music_unseen_generator"]
    assert not passed and "0.0050" in why


def test_the_voice_tripwire_fires_below_one_percent():
    report = leak_tripwires(_metric_set(0.10, eer_voice=0.004, eer_music=0.30),
                            "generator_disjoint", "")
    assert not report.results["L2_voice_unseen_generator"][0]
    assert report.results["L1_music_unseen_generator"][0]      # music is fine


@pytest.mark.parametrize("value,fires", [(0.0299, True), (0.0300, False),
                                         (0.0301, False)])
def test_the_music_threshold_is_asserted_on_the_quantity_it_names(value, fires):
    """On the EER itself, at the boundary. A test that asserted `0 < eer`
    would pass at every one of these values."""
    report = leak_tripwires(_metric_set(0.10, eer_music=value),
                            "generator_disjoint", "")
    assert report.results["L1_music_unseen_generator"][0] is not fires


def test_a_healthy_unseen_generator_result_passes():
    report = leak_tripwires(_metric_set(0.20, eer_voice=0.09, eer_music=0.31),
                            "generator_disjoint", "8 VAL families")
    assert report.ok
    assert "L1_music_unseen_generator" in report.ran      # ran, not skipped


def test_the_tripwires_skip_rather_than_pass_on_a_split_they_cannot_speak_about():
    """The `AuditReport.SKIP` convention. A tripwire reading PASS on a
    generator-overlapping split gives the run a green gate it did not earn."""
    report = leak_tripwires(_metric_set(0.10, eer_voice=0.0, eer_music=0.0),
                            "generator_overlapping", "3 families shared")
    for key in ("L1_music_unseen_generator", "L2_voice_unseen_generator"):
        assert key in report.skipped
        assert key not in report.ran
    # ... and the row that *does* apply to this split fires on those same numbers.
    assert not report.results["L3_perfect_separation"][0]


@pytest.mark.parametrize("kw,key", [({"eer_file": 0.0}, "eer_file"),
                                    ({"auc_vp": 1.0}, "auc_vp"),
                                    ({"auc_mp": 1.0}, "auc_mp")])
def test_the_file_and_presence_heads_are_guarded_on_a_disjoint_split(kw, key):
    """L1 reads `eer_music` and L2 reads `eer_voice`. Nothing read `eer_file` --
    the 0.45-weight head -- or either presence AUC, because L3 SKIPped on a
    generator-disjoint split saying "L1/L2 cover this one", which was false in
    three heads of five. Measured before the fix:
    `leak_tripwires(eer_file=0.0, auc_vp=1.0, auc_mp=1.0, "generator_disjoint")`
    returned `ok=True`.
    """
    healthy = {"eer_voice": 0.09, "eer_music": 0.31, "auc_vp": 0.9, "auc_mp": 0.9}
    report = leak_tripwires(_metric_set(**{**healthy, "eer_file": 0.2, **kw}),
                            "generator_disjoint", "8 VAL families")
    passed, why = report.results["L3_perfect_separation"]
    assert not passed, why
    assert key in why
    assert not report.ok
    # It RAN: a SKIP that happened to carry `False` would not be this row.
    assert "L3_perfect_separation" in report.ran


def test_l3_runs_rather_than_skipping_on_a_generator_disjoint_split():
    """The SKIP-vs-PASS distinction on the split a real run has. Removing the
    `AuditReport.SKIP` prefix from the old disjoint branch changed nothing that
    any test looked at -- so the branch is gone, and this pins that it is."""
    report = leak_tripwires(_metric_set(0.20, eer_voice=0.09, eer_music=0.31),
                            "generator_disjoint", "8 VAL families")
    assert "L3_perfect_separation" in report.ran
    assert "L3_perfect_separation" not in report.skipped
    assert report.results["L3_perfect_separation"][0]
    # And it says which heads it read, rather than claiming L1/L2 read them all.
    why = report.results["L3_perfect_separation"][1]
    assert "eer_file" in why and "eer_music" not in why.split("(L1/L2")[0]


def test_perfect_separation_on_an_overlapping_split_fires_on_every_head():
    for kw in ({"eer_file": 0.0}, {"eer_file": 0.2, "eer_voice": 0.0},
               {"eer_file": 0.2, "eer_music": 0.0},
               {"eer_file": 0.2, "auc_vp": 1.0}, {"eer_file": 0.2, "auc_mp": 1.0}):
        report = leak_tripwires(_metric_set(**{"eer_voice": 0.2, "eer_music": 0.2,
                                               "auc_vp": 0.9, "auc_mp": 0.9, **kw}),
                                "generator_overlapping", "")
        assert not report.results["L3_perfect_separation"][0], kw


def test_the_split_kind_is_measured_from_the_drawn_streams(tmp_path):
    """Measured, not declared. The caller who *thinks* the split is
    family-disjoint is exactly the caller whose 0.5% music EER needs explaining,
    so a `split_kind=` argument would be switched off by the same mistake the
    tripwires exist to catch."""
    manifest = synthetic_manifest(n_per_pool=240, n_whole_file=200,
                                  n_families=24, n_sources=8)
    index = ManifestIndex.from_frame(manifest)
    folds = build_folds(manifest, FoldConfig(n_folds=5)).frame
    resolved = fold_manifest(manifest, folds, fold=0)

    train = list(Sampler(resolved, DRAW, slice_="train").epoch_specs(600, seed=0))
    val = list(Sampler(resolved, DRAW, slice_="val").epoch_specs(600, seed=1))
    kind, detail = measured_split_kind(train, val, index)
    assert kind == "generator_disjoint", detail

    # The same corpus without the fold resolution: TRAIN and VAL share families.
    both = list(Sampler(manifest, DRAW).epoch_specs(600, seed=2))
    other = list(Sampler(manifest, DRAW).epoch_specs(600, seed=3))
    overlapping, why = measured_split_kind(both, other, index)
    assert overlapping == "generator_overlapping", why


def test_an_undecidable_split_is_undecidable_not_disjoint(corpus):
    """A stream with no generated component realises no families, and "disjoint
    from nothing" is vacuously true -- which would silently arm L1/L2."""
    manifest, index, rcfg = corpus
    real_only = [s for s in Sampler(manifest, DRAW).epoch_specs(200) if not s.file_fake]
    kind, _ = measured_split_kind(real_only, real_only, index)
    assert kind == "undecidable"
    report = leak_tripwires(_metric_set(0.0, eer_voice=0.0, eer_music=0.0), kind)
    assert "L1_music_unseen_generator" in report.skipped


# --------------------------------------------------------------------------- #
# 5. The gates


def test_vg5_catches_the_saturation_that_took_eer_from_0095_to_0302():
    """The documented failure, reproduced: saturating the top and bottom 40% of
    a column collapses ranking near the operating point with no other warning."""
    n = 1000
    rng = np.random.default_rng(0)
    clean = rng.random(n)
    saturated = np.clip(clean, 0.4, 0.6)
    saturated = np.where(saturated <= 0.4, 0.0, np.where(saturated >= 0.6, 1.0,
                                                         saturated))

    def frame(col):
        df = pd.DataFrame({"file_id": [f"s{i}" for i in range(n)]})
        for c in PREDICTION_COLUMNS:
            df[c] = col
        return df

    assert output_sanity(frame(clean)).results["VG5_B2_ranking_resolution"][0]
    assert not output_sanity(frame(saturated)).results["VG5_B2_ranking_resolution"][0]


def test_vg5_catches_a_constant_column_and_a_nan():
    n = 100
    df = pd.DataFrame({"file_id": [f"s{i}" for i in range(n)]})
    for c in PREDICTION_COLUMNS:
        df[c] = np.linspace(0.01, 0.99, n)
    assert output_sanity(df).ok

    constant = df.assign(FILE_FAKE_PROB=0.5)
    assert not output_sanity(constant).results["VG5_B3_no_constant_column"][0]

    nan = df.copy()
    nan.loc[0, "VOICE_FAKE_PROB"] = np.nan
    assert not output_sanity(nan).results["VG5_B5_no_fallback_nan"][0]
    out_of_range = df.assign(MUSIC_FAKE_PROB=df["MUSIC_FAKE_PROB"] * 2)
    assert not output_sanity(out_of_range).results["VG5_B1_finite_in_unit_interval"][0]


def test_vg5_b4_skips_rather_than_passes_without_a_reference():
    n = 20
    df = pd.DataFrame({"file_id": [f"s{i}" for i in range(n)]})
    for c in PREDICTION_COLUMNS:
        df[c] = np.linspace(0.01, 0.99, n)
    report = output_sanity(df)
    assert "VG5_B4_id_set_matches" in report.skipped
    assert output_sanity(df, list(df["file_id"])).results["VG5_B4_id_set_matches"][0]
    assert not output_sanity(df, list(df["file_id"])[::-1]
                             ).results["VG5_B4_id_set_matches"][0]


def test_vg1_is_wired_to_check_split_integrity_and_goes_red_on_a_broken_split(scored):
    """Not a re-implementation: `training.folds.check_split_integrity` is the
    gate, and this proves the wiring by breaking the table it reads."""
    _, _, report = scored
    manifest = synthetic_manifest(n_per_pool=240, n_whole_file=200,
                                  n_families=24, n_sources=8)
    folds = build_folds(manifest, FoldConfig(n_folds=5)).frame

    green = run_gates(report, folds=folds, scheme_version="synthetic-v1")
    assert green.results["VG1_A1_family_in_one_cell"][0], green

    broken = folds.copy()
    rotating = broken[(broken["slice"] == "train_val")
                      & broken["artifact_family"].notna()]
    fam = rotating["artifact_family"].iloc[0]
    rows = broken.index[broken["artifact_family"] == fam]
    assert len(rows) >= 2, "the fixture must have a family with rows to split"
    broken.loc[rows[: len(rows) // 2], "fold"] = \
        (int(broken.loc[rows[0], "fold"]) + 1) % 5
    red = run_gates(report, folds=broken, scheme_version="synthetic-v1")
    assert not red.results["VG1_A1_family_in_one_cell"][0]
    assert not red.ok


def test_vg1_a10_notices_a_scheme_version_mismatch(scored):
    _, _, report = scored
    manifest = synthetic_manifest(n_per_pool=240, n_whole_file=200,
                                  n_families=24, n_sources=8)
    folds = build_folds(manifest, FoldConfig(n_folds=5)).frame
    wrong = run_gates(report, folds=folds, scheme_version="something-else")
    keys = [k for k in wrong.results if "A10" in k]
    assert keys and not wrong.results[keys[0]][0]


def test_vg3_reports_skip_and_never_pass(scored):
    """The honest gap. A green stub would be worse than a SKIP: it would let a
    run claim a gate it never ran."""
    _, _, report = scored
    gates = run_gates(report)
    assert "VG3_adversarial_validation" in gates.skipped
    assert "VG3_adversarial_validation" not in gates.ran


def test_vg1_and_vg2_skip_without_the_inputs_they_need(scored):
    _, _, report = scored
    gates = run_gates(report)
    for key in ("VG1_split_integrity", "VG1_A8A9_eval_size_floors",
                "VG1_I5_split_safety", "VG2_shortcut_audit"):
        assert key in gates.skipped, key


def test_vg6_refuses_a_fourth_probe_opening(scored, tmp_path):
    """Enforced mechanically, not by discipline. The sealed slice is worthless
    once it has been optimized against."""
    _, _, report = scored
    log = tmp_path / "probe_openings.log"
    log.write_text("e1 first model\ne2 corpus freeze\ne3 final selection\n")
    assert run_gates(report, probe_log=log, opened_probe=True
                     ).results["VG6_probe_budget"][0]

    log.write_text(log.read_text() + "e4 one more look\n")
    assert not run_gates(report, probe_log=log, opened_probe=True
                         ).results["VG6_probe_budget"][0]
    # Opening PROBE with nowhere to record it is itself the failure.
    assert not run_gates(report, opened_probe=True).results["VG6_probe_budget"][0]
    # And a run that did not open PROBE skips rather than passes.
    assert "VG6_probe_budget" in run_gates(report).skipped


def test_vg4_uses_the_shared_t3_gap_implementation(scored):
    _, _, report = scored
    gates = run_gates(report)
    key = "VG4_corpus_identity"
    assert key in gates.results
    passed, why = gates.results[key]
    assert why.startswith(AuditReport.SKIP) or "gate <= 0.10" in why


# --------------------------------------------------------------------------- #
# 6. A fold, end to end


def test_validate_fold_cannot_produce_a_result_without_running_the_tripwires(
        corpus, model_cfg):
    """`train_specs` is required, so a `FoldResult` cannot exist without the
    split kind having been *measured* and the tripwires having run or said, by
    name, why they could not."""
    manifest, index, rcfg = corpus
    sampler = Sampler(manifest, DRAW)
    torch.manual_seed(0)
    model = _model(model_cfg)
    # `slice_` must name the slice the specs were DRAWN from, not the role
    # they are being used in. `SpecDataset.frozen` defaults to "val", and these
    # specs come from a `slice_="train"` sampler -- I5 caught exactly that
    # mislabelling here, which is the point of forwarding slice_/fold to it.
    eval_ds = SpecDataset.frozen(frozen_eval_specs(sampler, 24, seed=5), index,
                                 rcfg, slice_=sampler.slice_, fold=None)
    train_specs = list(sampler.epoch_specs(60, seed=0))

    result = validate_fold(model, eval_ds, train_specs=train_specs, fold=0)
    assert result.fold == 0
    assert set(result.tripwires.results) == {
        "L1_music_unseen_generator", "L2_voice_unseen_generator",
        "L3_perfect_separation"}
    # The gates that had no inputs skipped by name rather than passing quietly.
    assert "VG3_adversarial_validation" in result.gates.skipped
    assert "VG1_split_integrity" in result.gates.skipped
    # I5 SKIPs without a manifest and RUNS with one -- never a silent pass.
    assert "VG1_I5_split_safety" in result.gates.skipped
    with_manifest = validate_fold(model, eval_ds, train_specs=train_specs,
                                  fold=0, manifest=manifest)
    passed, why = with_manifest.gates.results["VG1_I5_split_safety"]
    assert "VG1_I5_split_safety" not in with_manifest.gates.skipped, why
    assert passed, why
    assert result.validation.per_cell is not None

    with pytest.raises(TypeError):
        validate_fold(model, eval_ds, fold=0)          # no train_specs
