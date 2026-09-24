"""Scoring a fold, the leak tripwires and the VG gates -- docs/validation/02.

Roughly half of what `training.loop` used to be, and none of it depends on the
trainer: everything here takes a model and a frozen dataset, so the gates and
the tripwires can be imported and reasoned about without pulling in the stage
runner. `training.loop` re-exports every public name, so existing imports are
unchanged.

Three things are load-bearing and each is a decision, not plumbing:

Critical: **never recompute a metric.** `metrics/` is verified end to end and
three details of the official EER are load-bearing (docs/architecture/08 §5).
This module builds a prediction frame and hands it to `metrics.dacon.dacon_score`,
`metrics.breakdown` and `metrics.aggregate.fold_mean`. It contains no ROC code
and must not grow any.

Critical: **the headline is the mean of per-fold metrics.** Pooling raw OOF
scores across folds measured **0.1705 against a true 0.100** because each fold is
scored by a different model. `aggregate_folds` delegates to
`metrics.aggregate.fold_mean` and nothing here concatenates score columns.

Critical: **the leak tripwires belong here** (docs/pipelines/05 §6): music-fake
unseen-generator < 3% EER, voice-fake < 1%, or perfect separation on a random
split. They fail the run loudly rather than being read off a dashboard. And
"unseen generator" is **measured** from the drawn streams rather than declared --
`measured_split_kind` compares the realised generator sets, in the house style of
`training.registries`, which measures time invariance instead of trusting a
field.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from metrics.aggregate import AggregateMetrics, fold_mean
from metrics.breakdown import breakdown_table, t3_gap
from metrics.dacon import PREDICTION_COLUMNS, MetricSet, dacon_score
from models.model import DeepVoiceNet
from training.audit import AuditReport, audit_specs
from training.collate import collate
from training.dataset import SpecDataset, eval_batches
from training.folds import check_split_integrity
from processing.audit import audit_specs as processing_audit_specs
from processing.render import ManifestIndex, render
from processing.ship import ship
from training.spec import SampleSpec
from training.stages import autocast_for

__all__ = [
    "NOT_QUOTABLE", "FoldResult", "RunReport", "ValidationReport",
    "aggregate_folds", "eval_file_id", "evaluate", "generator_key",
    "leak_tripwires", "measured_split_kind", "output_sanity", "predict",
    "prediction_frame", "run_gates", "validate_fold",
]


# --------------------------------------------------------------------------- #
# 1. Scoring one fold -- docs/validation/02
# --------------------------------------------------------------------------- #

def generator_key(spec: SampleSpec, index: ManifestIndex) -> str:
    """The per-sample **generator** label the per-family breakdown slices on.

    A composed sample has no single provenance, so this is a rule and it is
    written down: the key is the artifact family of the sample's **fake**
    components (joined, for cell 8, which has two), or ``real:<source>`` when no
    component is generated -- a real component has no `artifact_family` at all
    (`training.manifest`: real pools carry none).

    Caveat: both sides are therefore label-determining, which is exactly the case
    `metrics.breakdown.by(..., contrast="auto")` handles by scoring against a
    shared contrast pool. Do not "fix" that by giving REAL rows a fake family.
    """
    fake, real = [], []
    for comp in spec.components:
        row = index[comp.file_id]
        family = row.get("artifact_family")
        source = row.get("source_name")
        if family is not None and not (isinstance(family, float) and math.isnan(family)):
            fake.append(str(family))
        elif source is not None and not (isinstance(source, float) and math.isnan(source)):
            real.append(str(source))
    if fake:
        return "+".join(sorted(set(fake)))
    return "real:" + "+".join(sorted(set(real))) if real else "unknown"


def _pair_id(spec: SampleSpec, index: ManifestIndex) -> Any:
    for comp in spec.components:
        pid = index[comp.file_id].get("pair_id")
        if pid is not None and not (isinstance(pid, float) and math.isnan(pid)):
            return str(pid)
    return None


def eval_file_id(spec: SampleSpec) -> str:
    """The eval row's identity, derived from the spec and from nothing else.

    Critical: one function, called by `predict` and by `prediction_frame`, is what
    makes the join between the two **checkable**. The frame pairs `preds[i]`
    with `specs[i]` positionally, so without an identity travelling alongside
    the probabilities a permutation of `predict`'s output is invisible: measured
    on 64 rows, reversing each eval batch moved `score` 0.4650 -> 0.4919 and
    `eer_file` 0.6103 -> 0.4868 with every gate still green.
    """
    return f"s{spec.sample_id:08d}"


def predict(model: DeepVoiceNet, dataset: SpecDataset, *,
            batch_size: int = 8, device: str | torch.device = "cpu",
            precision: str = "fp32") -> dict[str, np.ndarray]:
    """The five submission columns over a frozen eval set, in dataset order.

    Critical: the returned mapping also carries `file_id` -- `eval_file_id` of the
    spec each row was rendered from, taken from the *same* list the batch was
    built out of, so a reordering moves both. `prediction_frame` refuses a
    mapping whose ids are not the eval set's own order, which is the only thing
    standing between a silently permuted eval pass and a ledger row.

    Caveat: `eval_batches` -- in order, unbucketed, nothing dropped. Bucketing the
    eval set would destroy the duration-vs-score check on the REAL class that
    docs/pipelines/04 §3 requires be measured on unbucketed batches.

    Critical: ``precision`` defaults to **fp32 and should stay there**, even though
    inference ships fp16. The submitted number is a probability and bf16 carries
    8 mantissa bits, so squashing a bf16 logit **ties files together** — the
    mechanism behind EER 0.0950 -> 0.3017 (docs/validation/02 §7).

    Measured, and **the effect is size-dependent**, which is how it would
    have escaped a small fixture. Unit-scale logits, `n_unique` against VG5's
    `> 0.5 n` gate:

    | n | bf16 | fp16 | fp32 |
    |---|---|---|---|
    | 400 (a test fixture) | 241 PASS | 370 PASS | 400 PASS |
    | **1,200 (the VAL floor)** | **399 FAIL** | 996 PASS | 1,200 PASS |

    So a bf16 evaluation passes VG5 on any fixture small enough to be convenient
    and fails at the size we actually validate on. `tests/test_validate.py`
    asserts the *phenomenon* at 1,200 rather than at the fixture size, on a
    column it builds itself -- and separately asserts the **default** of all
    three entry points, which is the value that ships and which flipping to
    "bf16" once passed the entire suite.
    """
    device = torch.device(device)
    model.to(device).eval()
    columns: dict[str, list[np.ndarray]] = {c: [] for c in PREDICTION_COLUMNS}
    ids: list[str] = []
    with torch.no_grad():
        for indices in eval_batches(dataset, batch_size):
            specs = [dataset.specs[i] for i in indices]
            ids.extend(eval_file_id(s) for s in specs)
            batch = collate([render(s, dataset.index, dataset.cfg) for s in specs])
            # 06 P8: the shipped chain, the same call the loop and `script.py` make
            wav = ship(batch["wav"].to(device), dataset.ship, batch["lengths"].to(device))
            with autocast_for(precision, device):
                out = model(wav, batch["lengths"].to(device))
                probs = model.submission_probs(out)
            for c in PREDICTION_COLUMNS:
                columns[c].append(probs[c].detach().double().cpu().numpy())
    out_cols: dict[str, np.ndarray] = {c: np.concatenate(v) for c, v in columns.items()}
    out_cols["file_id"] = np.asarray(ids, dtype=object)
    return out_cols


def prediction_frame(specs: Sequence[SampleSpec], preds: Mapping[str, np.ndarray],
                     index: ManifestIndex, *, fold: int | None = None) -> pd.DataFrame:
    """Ground truth + predictions, one row per eval sample, ready for `metrics/`.

    Critical: the truth columns come from `spec.labels`, i.e. from the **cell**, and
    nowhere else -- the same derivation the metric harness uses. An absent
    component's fake label is written as `0`, not `None`, per `training.spec`:
    `file_fake_label` survives `None` only incidentally, and the masked pools
    drop those rows anyway.

    Critical: `preds` must carry `file_id`, and it is **checked against the specs
    row by row** rather than trusted. The pairing is positional -- `preds[i]`
    with `specs[i]` -- so a permuted prediction pass is otherwise a silent
    relabelling of the whole eval set: it moves the score without moving a
    single gate. `predict` emits the ids; a caller assembling `preds` by hand
    supplies them the same way, because "I built this in order" is exactly the
    claim that needs checking.
    """
    expected = [eval_file_id(spec) for spec in specs]
    if "file_id" not in preds:
        raise ValueError(
            "prediction_frame: preds carries no 'file_id', so the join to the "
            "specs would be positional and unchecked -- `predict` emits it, and "
            "`eval_file_id` builds it from a spec")
    got = [str(x) for x in np.asarray(preds["file_id"]).tolist()]
    if got != expected:
        first = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b),
                     min(len(got), len(expected)))
        raise ValueError(
            f"prediction_frame: preds are not in the eval set's order -- "
            f"{len(got)} predicted id(s) against {len(expected)} spec(s), first "
            f"disagreement at row {first}: {got[first:first + 1]} vs "
            f"{expected[first:first + 1]}. The join is positional, so this would "
            f"score every row against another row's truth")

    rows = []
    for spec, file_id in zip(specs, expected):
        labels = spec.labels
        rows.append({
            "file_id": file_id,
            "cell": spec.cell,
            "fold": fold,
            "artifact_family": generator_key(spec, index),
            "pair_id": _pair_id(spec, index),
            "duration_s": float(spec.duration_s),
            "voice_present": int(labels["voice_present"]),
            "music_present": int(labels["music_present"]),
            "voice_fake": int(labels["voice_fake"] or 0),
            "music_fake": int(labels["music_fake"] or 0),
            "file_fake": int(labels["file_fake"]),
        })
    frame = pd.DataFrame(rows)
    n = len(frame)
    for c in PREDICTION_COLUMNS:
        col = np.asarray(preds[c], dtype=np.float64)
        if col.shape != (n,):
            raise ValueError(f"{c}: got {col.shape}, expected ({n},) -- one row per spec")
        frame[c] = col
    return frame


def _worst_slice(table: pd.DataFrame) -> float:
    """The Tier-2 worst slice of a breakdown: **max**, file head, thin excluded.

    Critical: every qualifier is load-bearing and each one used to be unheld.
    `breakdown_table` emits three heads, so an unfiltered max reported the worst
    *voice*- or *music*-head slice under a name P2 and P3 read as the file head.
    Thin slices are excluded because a per-family EER over 40 files is not
    evidence (`metrics.breakdown`, same rule one layer down), and it is a max
    because the criterion is "no cell regresses" / "no family collapses" -- a
    min would report the healthiest slice and pass every run.

    Cells and families are label-determining keys, so their file-head rows are
    scored against the shared contrast pool by construction; `tests/test_validate.py`
    pins that rather than filtering on a column that cannot vary.
    """
    usable = table[(table["head"] == "file") & (~table["thin"])
                   & table["eer"].notna()]
    return float(usable["eer"].max()) if len(usable) else float("nan")


def _worst_over_folds(values: Any) -> float:
    """The worst fold's worst slice. Caveat: a fold with nothing usable is skipped,
    not propagated -- `max` over a sequence containing NaN answers whatever the
    fold order happens to be."""
    finite = [v for v in values if np.isfinite(v)]
    return max(finite) if finite else float("nan")


@dataclass(frozen=True)
class ValidationReport:
    """One fold's numbers. Pooled *and* sliced, because pooled alone is not a result.

    docs/validation/02 §1 puts `per_cell_eer` and `per_family_eer` in Tier 3 and
    the pooled Score in Tier 2, and a good pooled EER routinely hides a collapsed
    cell -- cells 6 and 7 are the entire reason the competition has two fake
    heads. The breakdowns are therefore fields of the report rather than
    something a caller may forget to ask for, and `__str__` prints them.
    """

    fold: int | None
    metrics: MetricSet
    per_cell: pd.DataFrame
    per_generator: pd.DataFrame
    predictions: pd.DataFrame

    @property
    def worst_cell_eer(self) -> float:
        """P2's input: the worst of cells 1-9 on the **file** head."""
        return _worst_slice(self.per_cell)

    @property
    def worst_family_eer(self) -> float:
        """P3's input -- "no family may collapse" (docs/validation/03 §3).

        Critical: `worst_cell_eer` had no family counterpart, so P3's Tier-2 input was
        the one promotion criterion nothing here could fill, although
        `per_generator` has held the data all along.
        """
        return _worst_slice(self.per_generator)

    def __str__(self) -> str:
        m = self.metrics
        head = (f"fold {self.fold}: score {m.score:.4f} "
                f"(file {m.eer_file:.4f} / voice {m.eer_voice:.4f} / "
                f"music {m.eer_music:.4f} / vp {m.auc_vp:.4f} / mp {m.auc_mp:.4f}) "
                f"over n_file={m.n_file} n_voice={m.n_voice} n_music={m.n_music}")
        cells = "\n".join(
            f"  cell {int(r.value):>2}  eer {r.eer:.4f}  [{r.contrast}] n={r.n_slice}"
            + ("  (thin)" if r.thin else "")
            for r in self.per_cell[self.per_cell["head"] == "file"].itertuples())
        gens = "\n".join(
            f"  {str(r.value):<28} eer {r.eer:.4f}  [{r.contrast}] n={r.n_slice}"
            + ("  (thin)" if r.thin else "")
            for r in self.per_generator[self.per_generator["head"] == "file"].itertuples())
        return f"{head}\nper cell (file head):\n{cells}\nper generator (file head):\n{gens}"


def evaluate(model: DeepVoiceNet, dataset: SpecDataset, *,
             batch_size: int = 8, device: str | torch.device = "cpu",
             precision: str = "fp32", fold: int | None = None) -> ValidationReport:
    """Score one fold. Calls `metrics/`; computes nothing itself.

    Critical: `dacon_score` raises on an empty or single-class masked pool rather than
    returning 0.5. That propagates: a fold whose music pool has one class has no
    music EER, and averaging a plausible-looking 0.5 into the headline is exactly
    the failure `metrics.dacon.EmptyPoolError` exists to prevent.
    """
    if not dataset.is_frozen:
        raise ValueError(
            "evaluate needs the frozen eval set (SpecDataset.frozen): a dataset "
            "that can redraw makes the validation curve measure a different set "
            "every epoch, which no downstream assertion can see")
    preds = predict(model, dataset, batch_size=batch_size, device=device,
                    precision=precision)
    frame = prediction_frame(dataset.specs, preds, dataset.index,
                             fold=dataset.fold if fold is None else fold)
    return ValidationReport(
        fold=fold if fold is not None else dataset.fold,
        metrics=dacon_score(frame),
        per_cell=breakdown_table(frame, keys=("cell",)),
        per_generator=breakdown_table(frame, keys=("artifact_family",)),
        predictions=frame)


# --------------------------------------------------------------------------- #
# 2. Leak tripwires -- docs/architecture/08 §5, docs/pipelines/05 §6
# --------------------------------------------------------------------------- #

#: If local CV shows better than these on an unseen-generator split, suspect a
#: leak rather than success. Published cross-generator music detection is 46.4%
#: EER and ASVspoof 5's best voice system is ~4% (docs/survey/10).
MUSIC_UNSEEN_FLOOR = 0.03
VOICE_UNSEEN_FLOOR = 0.01


def measured_split_kind(train_specs: Sequence[SampleSpec],
                        val_specs: Sequence[SampleSpec],
                        index: ManifestIndex) -> tuple[str, str]:
    """Is VAL generator-disjoint from TRAIN? **Measured, not declared.**

    Returns ``(kind, detail)`` with ``kind`` in
    ``{"generator_disjoint", "generator_overlapping", "undecidable"}``.

    Critical: the alternative -- a `split_kind=` argument the caller passes -- would let
    the tripwires be switched off by the same mistake they exist to catch: a
    builder that thought it had a family-disjoint split is precisely the one
    whose 0.5% music EER needs explaining. This compares the realised generator
    sets of the two drawn streams, in the style `training.registries` uses to
    measure time invariance instead of trusting a declaration field.
    """
    def families(specs):
        out = set()
        for spec in specs:
            key = generator_key(spec, index)
            if not key.startswith("real:") and key != "unknown":
                out.update(key.split("+"))
        return out

    tr, va = families(train_specs), families(val_specs)
    if not tr or not va:
        return "undecidable", (f"TRAIN has {len(tr)} and VAL {len(va)} realised "
                               "generator families; disjointness is vacuous")
    shared = sorted(tr & va)
    if shared:
        return "generator_overlapping", (
            f"{len(shared)}/{len(va)} VAL generator famil(y/ies) also appear in "
            f"TRAIN: {shared[:5]}")
    return "generator_disjoint", (f"{len(va)} VAL generator famil(y/ies), none of "
                                  f"TRAIN's {len(tr)}")


def leak_tripwires(metrics: MetricSet, split_kind: str, detail: str = "") -> AuditReport:
    """Numbers so good they are evidence of a leak. Fails the run, loudly.

    | Row | Head | Suspicious if | Because |
    |---|---|---|---|
    | L1 | music fake, unseen generator | **< 3% EER** | cross-generator is 46.4% |
    | L2 | voice fake, unseen generator | **< 1% EER** | ASVspoof 5's best is ~4% |
    | L3 | every head L1/L2 did not read | perfect separation | EER 0.0 is never real |

    Critical: L1 and L2 read `eer_music` and `eer_voice`. **L3 is the only row that
    ever reads `eer_file` -- the 0.45-weight head -- or the two presence AUCs**,
    so it runs on every split rather than SKIPping on the disjoint one, which is
    the split a real run has.

    Caveat: takes the **`MetricSet` the official harness already produced**, not a
    prediction frame: re-deriving these EERs here would be a second EER
    implementation in the one repo that forbids them, and the tripwire would
    then be able to disagree with the number it is guarding.

    Caveat: which rows run depends on what `measured_split_kind` found, and the others
    **SKIP** -- they never pass. A tripwire that reports PASS on a split it
    cannot speak about is worse than no tripwire, because the run then carries a
    green gate it did not earn (`AuditReport.SKIP`, `training.audit`).
    """
    r: dict[str, tuple[bool, str]] = {}
    disjoint = split_kind == "generator_disjoint"

    for head, value, floor, key in (
            ("music", metrics.eer_music, MUSIC_UNSEEN_FLOOR, "L1_music_unseen_generator"),
            ("voice", metrics.eer_voice, VOICE_UNSEEN_FLOOR, "L2_voice_unseen_generator")):
        if not disjoint:
            r[key] = (True, AuditReport.SKIP + f"split is {split_kind}: {detail}")
            continue
        r[key] = (
            not (np.isfinite(value) and value < floor),
            f"{head} EER {value:.4f} on an unseen-generator split, floor {floor:.2f} "
            f"-- below it, suspect a leak rather than success ({detail})")

    # EER 0 *and* AUC 1: the presence heads are AUCs, and a presence head that
    # separates perfectly is the same finding (docs/validation/03 §7 stops
    # investing there, which is a different decision from trusting the number).
    #
    # Critical: L3 **runs on every split**, and on a generator-disjoint one it is the
    # only row that reads `eer_file` -- the 0.45-weight head -- or either
    # presence AUC. It used to SKIP there saying "L1/L2 cover this one", which
    # was false in three of the five heads: L1 reads `eer_music`, L2 reads
    # `eer_voice`, and nothing read the other three. Measured:
    # `leak_tripwires(eer_file=0.0, auc_vp=1.0, auc_mp=1.0, "generator_disjoint")`
    # returned `ok=True`.
    separations = {"eer_file": metrics.eer_file <= 0.0,
                   "eer_voice": metrics.eer_voice <= 0.0,
                   "eer_music": metrics.eer_music <= 0.0,
                   "auc_vp": metrics.auc_vp >= 1.0,
                   "auc_mp": metrics.auc_mp >= 1.0}
    covered = ("eer_music", "eer_voice") if disjoint else ()
    examined = [k for k in separations if k not in covered]
    perfect = sorted(k for k in examined if separations[k])
    scope = (f"over {examined} (L1/L2 already read {list(covered)} on this split)"
             if covered else f"over {examined}")
    r["L3_perfect_separation"] = (
        not perfect,
        (f"head(s) perfectly separated on a {split_kind} split: {perfect} {scope} "
         f"-- suspect a leak, and on a random split re-split by generator "
         f"({detail})") if perfect else
        (f"no head separates perfectly on a {split_kind} split {scope} "
         f"(worst-case margin: file EER {metrics.eer_file:.4f}, "
         f"AUC_vp {metrics.auc_vp:.4f}, AUC_mp {metrics.auc_mp:.4f})"))
    return AuditReport(r)


# --------------------------------------------------------------------------- #
# 3. The gates -- docs/validation/04
# --------------------------------------------------------------------------- #

#: Critical: below this many rows, B2's `n_unique > 0.5 n` ratio cannot resolve a
#: **precision-driven** tie failure, because the probability that two files
#: collide grows with n while the gate does not. Measured on unit-scale logits:
#: a bf16 column gives 241 unique of 400 (**passes**) and 399 of 1,200
#: (**fails**). Derived, not picked -- 1,200 is VG1 A8's per-class VAL floor, so
#: it is both the size the gate was calibrated at and the size any run that
#: passes A8 already has.
#:
#: Caveat: it does **not** switch B2 off below the floor: a gross failure (a constant
#: column, the 80% saturation that took EER 0.0950 -> 0.3017) is visible at any
#: n. What it does is stop a small-n PASS reading as "ranking resolution
#: confirmed" -- **B2a** reports SKIP instead, which is the same shape as
#: `training.audit.RESOLVABLE_GAP`.
RESOLUTION_FLOOR = 1_200


def output_sanity(preds: pd.DataFrame, reference_ids: Sequence[str] | None = None
                  ) -> AuditReport:
    """**VG5** B1-B5: the five columns still carry ranking information.

    Critical: the guard is `n_unique > 0.5 n`, and it is not a formality. Saturating the
    top and bottom 80% of a column took EER 0.0950 -> **0.3017** with no other
    warning, and the usual fix -- rank-normalising the column -- is a cross-file
    statistic forbidden by rule 2.4. The way to satisfy it is float64 logits and
    a float64 squash with no rounding, which is what `models.outputs` does and
    what `predict`'s fp32 default protects.

    Critical: B2 is a **ratio** against whatever `n` the caller passes, and on its own
    that has the shape of the defect it exists to catch: bf16 passes it at
    n=400 and fails at n=1,200. **B2a** is the companion that says so -- below
    `RESOLUTION_FLOOR` it reports SKIP, so a small-n green B2 can never be read
    as ranking resolution confirmed.
    """
    r: dict[str, tuple[bool, str]] = {}
    n = len(preds)
    bad = []
    for c in PREDICTION_COLUMNS:
        v = preds[c].to_numpy(dtype=np.float64)
        if not np.isfinite(v).all() or v.min() < 0.0 or v.max() > 1.0:
            bad.append(c)
    r["VG5_B1_finite_in_unit_interval"] = (
        not bad, f"{len(bad)} column(s) non-finite or outside [0, 1]: {bad}")

    uniques = {c: int(pd.Series(preds[c]).nunique()) for c in PREDICTION_COLUMNS}
    worst = min(uniques, key=uniques.get)
    r["VG5_B2_ranking_resolution"] = (
        uniques[worst] > 0.5 * n,
        f"worst column {worst!r} has {uniques[worst]} unique values over {n} files "
        f"(gate > {0.5 * n:.0f}); ties near the operating point took EER "
        f"0.0950 -> 0.3017")
    r["VG5_B2a_resolution_is_resolvable"] = (
        True,
        (AuditReport.SKIP + f"n={n} is below the {RESOLUTION_FLOOR}-row floor, so "
         f"B2's ratio cannot resolve a precision-driven tie failure -- a bf16 "
         f"column passes B2 at n=400 and fails it at n=1200. B2's verdict above "
         f"still covers gross failures (a constant or saturated column). VG1 A8's "
         f"per-class floor is what makes B2 meaningful, and it is a separate gate")
        if n < RESOLUTION_FLOOR else
        f"n={n} >= {RESOLUTION_FLOOR}: B2's ratio can resolve a precision-driven "
        f"tie failure at this size, measured against bf16 (399/1200)")

    constant = [c for c, u in uniques.items() if u <= 1]
    r["VG5_B3_no_constant_column"] = (
        not constant, f"constant column(s): {constant}")

    if reference_ids is None:
        r["VG5_B4_id_set_matches"] = (
            True, AuditReport.SKIP + "no reference id set given")
    else:
        same = list(preds["file_id"]) == list(reference_ids)
        r["VG5_B4_id_set_matches"] = (
            same, f"{len(preds)} rows against {len(reference_ids)} reference ids, "
                  f"same order: {same}")
    r["VG5_B5_no_fallback_nan"] = (
        bool(preds[list(PREDICTION_COLUMNS)].notna().all().all()),
        "no NaN in any prediction column (the per-file fallback path must emit a "
        "finite value)")
    return AuditReport(r)


def run_gates(report: ValidationReport, *,
              folds: pd.DataFrame | None = None,
              eval_specs: Sequence[SampleSpec] | None = None,
              manifest: pd.DataFrame | None = None,
              slice_: str | None = None, fold: int | None = None,
              scheme_version: str | None = None,
              probe_log: Path | str | None = None,
              opened_probe: bool = False,
              reference_ids: Sequence[str] | None = None) -> AuditReport:
    """VG1-VG6 for one experiment. No number is quotable without this.

    Wires the gates that **exist** and SKIPs the rest by name, so the run record
    says which of the six actually ran:

    | Gate | Here |
    |---|---|
    | VG1 A1-A7, A10 | `training.folds.check_split_integrity` |
    | VG1 A8/A9 | `training.audit.audit_specs(..., eval_floors=True)` |
    | VG1 A1-A6 at draw time | the same audit's **I5**, which needs `slice_`/`fold` |
    | VG2 | the same audit's **I1b**, E-S2 at spec level |
    | VG3 adversarial validation | **SKIP** -- not implemented |
    | VG4 | `metrics.breakdown.t3_gap` |
    | VG5 | `output_sanity` |
    | VG6 | the `probe_openings.log` line count |

    Caveat: `reference_ids` is VG5 **B4**'s only input, and without the parameter
    B4 was permanently SKIP wherever VG5 is actually wired -- reachable in
    `output_sanity` and unreachable through the function that runs it. It is the
    id set the run must reproduce (`sample_submission.csv` at submission time).
    A fold has no such external reference, so `validate_fold` passes none and B4
    SKIPs there by construction: what the fold needs instead is that the
    predictions line up with the specs, and `prediction_frame` refuses the frame
    outright when they do not.

    Caveat: VG3 is the honest gap. It needs a TRAIN-vs-VAL classifier over the
    metadata features VG2 uses, which is real work and would land as a green
    stub if it were faked here. It reports SKIP.

    Critical: `RunReport.quotable` does **not** count SKIPs, and this docstring used
    to claim it did. VG3 skips on every run today, so a SKIP that blocked would
    make nothing quotable and the distinction would stop being read. The
    shortfall reaches the ledger instead, per gate and machine-readably, as
    `vg3=na` in `RunReport.as_ledger_row()` -- which is where a ledger filter
    can act on it, and is what the docstring was promising.
    """
    r: dict[str, tuple[bool, str]] = {}

    if folds is None:
        r["VG1_split_integrity"] = (True, AuditReport.SKIP + "no folds table given")
    else:
        for k, v in check_split_integrity(folds, scheme_version).results.items():
            r[f"VG1_{k}"] = v

    if eval_specs is None:
        for k in ("VG1_A8A9_eval_size_floors", "VG1_I5_split_safety",
                  "VG2_shortcut_audit"):
            r[k] = (True, AuditReport.SKIP + "no eval specs given")
    else:
        # `slice_`/`fold` are what make **I5** run rather than SKIP -- the
        # draw-time form of VG1 A1-A6, which re-derives the allowed `file_id`
        # set from the manifest and reports any drawn component outside it.
        # 06 P8: with a manifest, the processing audit (tiles collapsed into
        # draws, the draw-feature probes); training's audit otherwise
        if manifest is not None:
            spec_report = processing_audit_specs(list(eval_specs), manifest,
                                                 slice_=slice_, fold=fold, eval_floors=True)
        else:
            spec_report = audit_specs(list(eval_specs), manifest=None,
                                      slice_=slice_, fold=fold, eval_floors=True)
        r["VG1_A8A9_eval_size_floors"] = spec_report.results["I7_eval_size_floors"]
        r["VG1_I5_split_safety"] = spec_report.results["I5_split_safety"]
        r["VG2_shortcut_audit"] = spec_report.results["I1b_metadata_shortcut_auc"]

    r["VG3_adversarial_validation"] = (
        True, AuditReport.SKIP + "not implemented: a TRAIN-vs-VAL classifier over "
        "the VG2 metadata features. A low AUC would be weak evidence of "
        "absence anyway (docs/validation/04 VG3); a green stub would be none")

    gap = t3_gap(report.predictions, head="voice")
    if not np.isfinite(gap["t3_gap"]):
        r["VG4_corpus_identity"] = (
            True, AuditReport.SKIP + f"no scorable T3 pair pool "
                                     f"(n_pairs={gap['n_pairs']}): {gap['note']}")
    else:
        r["VG4_corpus_identity"] = (
            bool(gap["passes_vg4"]),
            f"voice T3-pair EER {gap['t3_pair_eer']:.4f} - pooled "
            f"{gap['pooled_eer']:.4f} = {gap['t3_gap']:+.4f}, gate <= 0.10 "
            f"over {gap['n_pairs']} paired rows")

    r.update(output_sanity(report.predictions, reference_ids).results)

    if not opened_probe:
        r["VG6_probe_budget"] = (
            True, AuditReport.SKIP + "this run did not score against PROBE")
    elif probe_log is None:
        r["VG6_probe_budget"] = (
            False, "PROBE was opened with no probe_openings.log to record it in; "
                   "the budget is enforced mechanically, not by discipline")
    else:
        lines = [ln for ln in Path(probe_log).read_text().splitlines() if ln.strip()]
        r["VG6_probe_budget"] = (
            len(lines) <= 3,
            f"{len(lines)} PROBE opening(s) recorded in {probe_log}, budget 3")
    return AuditReport(r)


# --------------------------------------------------------------------------- #
# 4. Folds, and the run record
# --------------------------------------------------------------------------- #

#: Critical: a caveat carrying this phrase **voids the run**: `RunReport.quotable` is
#: False whatever the gates say. It is how a condition the gates cannot see
#: reaches the flag -- today that is `LoopConfig.max_steps`, whose truncated run
#: is Replay speed and therefore a filter rather than evidence
#: (docs/validation/03 §2), and whose caveat already ends "is not quotable".
#:
#: Caveat: a phrase rather than a field because `caveats` is the channel that is
#: actually wired: `StageResult.caveats` documents itself as the thing to pass
#: into `aggregate_folds(..., caveats=...)`, and `training.validate` does not
#: import the trainer. A run whose caveat *says* it is not quotable and whose
#: `quotable` column says otherwise is the contradiction this removes.
NOT_QUOTABLE = "not quotable"


@dataclass(frozen=True)
class FoldResult:
    """One fold, end to end: what it scored and whether it is allowed to count."""

    fold: int | None
    validation: ValidationReport
    gates: AuditReport
    tripwires: AuditReport

    @property
    def ok(self) -> bool:
        return self.gates.ok and self.tripwires.ok


@dataclass(frozen=True)
class RunReport:
    """The ledger row. Mean of per-fold metrics, never pooled OOF.

    Caveat: **a single-fold run is first-class**, not a degraded mode: the full 5-fold
    sweep is often unaffordable and Replay speed is fold 0 only by definition
    (docs/validation/03 §2). What a single fold does *not* give is `Score_sd`,
    which is a P4 input and the primary-source E5 tiebreaker -- so `fold_mean`
    returns 0.0
    there and this report carries a caveat saying that 0.0 is an absence, not a
    measurement. Reading it as "perfectly stable" is the failure mode.
    """

    aggregate: AggregateMetrics
    folds: tuple[FoldResult, ...]
    caveats: tuple[str, ...] = ()

    @property
    def score_mean(self) -> float:
        return self.aggregate.mean.score

    @property
    def score_sd(self) -> float:
        return self.aggregate.score_sd

    @property
    def sd_is_a_measurement(self) -> bool:
        return self.aggregate.n_folds > 1

    @property
    def blocking_caveats(self) -> tuple[str, ...]:
        """The caveats that void the run, by the `NOT_QUOTABLE` convention."""
        return tuple(c for c in self.caveats if NOT_QUOTABLE in c)

    @property
    def quotable(self) -> bool:
        """Every gate green, no tripwire fired, and no caveat that voids the run.

        Caveat: SKIPs do not block -- they are recorded and printed. VG3 is skipped on
        every run today, so treating a SKIP as a failure would make nothing
        quotable and the distinction would stop being read. Where the shortfall
        does land is `gate_status()`, which reports a gate with any skipped
        sub-check as `na` rather than `pass`, in the ledger row a filter reads.

        Critical: the gates are not the only way to lose the right to quote a number.
        A **truncated** run is not quotable however green it is -- it is Replay
        speed, and Replay is a filter rather than evidence (docs/validation/03
        §2) -- and truncation is a fact about the trainer, which this module
        deliberately does not import. `NOT_QUOTABLE` is the channel: it travels
        on `caveats`, the one `StageResult.caveats` is already documented to be
        passed through, and it is read here rather than left to a reader.
        """
        return all(f.ok for f in self.folds) and not self.blocking_caveats

    def skipped_gates(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for f in self.folds:
            out.update(f.gates.skipped)
            out.update(f.tripwires.skipped)
        return out

    def gate_status(self) -> dict[str, str]:
        """`vg1..vg6` as docs/validation/03 §5's ``pass|fail|na``, over all folds.

        Critical: **fail beats na beats pass**, so a gate is `pass` only when every one
        of its sub-checks ran and passed in every fold. VG1 is wired to nine of
        them and any one skipping leaves it `na` -- which is the honest answer
        and the whole reason the ledger schema has a third state. Anything that
        rounded `na` up to `pass` would be `quotable` all over again, one column
        further on.
        """
        out: dict[str, str] = {}
        for n in range(1, 7):
            rows = [v for f in self.folds for k, v in f.gates.results.items()
                    if k.startswith(f"VG{n}_")]
            if any(not passed for passed, _ in rows):
                out[f"vg{n}"] = "fail"
            elif not rows or any(why.startswith(AuditReport.SKIP) for _, why in rows):
                out[f"vg{n}"] = "na"
            else:
                out[f"vg{n}"] = "pass"
        return out

    def _slice_eers(self, attr: str) -> dict[str, float]:
        """One breakdown's file-head EERs, meaned over the folds that have them.

        Caveat: the same qualifiers as `_worst_slice`, deliberately -- a ledger whose
        `per_cell_eer` listed a cell that `worst_cell_eer` had excluded as thin
        would be two different definitions of "the per-cell EER" in one row.
        """
        acc: dict[str, list[float]] = {}
        for f in self.folds:
            table = getattr(f.validation, attr)
            usable = table[(table["head"] == "file") & (~table["thin"])
                           & table["eer"].notna()]
            for row in usable.itertuples():
                acc.setdefault(str(row.value), []).append(float(row.eer))
        return {k: float(np.mean(v)) for k, v in sorted(acc.items())}

    def as_ledger_row(self) -> dict[str, Any]:
        """The subset of docs/validation/03 §5's schema this module can fill.

        Critical: the `vg1..vg6` tri-state is the machine-readable half of the SKIP
        convention. `__str__` has always printed the SKIP lines, but a ledger
        filter reads the row, not the printout -- so a run with six skipped
        gates used to be a row of numbers with nothing saying they were
        unguarded.
        """
        m = self.aggregate.mean
        return {
            "score_mean": m.score, "score_sd": self.score_sd,
            "ads": m.ads, "cps": m.cps,
            "eer_file": m.eer_file, "eer_voice": m.eer_voice,
            "eer_music": m.eer_music, "auc_vp": m.auc_vp, "auc_mp": m.auc_mp,
            "n_folds": self.aggregate.n_folds,
            "excluded_folds": json.dumps(list(self.aggregate.excluded_folds)),
            "per_fold_score": json.dumps(
                {str(f.fold): f.validation.metrics.score for f in self.folds}),
            "per_cell_eer": json.dumps(self._slice_eers("per_cell")),
            "per_family_eer": json.dumps(self._slice_eers("per_generator")),
            "worst_cell_eer": _worst_over_folds(
                f.validation.worst_cell_eer for f in self.folds),
            "worst_family_eer": _worst_over_folds(
                f.validation.worst_family_eer for f in self.folds),
            **self.gate_status(),
            "quotable": self.quotable,
            "caveats": json.dumps(list(self.caveats)),
        }

    def __str__(self) -> str:
        lines = [f"Score {self.score_mean:.4f} ± {self.score_sd:.4f} "
                 f"over {self.aggregate.n_folds} fold(s) "
                 f"[{'quotable' if self.quotable else 'NOT QUOTABLE'}]"]
        lines += [str(f.validation) for f in self.folds]
        for f in self.folds:
            for key, why in {**f.gates.failures, **f.tripwires.failures}.items():
                lines.append(f"  FAIL {key}: {why}")
        lines += [f"  SKIP {k}" for k in sorted(self.skipped_gates())]
        lines += [f"  caveat: {c}" for c in self.caveats]
        return "\n".join(lines)


def validate_fold(model: DeepVoiceNet, eval_dataset: SpecDataset, *,
                  train_specs: Sequence[SampleSpec],
                  fold: int | None = None, batch_size: int = 8,
                  device: str | torch.device = "cpu", precision: str = "fp32",
                  folds: pd.DataFrame | None = None,
                  manifest: pd.DataFrame | None = None,
                  scheme_version: str | None = None,
                  probe_log: Path | str | None = None,
                  opened_probe: bool = False) -> FoldResult:
    """Score one fold **with its gates and its tripwires**, as one object.

    Critical: ``train_specs`` is a required argument, and that is the point of this
    function existing at all. The tripwires need to know whether VAL is
    generator-disjoint from TRAIN, that question is **measured** rather than
    declared (`measured_split_kind`), and measuring it needs both streams. Making
    it required means a `FoldResult` cannot be produced without the tripwires
    having run or having said, by name, why they could not.

    Caveat: everything after `precision` is a gate input, and each one that is left
    out makes its gate report **SKIP** rather than PASS -- visible in
    `RunReport.skipped_gates()` and printed by `RunReport.__str__`.
    """
    report = evaluate(model, eval_dataset, batch_size=batch_size, device=device,
                      precision=precision, fold=fold)
    kind, detail = measured_split_kind(train_specs, eval_dataset.specs,
                                       eval_dataset.index)
    return FoldResult(
        fold=report.fold,
        validation=report,
        gates=run_gates(report, folds=folds, eval_specs=eval_dataset.specs,
                        manifest=manifest, slice_=eval_dataset.slice_,
                        fold=eval_dataset.fold, scheme_version=scheme_version,
                        probe_log=probe_log, opened_probe=opened_probe),
        tripwires=leak_tripwires(report.metrics, kind, detail))


def aggregate_folds(results: Sequence[FoldResult],
                    min_pool: int | None = None,
                    caveats: Sequence[str] = ()) -> RunReport:
    """Mean of per-fold metrics. Never a pooled OOF score.

    Caveat: ``caveats`` is where `StageResult.caveats` goes -- the frozen-frontend
    override, a truncated run. They are printed by `RunReport.__str__` and land
    in `as_ledger_row()`, because a caveat that stops at the training result is
    one nobody reads. One of them is more than a note: a caveat containing
    `NOT_QUOTABLE` voids the run, which is how a truncated (Replay-speed) stage
    reaches the `quotable` column rather than only the prose beside it.

    Each fold is scored by a *different model*, so their score scales differ and
    EER is computed on the merged ranking: concatenating raw OOF scores measured
    **0.1705 against a true 0.100** (docs/validation/02 §4). The aggregation is
    `metrics.aggregate.fold_mean`, which also records any fold excluded for
    falling below the masked-pool size floor rather than dropping it silently.
    """
    if not results:
        raise ValueError("aggregate_folds: no folds given")
    agg = fold_mean([r.validation.metrics for r in results],
                    min_pool=min_pool,
                    fold_ids=[r.fold for r in results])
    caveats: list[str] = list(caveats)
    if agg.n_folds == 1:
        caveats.append(
            "single fold: Score_sd is 0.0 because there is nothing to vary, not "
            "because the run is stable. P4 (fold variance) and the "
            "primary-source E5 "
            "tiebreaker cannot be evaluated from this run")
    if agg.excluded_folds:
        caveats.append(
            f"fold(s) {list(agg.excluded_folds)} fell below min_pool={min_pool} "
            "and were excluded from the mean (VG1 A8 should have caught it first)")
    for r in results:
        if not r.gates.ok:
            caveats.append(f"fold {r.fold}: {len(r.gates.failures)} gate(s) red — "
                           "the number is not quotable, comparable or promotable")
        if not r.tripwires.ok:
            caveats.append(f"fold {r.fold}: a leak tripwire fired — "
                           "suspect the split, not the model")
    return RunReport(agg, tuple(results), tuple(caveats))
