"""``Sampler.epoch_specs`` -> ``render`` -> ``collate``, as a torch ``Dataset``.

Two modes, and the difference between them is the point of the module:

* **training** -- ``SpecDataset.from_sampler(...)``. The spec list is redrawn
  every epoch (``set_epoch``), because an epoch is a fixed count of drawn specs
  and ``epoch`` is part of the RNG key.
* **evaluation** -- ``SpecDataset.frozen(frozen_eval_specs(...))``. The spec list
  is a **value**, drawn once and carried, not a seed re-run later. Critical:
  that is first-class rather than a convenience: docs/pipelines/01 §1's whole
  argument for separating sampling from rendering is that the eval set can
  then be frozen as *specs*. A seed is not a frozen set -- it reproduces only
  against the same sampler, the same manifest and the same code, and every one
  of those changes during a competition.

Critical: **the fold is resolved before anything is drawn, never after.** A
composed sample's components can straddle folds, so "the fold of a composed
sample" is undefined -- see ``fold_manifest`` below.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
from torch.utils.data import Dataset

from processing.audit import audit_specs as processing_audit_specs
from processing.render import ManifestIndex, RenderConfig, RenderedSample, render
from processing.ship import ShipConfig
from training.audit import AuditReport, audit_specs
from training.collate import bucket_batches, spec_durations
from training.folds import apply_folds
from training.sampler import Sampler
from training.spec import SampleSpec

__all__ = [
    "UNSET", "SpecDataset", "eval_batches", "fold_manifest",
    "frozen_eval_specs", "training_batches",
]


class _Unset:
    """Sentinel type: "this argument was never passed", distinct from ``None``.

    Critical: needed because ``None`` is a *meaningful* value for ``fold`` --
    it says "the frame is fold-resolved, the clause is vacuous" -- so a plain
    ``None`` default would make "I decided" and "I never thought about it" the
    same argument, which is how the frozen eval set came to report a PASS it
    had not earned.

    Caveat: **do not "simplify" this to a required keyword or a plain
    default.** Both were tried. A required keyword forces every caller to type
    something, but a caller can still satisfy it with the value that happens to
    work and get a PASS they never reasoned about; and it cannot express
    "explicitly None" distinctly from "had to write something". A plain default
    is the original defect. The sentinel is the only one of the three that lets
    `audit_specs` *measure* which case it is in -- SKIP when nobody decided,
    PASS-with-reason when someone did and the clause is genuinely vacuous.
    """

    def __repr__(self) -> str:                                # pragma: no cover
        return "UNSET"


UNSET = _Unset()


def fold_manifest(manifest: pd.DataFrame, folds: pd.DataFrame, *,
                  fold: int) -> pd.DataFrame:
    """The manifest one fold's `Sampler` reads. ``fold`` is required, always.

    Critical: **this is where the straddling question is answered, and the
    answer is that it never gets asked.** A composed sample draws two or three
    component files, and nothing stops those files belonging to different
    families -- so "the fold of a composed sample" is undefined, and any rule
    that picked one (the first component's fold, the voice component's fold)
    would be an accident of draw order dressed up as a policy.

    Instead the fold is resolved **one level up, on the manifest**, before a
    single spec is drawn: ``apply_folds(..., fold=k)`` turns `train_val` into
    `train`/`val` per family, and the `Sampler` is then built on one slice of
    that resolved frame. Every component of every sample is in that slice
    because there is nothing else to draw from -- containment is structural, not
    checked after the fact.

    Caveat: it is still *measured*: `SpecDataset.audit` forwards `slice_` and
    `fold` to `audit_specs`, whose **I5** re-derives the allowed `file_id` set
    from the manifest and reports any drawn component outside it. Do not
    rebuild that check here (docs/pipelines/05 §5) -- pass the manifest and
    read I5.

    This is a thin wrapper on `apply_folds` and exists only to make the required
    `fold=` visible at the dataset boundary, where forgetting it would train on
    the validation set.
    """
    return apply_folds(manifest, folds, fold=fold)


def frozen_eval_specs(sampler: Sampler, n: int, *, epoch: int = 0,
                      seed: int = 0) -> tuple[SampleSpec, ...]:
    """Draw the evaluation set **once**, as a tuple of specs to be carried.

    Caveat: deliberately a `tuple`, and deliberately not a generator: an eval
    set that is regenerated per call is a seed, not a frozen set. Write it to
    `val_specs.parquet` (VG1 A8/A9 in `audit_specs(..., eval_floors=True)`) and
    the same rows are scored in fold 0 and in the 2nd-stage rerun.
    """
    return tuple(sampler.epoch_specs(n, epoch=epoch, seed=seed))


class SpecDataset(Dataset):
    """``index -> RenderedSample``. Holds specs, not audio.

    Caveat: ``__getitem__`` returns a `RenderedSample`, not a batch: batching
    is `training.collate.collate`, passed to the `DataLoader` as `collate_fn`.
    The two stay separate so the collator can be tested on hand-built samples
    that no sampler would draw (all-mono batches, a batch with no present
    component).
    """

    def __init__(self, specs: Sequence[SampleSpec],
                 index: ManifestIndex | pd.DataFrame,
                 cfg: RenderConfig | None = None, *,
                 slice_: str | None, fold: int | None,
                 sampler: Any | None = None, n: int | None = None,
                 seed: int = 0, epoch: int = 0, ship: ShipConfig | None = None):
        self._specs: tuple[SampleSpec, ...] = tuple(specs)
        self.index = ManifestIndex.coerce(index)
        # 06 P8: rendered by `processing.render` (a `training.render.RenderConfig`
        # is lifted into its config), shipped by `processing.ship` -- the ONE
        # chain train and test share. The loop and the validator read `ship`
        # from here, so a dataset is the whole path from spec to model input.
        self.cfg = RenderConfig.coerce(cfg)
        self.ship = ship or ShipConfig()
        self.slice_, self.fold = slice_, fold
        self._sampler, self._n = sampler, n
        self.seed, self.epoch = seed, epoch
        #: Whether a fold was named. `__init__` takes `fold` positionally in the
        #: keyword sense -- it is required there -- so anything built this way
        #: has stated one; `frozen` is where it can be omitted.
        self._fold_stated = True
        self._slice_stated = True
        if not self._specs:
            raise ValueError("SpecDataset needs at least one spec")

    # -- construction -------------------------------------------------------- #

    @classmethod
    def from_sampler(cls, sampler: Any, n: int,
                     index: ManifestIndex | pd.DataFrame,
                     cfg: RenderConfig | None = None, *,
                     epoch: int = 0, seed: int = 0, ship: ShipConfig | None = None
                     ) -> SpecDataset:
        """The training mode: a spec list that `set_epoch` redraws. ``sampler``
        is anything with ``epoch_specs``, ``slice_`` and ``fold`` -- the
        training sampler or ``processing.sampler.Sampler``."""
        return cls(sampler.epoch_specs(n, epoch=epoch, seed=seed), index, cfg,
                   slice_=sampler.slice_, fold=sampler.fold,
                   sampler=sampler, n=n, seed=seed, epoch=epoch, ship=ship)

    @classmethod
    def frozen(cls, specs: Sequence[SampleSpec],
               index: ManifestIndex | pd.DataFrame,
               cfg: RenderConfig | None = None, *,
               slice_: str | _Unset = UNSET,
               fold: int | None | _Unset = UNSET,
               ship: ShipConfig | None = None) -> SpecDataset:
        """The evaluation mode: this exact spec list, for as long as it lives.

        Critical: ``fold`` and ``slice_`` both distinguish **stated** from
        **omitted**, and that is the whole point. Each used to carry a default,
        and a review found the same consequence twice: the frozen evaluation
        set -- the instrument this project trusts over the leaderboard --
        audited with half of I5 inactive and reported PASS, in a module whose
        own convention is that a skipped check must never read as a pass.

        * ``fold=k`` -- the specs were drawn under fold ``k``; I5 checks both of
          its clauses.
        * ``fold=None`` **explicitly** -- "the frame is already fold-resolved
          (`fold_manifest`), so the clause has nothing to add". I5 passes and
          says the clause was vacuous.
        * ``fold`` omitted -- nobody decided. `audit` reports I5 as **SKIP**,
          whatever the manifest looks like. Not passing an argument is not an
          assertion.

        Caveat: ``slice_`` names where the specs were **drawn from**, not the
        role they are being used in, and that is exactly why its old ``"val"``
        default was a trap: an eval set drawn from a ``slice_="train"`` sampler
        inherited the label ``"val"``, and when `training.loop` wired I5 in
        properly it reported **23 file_ids outside the declared slice**. The
        specs were fine; the declaration was invented by a default. Pass
        ``sampler.slice_``.
        """
        fold_stated = not isinstance(fold, _Unset)
        slice_stated = not isinstance(slice_, _Unset)
        ds = cls(specs, index, cfg,
                 slice_=None if not slice_stated else slice_,
                 fold=None if not fold_stated else fold, ship=ship)
        ds._fold_stated = fold_stated
        ds._slice_stated = slice_stated
        return ds

    # -- the dataset protocol ------------------------------------------------ #

    @property
    def specs(self) -> tuple[SampleSpec, ...]:
        return self._specs

    @property
    def is_frozen(self) -> bool:
        return self._sampler is None

    @property
    def n_per_epoch(self) -> int | None:
        """How many specs `set_epoch` redraws, or None on a frozen set.

        Caveat: not `len(self)`: they agree today, but the count is what the
        *draw* is keyed on and a resume has to record it
        (`training.loop.SamplerState`). A frozen set has no such number -- its
        length is the whole story.
        """
        return self._n

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, i: int) -> RenderedSample:
        return render(self._specs[i], self.index, self.cfg)

    def set_epoch(self, epoch: int) -> None:
        """Redraw the epoch. Critical: refused on a frozen set, loudly.

        Caveat: the refusal is the feature. Calling `set_epoch` on the
        evaluation set would silently swap the rows under the validation curve,
        and the metric would keep going up while measuring a different set
        every epoch -- a failure no assertion downstream can see.
        """
        if self._sampler is None or self._n is None:
            raise RuntimeError(
                "this SpecDataset is frozen: its spec list is a value, not a "
                "seed. Re-draw with SpecDataset.from_sampler if you meant the "
                "training set")
        self.epoch = int(epoch)
        self._specs = tuple(self._sampler.epoch_specs(
            self._n, epoch=self.epoch, seed=self.seed))

    # -- the audits ---------------------------------------------------------- #

    def audit(self, manifest: pd.DataFrame | None = None, *,
              batches: Sequence[Sequence[int]] | None = None,
              **kw: Any) -> AuditReport:
        """`audit_specs` over this dataset's specs, with slice and fold wired in.

        Critical: forwarding `slice_`/`fold` is what makes **I5** run rather
        than SKIP. Called without a manifest it reports SKIP for I4, I5 and
        I21 -- which is the `AuditReport.SKIP` convention, never a pass.

        Critical: pass ``batches`` -- the plan from `training_batches` -- to
        audit **C2 on the batches the optimiser will actually see**.
        `audit_specs` otherwise cuts the stream in draw order, and M4 made the
        training order bucketed by duration, so draw-order slices are batches
        that never get stepped on. `training/audit.py`'s own note says "audit
        the same order you train in"; this is the argument that makes that
        possible.
        """
        # 06 P8 / 05 C5: with a manifest, the processing audit -- it collapses
        # the tiles of a slot into one draw (I3 read tiles as draws, 0.100
        # FAIL) and adds the draw-feature and presence probes. Without one,
        # the training audit, which SKIPs what needs the manifest.
        if manifest is not None:
            report = processing_audit_specs(self._specs, manifest, slice_=self.slice_,
                                            fold=self.fold, batches=batches, **kw)
        else:
            report = audit_specs(self._specs, manifest=None, slice_=self.slice_,
                                 fold=self.fold, batches=batches, **kw)
        if not self._slice_stated:
            # Critical: same rule as the fold below. With no slice named,
            # `audit_specs` already SKIPs I5 outright -- there is nothing to
            # compare against -- so this only has to make sure that stays a
            # SKIP and never becomes a quiet pass if the upstream branch is
            # ever loosened.
            passed, why = report.results["I5_split_safety"]
            if not why.startswith(AuditReport.SKIP):        # pragma: no cover
                report.results["I5_split_safety"] = (passed, AuditReport.SKIP + (
                    f"{why}; but no slice_ was ever passed to SpecDataset.frozen, "
                    f"so the provenance I5 checks against was never declared"))
        if not self._fold_stated:
            # Critical: no fold was ever named, so I5's second clause rests on
            # nothing a caller decided. Downgrade rather than report a PASS:
            # the `AuditReport.SKIP` convention exists because a review found
            # I7 printing PASS for a check that existed nowhere.
            passed, why = report.results["I5_split_safety"]
            if not why.startswith(AuditReport.SKIP):
                report.results["I5_split_safety"] = (passed, AuditReport.SKIP + (
                    f"{why}; but no fold was ever passed to SpecDataset.frozen, "
                    f"so the fold half of I5 rests on a default, not a decision"))
        return report


# --------------------------------------------------------------------------- #
# Batch index plans


def training_batches(dataset: SpecDataset, batch_size: int, *,
                     n_buckets: int = 4, seed: int = 0) -> list[list[int]]:
    """Duration-bucketed batches over a training dataset's specs.

    The bucketing key comes from the **specs**, so no audio is decoded to decide
    the batch plan (docs/pipelines/04 §3).
    """
    if dataset.is_frozen:
        raise ValueError(
            "refusing to bucket a frozen spec list: the eval set is unbucketed "
            "by construction, because the duration-vs-score check on the REAL "
            "class must be measured on unbucketed batches (docs/pipelines/04 §3)")
    return bucket_batches(spec_durations(dataset.specs), batch_size,
                          n_buckets=n_buckets, seed=seed)


def eval_batches(dataset: SpecDataset, batch_size: int) -> list[list[int]]:
    """Batches over the frozen eval set: **in order, unbucketed, nothing dropped**.

    Critical: unbucketed is not an oversight. docs/pipelines/04 §3 requires the
    duration-vs-score check on the REAL class (architecture/04 §6.1: `max`
    aggregation carries a **1.63** duration bias) to be measured on unbucketed
    batches, and the frozen eval list gives that for free -- as long as nobody
    buckets it. In order, so a run is comparable row by row with the previous
    one; nothing dropped, because every eval row must be scored.
    """
    if not dataset.is_frozen:
        raise ValueError(
            "refusing to build an eval plan over a training dataset: its spec "
            "list is redrawn by `set_epoch`, so the plan would silently start "
            "scoring a different set at the same length -- a validation curve "
            "whose rows change underneath it. Freeze it with "
            "SpecDataset.frozen(frozen_eval_specs(...))")
    n = len(dataset)
    return [list(range(start, min(start + batch_size, n)))
            for start in range(0, n, batch_size)]
