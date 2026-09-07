"""``Sampler.epoch_specs`` -> ``render`` -> ``collate``, as a torch ``Dataset``.

Two modes, and the difference between them is the point of the module:

* **training** -- ``SpecDataset.from_sampler(...)``. The spec list is redrawn
  every epoch (``set_epoch``), because an epoch is a fixed count of drawn specs
  and ``epoch`` is part of the RNG key.
* **evaluation** -- ``SpecDataset.frozen(frozen_eval_specs(...))``. The spec list
  is a **value**, drawn once and carried, not a seed re-run later. 🔴 That is
  first-class rather than a convenience: docs/pipelines/01 §1's whole argument
  for separating sampling from rendering is that the eval set can then be frozen
  as *specs*. A seed is not a frozen set -- it reproduces only against the same
  sampler, the same manifest and the same code, and every one of those changes
  during a competition.

🔴 **The fold is resolved before anything is drawn, never after.** A composed
sample's components can straddle folds, so "the fold of a composed sample" is
undefined -- see ``fold_manifest`` below.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
from torch.utils.data import Dataset

from training.audit import AuditReport, audit_specs
from training.collate import bucket_batches, spec_durations
from training.folds import apply_folds
from training.render import ManifestIndex, RenderConfig, RenderedSample, render
from training.sampler import Sampler
from training.spec import SampleSpec

__all__ = [
    "SpecDataset", "eval_batches", "fold_manifest", "frozen_eval_specs",
    "training_batches",
]


def fold_manifest(manifest: pd.DataFrame, folds: pd.DataFrame, *,
                  fold: int) -> pd.DataFrame:
    """The manifest one fold's `Sampler` reads. ``fold`` is required, always.

    🔴 **This is where the straddling question is answered, and the answer is
    that it never gets asked.** A composed sample draws two or three component
    files, and nothing stops those files belonging to different families -- so
    "the fold of a composed sample" is undefined, and any rule that picked one
    (the first component's fold, the voice component's fold) would be an
    accident of draw order dressed up as a policy.

    Instead the fold is resolved **one level up, on the manifest**, before a
    single spec is drawn: ``apply_folds(..., fold=k)`` turns `train_val` into
    `train`/`val` per family, and the `Sampler` is then built on one slice of
    that resolved frame. Every component of every sample is in that slice
    because there is nothing else to draw from -- containment is structural, not
    checked after the fact.

    ⚠️ It is still *measured*: `SpecDataset.audit` forwards `slice_` and `fold`
    to `audit_specs`, whose **I5** re-derives the allowed `file_id` set from the
    manifest and reports any drawn component outside it. Do not rebuild that
    check here (docs/pipelines/05 §5) -- pass the manifest and read I5.

    This is a thin wrapper on `apply_folds` and exists only to make the required
    `fold=` visible at the dataset boundary, where forgetting it would train on
    the validation set.
    """
    return apply_folds(manifest, folds, fold=fold)


def frozen_eval_specs(sampler: Sampler, n: int, *, epoch: int = 0,
                      seed: int = 0) -> tuple[SampleSpec, ...]:
    """Draw the evaluation set **once**, as a tuple of specs to be carried.

    ⚠️ Deliberately a `tuple`, and deliberately not a generator: an eval set that
    is regenerated per call is a seed, not a frozen set. Write it to
    `val_specs.parquet` (VG1 A8/A9 in `audit_specs(..., eval_floors=True)`) and
    the same rows are scored in fold 0 and in the 2nd-stage rerun.
    """
    return tuple(sampler.epoch_specs(n, epoch=epoch, seed=seed))


class SpecDataset(Dataset):
    """``index -> RenderedSample``. Holds specs, not audio.

    ⚠️ ``__getitem__`` returns a `RenderedSample`, not a batch: batching is
    `training.collate.collate`, passed to the `DataLoader` as `collate_fn`. The
    two stay separate so the collator can be tested on hand-built samples that
    no sampler would draw (all-mono batches, a batch with no present component).
    """

    def __init__(self, specs: Sequence[SampleSpec],
                 index: ManifestIndex | pd.DataFrame,
                 cfg: RenderConfig | None = None, *,
                 slice_: str, fold: int | None,
                 sampler: Sampler | None = None, n: int | None = None,
                 seed: int = 0, epoch: int = 0):
        self._specs: tuple[SampleSpec, ...] = tuple(specs)
        self.index = ManifestIndex.coerce(index)
        self.cfg = cfg or RenderConfig()
        self.slice_, self.fold = slice_, fold
        self._sampler, self._n = sampler, n
        self.seed, self.epoch = seed, epoch
        if not self._specs:
            raise ValueError("SpecDataset needs at least one spec")

    # -- construction -------------------------------------------------------- #

    @classmethod
    def from_sampler(cls, sampler: Sampler, n: int,
                     index: ManifestIndex | pd.DataFrame,
                     cfg: RenderConfig | None = None, *,
                     epoch: int = 0, seed: int = 0) -> SpecDataset:
        """The training mode: a spec list that `set_epoch` redraws."""
        return cls(sampler.epoch_specs(n, epoch=epoch, seed=seed), index, cfg,
                   slice_=sampler.slice_, fold=sampler.fold,
                   sampler=sampler, n=n, seed=seed, epoch=epoch)

    @classmethod
    def frozen(cls, specs: Sequence[SampleSpec],
               index: ManifestIndex | pd.DataFrame,
               cfg: RenderConfig | None = None, *,
               slice_: str = "val", fold: int | None = None) -> SpecDataset:
        """The evaluation mode: this exact spec list, for as long as it lives."""
        return cls(specs, index, cfg, slice_=slice_, fold=fold)

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

        ⚠️ Not `len(self)`: they agree today, but the count is what the *draw* is
        keyed on and a resume has to record it (`training.loop.SamplerState`).
        A frozen set has no such number -- its length is the whole story.
        """
        return self._n

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, i: int) -> RenderedSample:
        return render(self._specs[i], self.index, self.cfg)

    def set_epoch(self, epoch: int) -> None:
        """Redraw the epoch. 🔴 Refused on a frozen set, loudly.

        ⚠️ The refusal is the feature. Calling `set_epoch` on the evaluation set
        would silently swap the rows under the validation curve, and the metric
        would keep going up while measuring a different set every epoch -- a
        failure no assertion downstream can see.
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

    def audit(self, manifest: pd.DataFrame | None = None, **kw: Any) -> AuditReport:
        """`audit_specs` over this dataset's specs, with slice and fold wired in.

        🔴 Forwarding `slice_`/`fold` is what makes **I5** run rather than SKIP.
        Called without a manifest it reports SKIP for I4, I5 and I21 -- which is
        the `AuditReport.SKIP` convention, never a pass.
        """
        kw.setdefault("batch_size", 32)
        return audit_specs(self._specs, manifest=manifest,
                           slice_=self.slice_, fold=self.fold, **kw)


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

    🔴 Unbucketed is not an oversight. docs/pipelines/04 §3 requires the
    duration-vs-score check on the REAL class (architecture/04 §6.1: `max`
    aggregation carries a **1.63** duration bias) to be measured on unbucketed
    batches, and the frozen eval list gives that for free -- as long as nobody
    buckets it. In order, so a run is comparable row by row with the previous
    one; nothing dropped, because every eval row must be scored.
    """
    n = len(dataset)
    return [list(range(start, min(start + batch_size, n)))
            for start in range(0, n, batch_size)]
