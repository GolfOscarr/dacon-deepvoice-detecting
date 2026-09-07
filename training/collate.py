"""``list[RenderedSample] -> batch``, and the decisions the batch layout forces.

The batch is exactly the five keys docs/pipelines/04 §2 fixes::

    wav              (B, C_max, S_max) float32, zero-padded
    lengths          (B,) int64 -- valid samples per row, never inferred
    targets          the five keys of losses.TARGET_FOR_COLUMN, each (B,) float32
    frame_intervals  list[dict], absolute seconds, rasterised late
    specs            list[SampleSpec], for the ledger and the spec-level audits

🔴 **Rule 2.4 governs everything here**: nothing a file's score depends on may
come from another file. Three violations have already shipped in this repo
(`bandpass` over the padded batch, `align_time` over padded frame counts,
`frame_max` padding-sensitive in the submitted probability), so every decision
below is written against that rule and tested against it in
``tests/test_collate.py``.

⚠️ **The collator does not touch the audio.** No downmix (that is
``prepare_waveform``'s job, at the same call site as inference), no bandpass, no
preprocess chain, no cast to the training precision -- float32 out, and the
model casts (docs/pipelines/04 §4: GeM overflowed in fp16 and produced NaN
attention, so the same batch must be replayable through an fp16 inference path).
No resampling and no time shift: steps 4-5 are time-invariant, ``frame_intervals``
are absolute seconds, and a collator that moved audio would desynchronise them
silently.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from models.losses import TARGET_FOR_COLUMN
from training.render import RenderedSample
from training.spec import SampleSpec

__all__ = [
    "BATCH_KEYS", "bucket_batches", "bucket_edges", "bucket_of", "collate",
    "padding_fraction", "promote_channels", "spec_durations",
]

#: The batch's keys, in the order docs/pipelines/04 §2 lists them. Exported so a
#: test can assert the contract rather than restate it.
BATCH_KEYS = ("wav", "lengths", "targets", "frame_intervals", "specs")


def promote_channels(wav: Tensor, channels: int) -> Tensor:
    """Bring a ``(C, S)`` row up to ``channels`` by **repeating its own channels**.

    🔴 This is the rule-2.4 surface of the batch layout, and the choice is
    deliberate. ``wav`` is ``(B, C, S_max)`` and rows carry 1 or 2 channels, so a
    batch that mixes them must agree on a channel axis -- and ``C_max`` is a
    property of the *other* rows. Two candidate rules:

    * **zero-fill** the missing channels. Rejected: under
      ``AudioConfig.channels="downmix"`` a mono row batched with a stereo one
      would be averaged against a channel of silence and come out at **half
      amplitude**, i.e. its score would depend on what shared its batch. That is
      the violation, not a rounding deviation.
    * **cyclic repeat** of the row's own channels. Chosen, because it is the
      identity at the model boundary: for every shipped policy
      (``models.audio.CHANNEL_POLICIES``) ``prepare_waveform`` of the promoted
      row is **bitwise** the un-promoted row -- ``mean(x, x) == x``,
      ``left(x, x) == x``, ``0.5*(x + x) == x``, all exact in float32 because the
      operations are multiplications by a power of two.

    ⚠️ So the invariance is not a property of *this* function alone: it is a
    joint property of the promotion and the channel policy, and a new policy
    could break it (a hypothetical ``side`` policy would read 0 from a promoted
    mono row rather than its true absence of a second channel). It is therefore
    **measured** rather than declared --
    ``tests/test_collate.py::test_promotion_is_invisible_at_the_model_boundary``
    enumerates ``CHANNEL_POLICIES`` from the source of truth, so adding a policy
    without checking it fails the suite instead of shipping.

    ⚠️ Also why the collator does not simply downmix: that would fork the channel
    policy into two places, disable the A-B3 channel augmentations, and make the
    ``mid_side`` leak test (09 B8) impossible to run.
    """
    have = int(wav.shape[0])
    if have == channels:
        return wav
    if have > channels:                                       # pragma: no cover
        raise ValueError(f"row has {have} channels, more than the batch's {channels}")
    reps = -(-channels // have)                               # ceil
    return wav.repeat(reps, 1)[:channels]


def collate(samples: Sequence[RenderedSample], *,
            pad_value: float = 0.0) -> dict[str, Any]:
    """The batch of docs/pipelines/04 §2. Zero-padded, float32, mandatory lengths.

    ``pad_value`` exists **for the tests**, and is a contract rather than a knob:
    docs/pipelines/04 §2 says the padding value must not matter, and the way to
    know that is to collate the same samples twice with different fillings and
    check the *submitted probability* is unchanged (I16). It defaults to 0.0 and
    training never passes it.

    🔴 ``lengths`` is mandatory and comes from each row's own sample count. It is
    what makes the frame mask correct; without it every padded frame counts as
    real audio and ``frame_max`` starts reading another file's padding.
    """
    if not samples:
        raise ValueError("collate needs at least one sample")

    rates = {s.sample_rate for s in samples}
    if len(rates) != 1:
        # A mixed-rate batch would make `lengths` mean different things per row,
        # and the frame grid downstream is derived from a single rate.
        raise ValueError(f"batch mixes sample rates {sorted(rates)}; "
                         f"P-S2 resamples every file at decode time")
    for s in samples:
        if s.wav.dim() != 2:
            raise ValueError(f"expected (C, S) per sample, got {tuple(s.wav.shape)}")

    n = len(samples)
    channels = max(int(s.wav.shape[0]) for s in samples)
    s_max = max(int(s.wav.shape[-1]) for s in samples)

    wav = torch.full((n, channels, s_max), float(pad_value), dtype=torch.float32)
    lengths = torch.empty(n, dtype=torch.int64)
    for i, sample in enumerate(samples):
        row = promote_channels(sample.wav.to(torch.float32), channels)
        take = int(row.shape[-1])
        wav[i, :, :take] = row
        lengths[i] = take

    targets = {
        key: torch.tensor([float(s.targets[key]) for s in samples],
                          dtype=torch.float32)
        for key in TARGET_FOR_COLUMN.values()
    }
    return {
        "wav": wav,
        "lengths": lengths,
        "targets": targets,
        # 🔴 Carried through untouched, in **absolute seconds**. Rasterising to a
        # frame grid here would hard-code a frame rate the collator does not
        # know -- the `align_time` defect, one layer up.
        "frame_intervals": [dict(s.frame_intervals) for s in samples],
        "specs": [s.spec for s in samples],
    }


# --------------------------------------------------------------------------- #
# Duration bucketing -- docs/pipelines/04 §3
#
# ✅ The objection that used to block this has been resolved away rather than
# solved: `pairwise_ranking_loss` formed positive/negative pairs *within* the
# batch, which would have made batch composition part of the objective. With
# `LossConfig.ranking_weight` committed at 0 and stage S4 dropped, no loss term
# is sensitive to batch composition and bucketing may be chosen purely for
# padding efficiency.
#
# 🔴 Two caveats survive and both are live. Bucketing must not fight **C2**: a
# duration bucket is still a batch and must meet the per-head present-count
# floor, and duration and cell are not independent (sequential compositions run
# long), so bucketing reshapes the per-batch cell mix. Bucketed batches are
# therefore auditable with `training.audit.audit_specs(..., batch_size=...)` --
# see `tests/test_collate.py::test_bucketing_does_not_starve_a_masked_head`,
# which sweeps seeds rather than trusting one. And the duration-vs-score check
# on the REAL class must be measured on **unbucketed** eval batches, which the
# frozen eval spec list is by construction (`training.dataset`).


def bucket_edges(durations: Sequence[float], n_buckets: int = 4) -> tuple[float, ...]:
    """Interior bucket boundaries, as quantiles of the epoch's own durations.

    Quantiles rather than equal width over ``[min_seconds, max_seconds]``: the
    draw is ``U(4, 60)`` today but the *bucket occupancies* are what decide both
    padding efficiency and how many batches each bucket yields, and equal-width
    edges over a non-uniform stream leave one bucket nearly empty -- which
    silently drops samples once `drop_last` bites.
    """
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
    d = np.asarray(durations, dtype=float)
    if d.size == 0:
        raise ValueError("bucket_edges needs at least one duration")
    if n_buckets == 1:
        return ()
    qs = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    return tuple(float(x) for x in np.quantile(d, qs))


def bucket_of(duration: float, edges: Sequence[float]) -> int:
    """Which bucket a duration falls in, given ``bucket_edges``' interior edges."""
    return int(np.searchsorted(np.asarray(edges, dtype=float), float(duration),
                               side="right"))


def bucket_batches(durations: Sequence[float], batch_size: int, *,
                   n_buckets: int = 4, seed: int = 0,
                   drop_last: bool = True) -> list[list[int]]:
    """Index lists, one per batch, grouped so a batch spans one duration bucket.

    ⚠️ ``drop_last`` defaults to True and matches what training does, which is
    also what `audit_specs`' C2 check assumes: C2's floor is absolute (the
    gradient norm scales as 1/√n), so a short trailing batch can never satisfy
    it and the optimiser never sees one.

    Deterministic in ``seed``: the same durations and seed give the same
    batches, in the same order, in every process.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    d = np.asarray(durations, dtype=float)
    edges = bucket_edges(d, n_buckets)
    rng = np.random.default_rng(seed)

    batches: list[list[int]] = []
    for b in range(n_buckets):
        idx = np.flatnonzero(np.array([bucket_of(x, edges) for x in d]) == b)
        rng.shuffle(idx)
        stop = len(idx) - (len(idx) % batch_size) if drop_last else len(idx)
        for start in range(0, stop, batch_size):
            batches.append([int(i) for i in idx[start:start + batch_size]])
    # 🔴 The batch *order* is shuffled too. Otherwise every epoch would walk the
    # short files first and the long ones last, so the optimiser would see a
    # duration schedule -- batch composition back in the objective by the side
    # door, which is exactly what §3 argues bucketing is free of.
    order = rng.permutation(len(batches))
    return [batches[int(i)] for i in order]


def padding_fraction(durations: Sequence[float],
                     batches: Sequence[Sequence[int]]) -> float:
    """Fraction of the batched tensor that is padding. The reason to bucket.

    Measured rather than asserted: `tests/test_collate.py` compares this against
    unbucketed batching over the same durations, so "bucketing helps" is a
    number in the suite and not a claim in a docstring.
    """
    d = np.asarray(durations, dtype=float)
    real = total = 0.0
    for batch in batches:
        if not len(batch):
            continue
        take = d[list(batch)]
        real += float(take.sum())
        total += float(take.max()) * len(batch)
    return 1.0 - real / total if total else 0.0


def spec_durations(specs: Sequence[SampleSpec]) -> list[float]:
    """The bucketing key, taken from the **spec** so no audio is decoded first.

    ⚠️ It is `spec.duration_s`, the drawn timeline, not the rendered length --
    bucketing has to happen before rendering or it buys nothing. The two agree
    to within resampler rounding, and `render` refuses a sample whose *rendered*
    length leaves the length regime (I12), so the approximation cannot drift.
    """
    return [float(s.duration_s) for s in specs]
