"""``list[RenderedSample] -> batch``, and the decisions the batch layout forces.

The batch is exactly the five keys docs/pipelines/04 §2 fixes::

    wav              (B, C_max, S_max) float32, zero-padded
    lengths          (B,) int64 -- valid samples per row, never inferred
    targets          the five keys of losses.TARGET_FOR_COLUMN, each (B,) float32
    frame_intervals  list[dict], absolute seconds, rasterised late
    specs            list[SampleSpec], for the ledger and the spec-level audits

Critical: **Rule 2.4 governs everything here**: nothing a file's score depends
on may come from another file. Three violations have already shipped in this
repo (`bandpass` over the padded batch, `align_time` over padded frame counts,
`frame_max` padding-sensitive in the submitted probability), so every decision
below is written against that rule and tested against it in
``tests/test_collate.py``.

Caveat: **the collator does not touch the audio.** No downmix (that is
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

    Critical: this is the rule-2.4 surface of the batch layout, and the choice
    is deliberate. ``wav`` is ``(B, C, S_max)`` and rows carry 1 or 2 channels,
    so a batch that mixes them must agree on a channel axis -- and ``C_max`` is
    a property of the *other* rows. Two candidate rules:

    * **zero-fill** the missing channels. Rejected: under
      ``AudioConfig.channels="downmix"`` a mono row batched with a stereo one
      would be averaged against a channel of silence and come out at **half
      amplitude**, i.e. its score would depend on what shared its batch. That is
      the violation, not a rounding deviation.
    * **cyclic repeat** of the row's own channels. Chosen, but critically
      **only when ``channels`` is an exact multiple of ``have``**, and that
      qualifier is load-bearing. When it divides, every shipped policy
      (``models.audio.CHANNEL_POLICIES``) gives **bitwise** the un-promoted row:
      each source channel appears the same number of times, so ``downmix``'s mean
      is unchanged, ``left`` reads channel 0 either way and ``mid_side``'s
      ``0.5*(c0 + c1)`` is unchanged -- exact in float32, since the arithmetic is
      multiplication by a power of two. When it does **not** divide, the repeat
      is uneven and the identity fails outright: 2 -> 3 gives ``[L, R, L]``,
      which ``downmix`` averages to ``(2L + R)/3`` instead of ``(L + R)/2``.
      Measured max deviation **1.05** raw, and **1.12** end to end at the model
      boundary for a stereo row collated beside a 3-channel one -- the same
      half-amplitude class of violation this rejects zero-fill for.

    So an uneven promotion is refused rather than performed. Caveat: it is
    latent today only because no corpus file has more than 2 channels, which is
    a property of the corpus and not an invariant of this code: ``sf.read`` is
    called with ``always_2d=True``, ``render._to_channels`` keeps the leading
    channels of a multi-channel source rather than downmixing it, and the
    ``channels`` normalize draw accepts ``null``. A 5.1 file entering the corpus
    must fail loudly here, not silently rescale a stereo neighbour's score.

    Caveat: the invariance is not a property of *this* function alone either:
    it is a joint property of the promotion and the channel policy, and a new
    policy could break it (a hypothetical ``side`` policy would read 0 from a
    promoted mono row rather than its true absence of a second channel). It is
    therefore **measured** rather than declared --
    ``tests/test_collate.py::test_promotion_is_invisible_at_the_model_boundary``
    enumerates ``CHANNEL_POLICIES`` from the source of truth, so adding a policy
    without checking it fails the suite instead of shipping.

    Caveat: also why the collator does not simply downmix -- that would fork
    the channel policy into two places, disable the A-B3 channel augmentations,
    and make the ``mid_side`` leak test (09 B8) impossible to run.
    """
    have = int(wav.shape[0])
    if have == channels:
        return wav
    if have > channels:                                       # pragma: no cover
        raise ValueError(f"row has {have} channels, more than the batch's {channels}")
    if channels % have:
        raise ValueError(
            f"cannot promote a {have}-channel row into a {channels}-channel "
            f"batch: {channels} is not a multiple of {have}, so the repeat is "
            f"uneven and the row's own content would change -- 2 -> 3 downmixes "
            f"to (2L+R)/3 rather than (L+R)/2, a rule-2.4 violation of up to "
            f"1.12 at the model boundary. Split the batch by channel count, or "
            f"apply a channel policy before collation")
    reps = channels // have
    return wav.repeat(reps, 1)


def collate(samples: Sequence[RenderedSample], *,
            pad_value: float = 0.0) -> dict[str, Any]:
    """The batch of docs/pipelines/04 §2. Zero-padded, float32, mandatory lengths.

    ``pad_value`` exists **for the tests**, and is a contract rather than a knob:
    docs/pipelines/04 §2 says the padding value must not matter, and the way to
    know that is to collate the same samples twice with different fillings and
    check the *submitted probability* is unchanged (I16). It defaults to 0.0 and
    training never passes it.

    Critical: ``lengths`` is mandatory and comes from each row's own sample
    count. It is what makes the frame mask correct; without it every padded
    frame counts as real audio and ``frame_max`` starts reading another file's
    padding.
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

    # Critical: refused here, with the whole batch in hand, rather than one
    # row at a time inside `promote_channels`: the caller needs to know which
    # counts collided, and the answer ("split the batch by channel count") is
    # a statement about the batch. See `promote_channels` for the arithmetic.
    uneven = sorted({c for c in (int(s.wav.shape[0]) for s in samples)
                     if channels % c})
    if uneven:
        raise ValueError(
            f"batch mixes channel counts {uneven} with a {channels}-channel row; "
            f"{channels} is not a multiple of {uneven}, so promotion would change "
            f"those rows' own content (2 -> 3 downmixes to (2L+R)/3, not (L+R)/2)")

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
        # Critical: carried through untouched, in **absolute seconds**.
        # Rasterising to a frame grid here would hard-code a frame rate the
        # collator does not know -- the `align_time` defect, one layer up.
        "frame_intervals": [dict(s.frame_intervals) for s in samples],
        "specs": [s.spec for s in samples],
    }


# --------------------------------------------------------------------------- #
# Duration bucketing -- docs/pipelines/04 §3
#
# The objection that used to block this has been resolved away rather than
# solved: `pairwise_ranking_loss` formed positive/negative pairs *within* the
# batch, which would have made batch composition part of the objective. With
# `LossConfig.ranking_weight` committed at 0 and stage S4 dropped, no loss term
# is sensitive to batch composition and bucketing may be chosen purely for
# padding efficiency.
#
# **Duration and cell are independent, by construction rather than by luck.**
# `sample_spec` draws `duration_s` FIRST, from the test distribution U(4, 60),
# and only then draws the cell and fits the components into that timeline
# (`take = min(seg, row.duration_s)`). Nothing downstream can feed back into the
# length, so a duration bucket cannot be a cell filter. Measured over 2,000 drawn
# specs cut into four quantile buckets: voice presence 0.682 / 0.678 / 0.676 /
# 0.694 and music presence 0.656 / 0.678 / 0.674 / 0.688, flat across the whole
# 4-60 s span.
#
# Caveat: an earlier version of this note claimed the opposite -- "sequential
# compositions run long" -- and it is simply false against this sampler: a
# sequential draw reuses the same drawn duration rather than concatenating two,
# so mean duration is 30.85 s sequential against 30.38 s overlap, and sequential
# runs at 40-52 per 500 in every bucket. The independence is a property we impose
# (draw the length from the test distribution, then fit), which is a stronger
# guarantee than the accident that was being claimed.
#
# Critical: the **C2** check therefore stays as cheap insurance, not as a live
# hazard: `audit_specs(specs, batches=training_batches(...))` measures the
# per-head present count on the plan the optimiser will actually step on. Pass
# the plan -- a `batch_size` alone cuts the stream in draw order, which stopped
# being the training order the moment bucketing arrived.
#
# And the duration-vs-score check on the REAL class must be measured on
# **unbucketed** eval batches, which the frozen eval spec list is by construction
# (`training.dataset.eval_batches`).


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
    """Which bucket a duration falls in, given ``bucket_edges``' interior edges.

    Caveat: ``side="right"`` is a tie-break convention with **no observable
    consequence here**, and it is documented rather than tested because there is
    nothing to test. It differs from ``side="left"`` only for a duration exactly
    equal to an edge, and over 2,000 drawn specs there were 0 such rows and 0
    rows assigned differently by the two -- durations come from
    ``rng.uniform(4, 60)`` and the edges from ``np.quantile``'s interpolation, so
    exact equality has measure zero. Asserting one of them would pin an arbitrary
    choice, which is how a test starts confirming the implementation.

    Critical: the property that *is* worth holding, and is tested -- bucketing
    is a function of the **duration**, so two equal durations always land in
    the same bucket. A rank-based bucketer (``np.array_split`` over
    ``argsort``) is the plausible alternative and splits ties across a
    boundary, which makes the batch a row happens to land in depend on the
    other rows.
    """
    return int(np.searchsorted(np.asarray(edges, dtype=float), float(duration),
                               side="right"))


def bucket_batches(durations: Sequence[float], batch_size: int, *,
                   n_buckets: int = 4, seed: int = 0,
                   drop_last: bool = True) -> list[list[int]]:
    """Index lists, one per batch, grouped so a batch spans one duration bucket.

    Caveat: ``drop_last`` defaults to True and matches what training does,
    which is also what `audit_specs`' C2 check assumes: C2's floor is absolute
    (the gradient norm scales as 1/√n), so a short trailing batch can never
    satisfy it and the optimiser never sees one.

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
    # Critical: the batch *order* is shuffled too. Otherwise every epoch would
    # walk the short files first and the long ones last, so the optimiser
    # would see a duration schedule -- batch composition back in the objective
    # by the side door, which is exactly what §3 argues bucketing is free of.
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

    Caveat: it is `spec.duration_s`, the drawn timeline, not the rendered
    length -- bucketing has to happen before rendering or it buys nothing. The
    two agree to within resampler rounding, and `render` refuses a sample whose
    *rendered* length leaves the length regime (I12), so the approximation
    cannot drift.
    """
    return [float(s.duration_s) for s in specs]
