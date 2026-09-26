"""Data-parallel plumbing: the process group, who this process is, and the few
collectives the loop needs (docs/training/14 D0).

Critical: **one process (no `torchrun`) takes exactly the single-GPU path.**
`current()` reports world 1 whenever no process group is up, and every helper
here is then a no-op returning its input, so `train_stage` and `multitask_loss`
run bitwise what they ran before DDP existed. The 8-task job arrays depend on
it, and `tests/test_ddp.py` pins it.

A process group of size 1 is also world 1: there is nothing to average, and
wrapping the model in DDP there would only add a second code path to test.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

__all__ = ["Dist", "SINGLE", "all_gather_object", "all_reduce_max", "all_reduce_sum",
           "barrier", "current", "init_from_env", "teardown"]


@dataclass(frozen=True)
class Dist:
    rank: int = 0
    world: int = 1
    local_rank: int = 0

    @property
    def enabled(self) -> bool:
        return self.world > 1

    @property
    def main(self) -> bool:
        return self.rank == 0


SINGLE = Dist()


def init_from_env(*, device: str = "cuda", timeout_min: int = 30) -> Dist:
    """Join the process group `torchrun` describes in the environment.

    No `WORLD_SIZE` (a plain `python scripts/train.py`) or `WORLD_SIZE=1`
    returns `SINGLE` without creating a group. NCCL on CUDA, gloo on CPU; on
    CUDA the process is pinned to `cuda:LOCAL_RANK`.

    The timeout is generous on purpose: at a pass boundary rank 0 writes a
    multi-GB checkpoint while the other ranks wait at a barrier.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return SINGLE
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if cuda else "gloo",
                                timeout=datetime.timedelta(minutes=timeout_min))
    return Dist(rank=rank, world=world, local_rank=local_rank)


def current() -> Dist:
    """Who this process is, read from the live process group (world 1 if none)."""
    if not (dist.is_available() and dist.is_initialized()):
        return SINGLE
    world = dist.get_world_size()
    if world <= 1:
        return SINGLE
    rank = dist.get_rank()
    return Dist(rank=rank, world=world,
                local_rank=int(os.environ.get("LOCAL_RANK", rank)))


def all_reduce_sum(t: Tensor, d: Dist) -> Tensor:
    """The sum over ranks, as a new tensor; the input itself when world is 1."""
    if not d.enabled:
        return t
    out = t.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def all_reduce_max(t: Tensor, d: Dist) -> Tensor:
    if not d.enabled:
        return t
    out = t.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.MAX)
    return out


def all_gather_object(obj: Any, d: Dist) -> list[Any]:
    """Every rank's `obj`, in rank order; `[obj]` when world is 1."""
    if not d.enabled:
        return [obj]
    out: list[Any] = [None] * d.world
    dist.all_gather_object(out, obj)
    return out


def barrier(d: Dist) -> None:
    if d.enabled:
        dist.barrier()


def teardown(d: Dist) -> None:
    """Leave the process group. Safe to call when there is none."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
