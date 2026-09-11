"""The EMA, the checkpoint format, the resume state and the weight soup.

Split out of `training.loop`: this is the on-disk contract of a run, and it is
readable without the trainer that writes it. Nothing here imports
`training.loop` or `training.validate`.

Critical: **a resumed run must be the run it claims to be.** Restoring weights
and the optimizer is not enough: the corpus is drawn, so a resume that forgets
*where in the draw it was* silently trains on a different corpus and reports it
under the same exp_id. `Sampler.epoch_specs` is keyed on epoch-local `i`,
`epoch` and `seed` for exactly this reason -- there is no hidden generator to
serialise, so the sampler's state is the tuple `(pass_index, epoch_seed,
n_specs, batch_seed, batch_index)` and it lives in the checkpoint. `tests/test_loop.py`
proves a resumed run reproduces an uninterrupted one **bitwise**, and proves the
check goes red when that tuple is dropped.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from models.config import TrainConfig, dump_config
from models.model import DeepVoiceNet

__all__ = [
    "EMA", "SamplerState", "TrainCheckpoint",
    "checkpoint_soup", "load_train_checkpoint", "save_train_checkpoint",
]


# --------------------------------------------------------------------------- #
# 1. EMA -- docs/architecture/08 §4
# --------------------------------------------------------------------------- #

class EMA:
    """Bias-corrected exponential moving average of the float parameters.

    Critical: bias-corrected, like Adam's moments, and that is not decoration. A
    raw EMA is initialised at the *starting* weights, so at decay 0.999 it is
    still 63% initialisation after 1,000 steps -- on a short schedule the "EMA
    weights" would mostly be the random init, and the run would report a number
    for a model it never trained. With the correction the EMA after one update
    is exactly the current weights, which is a testable statement and is tested.

    Caveat: integer buffers (`num_batches_tracked` and friends) are carried, not
    averaged: an averaged counter is meaningless and `load_state_dict(strict=True)`
    would reject a float one.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.steps = 0
        self._shadow = {k: torch.zeros_like(v, dtype=torch.float32)
                        for k, v in model.state_dict().items()
                        if v.is_floating_point()}
        self._frozen = {k: v.clone() for k, v in model.state_dict().items()
                        if not v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for k, shadow in self._shadow.items():
            shadow.mul_(self.decay).add_(state[k].detach().to(torch.float32),
                                         alpha=1.0 - self.decay)
        for k in self._frozen:
            self._frozen[k] = state[k].clone()
        self.steps += 1

    def state_dict_for(self, model: torch.nn.Module) -> dict[str, Tensor]:
        """The bias-corrected weights, shaped to load into `model` strictly."""
        if self.steps == 0:
            raise RuntimeError("EMA has taken no update; there is nothing to materialise")
        correction = 1.0 - self.decay ** self.steps
        reference = model.state_dict()
        out = {k: (v / correction).to(reference[k].dtype)
               for k, v in self._shadow.items()}
        out.update({k: v.clone() for k, v in self._frozen.items()})
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "steps": self.steps,
                "shadow": {k: v.clone() for k, v in self._shadow.items()},
                "frozen": {k: v.clone() for k, v in self._frozen.items()}}

    def load_state_dict(self, blob: Mapping[str, Any]) -> None:
        self.decay = float(blob["decay"])
        self.steps = int(blob["steps"])
        self._shadow = {k: v.clone() for k, v in blob["shadow"].items()}
        self._frozen = {k: v.clone() for k, v in blob["frozen"].items()}


# --------------------------------------------------------------------------- #
# 2. Checkpoints, resume, and the soup
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SamplerState:
    """Where in the draw a run is. Critical: the part a resume silently gets wrong.

    There is no hidden generator to serialise -- `Sampler.epoch_specs` is a pure
    function of `(i, epoch, seed)` and `bucket_batches` a pure function of
    `(durations, seed)` -- so the whole sampler state is these five numbers. That
    is the *reason* the sampler was built stateless (docs/pipelines/02 §6), and
    it is why forgetting them is so easy: nothing crashes, the loss curve looks
    fine, and the run trains on a corpus it never reports.

    ``pass_index`` indexes the flat ``(branch group, epoch)`` schedule, so it
    doubles as the epoch key for the draw: every S1 branch pass sees its own
    epoch of specs rather than four replays of epoch 0.
    """

    pass_index: int
    epoch_seed: int
    n_specs: int
    batch_seed: int
    batch_index: int

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class TrainCheckpoint:
    """A resumable run, on disk. Rebuilt through `models.model`'s own contract.

    Caveat: carries `config` for the same reason `models.model.save_checkpoint`
    does: the model is rebuilt from the stored config before a **strict** load,
    so a mismatch raises instead of producing a well-shaped model full of noise.
    """

    path: Path
    stage: str
    global_step: int
    sampler: SamplerState


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Mapping[str, Any]) -> None:
    """Restore the generators.

    🔴 `.cpu()` on both, and it is load-bearing. `train_stage` loads the
    checkpoint with `map_location=device`, so on a GPU run `torch.load` moves
    *every* tensor in the blob to CUDA -- including these RNG ByteTensors, which
    `torch.set_rng_state` and `torch.cuda.set_rng_state_all` both require on the
    CPU. Without the cast, resuming raises `TypeError: RNG state must be a
    torch.ByteTensor` and the bitwise-resume guarantee is unreachable on the only
    device a real run uses.

    It went unseen because `LoopConfig.device` defaults to "cpu" and the suite
    never probes cuda, so `map_location` was always "cpu" and the tensors never
    moved. Found by running the resume test on an allocated H200 rather than on
    the test hardware.
    """
    torch.set_rng_state(state["cpu"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_train_checkpoint(path: Path | str, *, model: DeepVoiceNet,
                          optimizer: torch.optim.Optimizer | None,
                          ema: EMA | None, stage: str, global_step: int,
                          sampler: SamplerState, train_cfg: TrainConfig,
                          scaler: "torch.amp.GradScaler | None" = None,
                          extra: Mapping[str, Any] | None = None) -> TrainCheckpoint:
    """Weights, optimizer, EMA, **sampler state** and the torch RNG state.

    Critical: the last two are the ones that get dropped. Dropout draws from the
    global torch generator, so without its state a resumed run diverges from an
    uninterrupted one on the very first step -- and without `sampler` it diverges
    on the very first *batch*, which is worse because the loss curve does not
    show it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": dump_config(model.cfg),
        "train_config": dump_config(train_cfg),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        # The fp16 loss scale is state too: it adapts over the run, so a resume
        # that rebuilds it at the default replays the scaler's warm-up and takes
        # different steps for the first few hundred batches.
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "stage": stage,
        "global_step": int(global_step),
        "sampler": sampler.as_dict(),
        "rng": _rng_state(),
        "extra": dict(extra or {}),
    }, path)
    return TrainCheckpoint(path, stage, int(global_step), sampler)


def load_train_checkpoint(path: Path | str, map_location="cpu") -> dict[str, Any]:
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    missing = [k for k in ("state_dict", "sampler", "rng", "stage") if k not in blob]
    if missing:
        raise ValueError(
            f"{path}: not a training checkpoint (missing {missing}). A checkpoint "
            "without `sampler` cannot resume the draw and a resumed run would "
            "quietly train on a different corpus")
    blob["sampler"] = SamplerState(**blob["sampler"])
    return blob


def checkpoint_soup(paths: Sequence[Path | str]) -> dict[str, Tensor]:
    """Uniform average of several checkpoints' weights.

    Free ensembling at zero inference cost (primary source,
    docs/architecture/05 §3), across **epochs and seeds** -- which is why this
    takes a list of files rather than an in-run buffer: the across-seed soup is
    assembled from separate runs.

    Critical: refuses a mismatched set instead of averaging what it can. A soup
    of two architectures, or of one model with a differently-shaped head, is not
    a worse model -- it is a `load_state_dict` failure deferred to whoever ships
    it, or worse, a silent partial average. There are four refusals -- fewer than
    two files, a missing `config`, a differing key set, a differing shape -- and
    `tests/test_checkpoint.py` gives each one a case that **isolates** it,
    because a single mismatched pair trips whichever check is left standing and
    would have hidden the deletion of any of the others. Deleting any of the
    four is **silent** -- nothing raises, a soup comes out -- and the two worth
    naming are that an extra key is dropped from the average without comment,
    and that a `(1, n)` tensor against an `(n, n)` one broadcasts.

    Caveat: non-float entries are taken from the first checkpoint rather than
    averaged, for the reason `EMA` gives.
    """
    if len(paths) < 2:
        raise ValueError(
            f"a soup of {len(paths)} checkpoint(s) is that checkpoint; pass two or more")
    states, configs = [], []
    for p in paths:
        blob = torch.load(Path(p), map_location="cpu", weights_only=False)
        if "state_dict" not in blob:
            raise ValueError(f"{p}: not a checkpoint (no 'state_dict')")
        # Critical: `blob.get("config")` made the check below vacuous for any
        # checkpoint without one -- two such files both read as `None`, compare
        # equal, and get souped with nothing having compared their
        # architectures. A missing config is "cannot tell", not "the same".
        if "config" not in blob:
            raise ValueError(
                f"{p}: has no 'config', so it cannot be checked against the "
                "others. Souping it would be averaging architectures that were "
                "never compared")
        states.append(blob["state_dict"])
        configs.append(blob["config"])

    reference = states[0]
    for p, state in zip(paths[1:], states[1:]):
        if set(state) != set(reference):
            diff = sorted(set(state) ^ set(reference))[:5]
            raise ValueError(f"{p}: key set differs from {paths[0]} (e.g. {diff})")
        for k in reference:
            if state[k].shape != reference[k].shape:
                raise ValueError(
                    f"{p}: {k} is {tuple(state[k].shape)}, "
                    f"{paths[0]} has {tuple(reference[k].shape)}")
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError(
            "refusing to soup checkpoints built from different configs: the "
            "average of two architectures is not a model")

    out: dict[str, Tensor] = {}
    for k, v in reference.items():
        if not v.is_floating_point():
            out[k] = v.clone()
            continue
        acc = torch.zeros_like(v, dtype=torch.float64)
        for state in states:
            acc += state[k].to(torch.float64)
        out[k] = (acc / len(states)).to(v.dtype)
    return out
