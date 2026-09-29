"""Test time: a directory of files -> the five submission columns, through the
ONE shipped chain (docs/processing/06 P8, D14; docs/competition/02).

The path is ``load_audio`` (robust decode, ffmpeg fallback) -> pad to a batch
-> ``processing.ship.ship`` -> the model -> ``submission_probs``. It is the
same ``ship`` call the training loop and the validator make, on the same
``ShipConfig`` the run wrote beside its weights (``processing.json``), which is
what "train/test chain identical" means as code rather than as a sentence.

Per file: a decode error yields a fallback row of 0.5 and is counted; a
submission with more than ``max_fallback`` of them raises, because a silent
half-and-half CSV scores exactly 0.5000 and misleads (02 §guards).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from metrics.dacon import PREDICTION_COLUMNS
from processing.ship import ShipConfig, ship
from training.render import DecodeError, load_audio

__all__ = ["AUDIO_SUFFIXES", "ENSEMBLE_RULES", "ConstantModel", "InferenceReport",
           "ensemble_probs", "list_test_files", "predict_files", "write_submission"]

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus")
#: how `ensemble_probs` combines members' per-file probabilities
ENSEMBLE_RULES = ("prob_mean", "logit_mean")


def list_test_files(test_dir: Path) -> list[Path]:
    """Every audio file under ``test_dir``, sorted, so the row order is stable."""
    return sorted(p for p in Path(test_dir).rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


@dataclass
class InferenceReport:
    ids: list[str] = field(default_factory=list)
    probs: dict[str, list[float]] = field(
        default_factory=lambda: {c: [] for c in PREDICTION_COLUMNS})
    n_fallback: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.ids)

    def frame(self):
        import pandas as pd
        return pd.DataFrame({"ID": self.ids, **{c: self.probs[c] for c in PREDICTION_COLUMNS}})


class ConstantModel:
    """P0-a: every column a constant. The contract probe (docs/architecture/03):
    the packaging works iff this scores exactly 0.5000."""

    def __init__(self, value: float = 0.5, sample_rate: int = 16_000):
        self.value = float(value)
        self.sample_rate = sample_rate

    def probs(self, wav: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        n = int(wav.shape[0])
        return {c: torch.full((n,), self.value, dtype=torch.float64) for c in PREDICTION_COLUMNS}


def _model_probs(model: Any, wav: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
    if hasattr(model, "probs"):
        return model.probs(wav, lengths)
    out = model(wav, lengths)
    return model.submission_probs(out)


def predict_files(model: Any, files: Sequence[Path], ship_cfg: ShipConfig, *,
                  batch_size: int = 8, device: str | torch.device = "cpu",
                  max_fallback: float = 0.01,
                  decode: Callable[..., np.ndarray] = load_audio) -> InferenceReport:
    """Decode, ship, predict; one fallback row per undecodable file."""
    device = torch.device(device)
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    sr = int(ship_cfg.sample_rate)
    report = InferenceReport()
    with torch.no_grad():
        for start in range(0, len(files), batch_size):
            chunk = files[start:start + batch_size]
            pieces: list[np.ndarray | None] = []
            for path in chunk:
                try:
                    pieces.append(decode(path, sample_rate=sr, file_id=path.name))
                except (DecodeError, OSError, ValueError) as exc:
                    pieces.append(None)
                    report.failures.append((path.name, str(exc)[:200]))
            good = [(i, p) for i, p in enumerate(pieces) if p is not None]
            probs: dict[str, np.ndarray] = {c: np.full(len(chunk), 0.5) for c in PREDICTION_COLUMNS}
            if good:
                channels = max(p.shape[0] for _, p in good)
                longest = max(p.shape[-1] for _, p in good)
                wav = torch.zeros(len(good), channels, longest, dtype=torch.float32)
                lengths = torch.zeros(len(good), dtype=torch.int64)
                for row, (_, p) in enumerate(good):
                    x = torch.from_numpy(np.ascontiguousarray(p, dtype=np.float32))
                    if x.shape[0] == 1 and channels > 1:
                        x = x.expand(channels, -1)
                    wav[row, : x.shape[0], : x.shape[-1]] = x
                    lengths[row] = x.shape[-1]
                shipped = ship(wav.to(device), ship_cfg, lengths.to(device))
                out = _model_probs(model, shipped, lengths.to(device))
                for c in PREDICTION_COLUMNS:
                    vals = out[c].detach().double().cpu().numpy()
                    for row, (i, _) in enumerate(good):
                        probs[c][i] = float(vals[row])
            for i, path in enumerate(chunk):
                report.ids.append(path.stem)
                for c in PREDICTION_COLUMNS:
                    report.probs[c].append(float(probs[c][i]))
            report.n_fallback += len(chunk) - len(good)
    if report.n and report.n_fallback / report.n > max_fallback:
        raise RuntimeError(
            f"{report.n_fallback} of {report.n} files fell back to 0.5 "
            f"(> {max_fallback:.0%}): {report.failures[:3]}")
    return report


def ensemble_probs(members: Sequence[Mapping[str, Sequence[float]]], rule: str = "prob_mean",
                   weights: Sequence[float] | None = None,
                   valid: Sequence[Sequence[bool]] | None = None) -> dict[str, np.ndarray]:
    """Per-file, per-column weighted average of several models' probabilities.

    ``members[m][col]`` is model m's column over the SAME file order. Each file's
    output depends only on that file's member rows (rule 2.4: no cross-file
    statistics). ``valid[m][i] = False`` drops member m from file i (a decode
    fallback row); a file no member scored is 0.5. ``logit_mean`` averages
    log-odds (probabilities clipped to [1e-12, 1 - 1e-12]).
    """
    if rule not in ENSEMBLE_RULES:
        raise ValueError(f"unknown ensemble rule {rule!r}; expected one of {ENSEMBLE_RULES}")
    if not members:
        raise ValueError("no ensemble members")
    w = np.ones(len(members)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(members),) or not (w >= 0).all() or not w.sum() > 0:
        raise ValueError(f"ensemble weights {list(w)} do not fit {len(members)} members")
    n = len(members[0][PREDICTION_COLUMNS[0]])
    ok = (np.ones((len(members), n), dtype=bool) if valid is None
          else np.asarray(valid, dtype=bool).reshape(len(members), n))
    wm = w[:, None] * ok                                   # (members, files)
    total = wm.sum(0)
    out = {}
    for c in PREDICTION_COLUMNS:
        p = np.stack([np.asarray(m[c], dtype=np.float64) for m in members])
        if p.shape != (len(members), n):
            raise ValueError(f"column {c}: members disagree on the number of files")
        if rule == "logit_mean":
            q = np.clip(p, 1e-12, 1 - 1e-12)
            p = np.log(q / (1 - q))
        num = (wm * p).sum(0)
        avg = np.divide(num, total, out=np.zeros(n), where=total > 0)
        if rule == "logit_mean":
            avg = 1 / (1 + np.exp(-avg))
        out[c] = np.where(total > 0, avg, 0.5)
    return out


def write_submission(report: InferenceReport, out_path: Path, *,
                     require_variation: bool = True) -> Path:
    """``output/submission.csv``: ``ID`` + the five columns, UTF-8. With
    ``require_variation`` a constant column is refused (02 §guards, 5) -- the
    P0-a probe passes ``False``, being constant on purpose."""
    frame = report.frame()
    if require_variation:
        spread = frame[list(PREDICTION_COLUMNS)].std(axis=0).min()
        if not spread > 1e-6:
            raise RuntimeError("a prediction column is constant: the model did not run")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def iter_ids(files: Iterable[Path]) -> list[str]:
    return [Path(f).stem for f in files]
