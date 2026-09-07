"""The model.

Candidates A and B are the same class with different configs, which is what
"B strictly contains A" means in code (docs/architecture/03). A names one
frontend and points all five branches at it; B names two and gives the file
branch both.

🔴 The branches are **outputs, not input types**. Every file passes through
every branch; nothing is routed anywhere. `VOICE_FAKE_PROB` on a music-only
file is simply never scored -- the organizers mask it using their own presence
labels -- so it costs nothing and buys nothing. Routing by input type is
candidate D, which is rejected because a 혼합 file has both components present
and independently fake, leaving a hard switch undefined.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from models.config import ModelConfig, dump_config, _model_from_dict
from models.frontends import build_frontend
from models.heads import SEDHead
from models.utils import align_time

__all__ = ["DeepVoiceNet", "load_checkpoint", "save_checkpoint"]


class DeepVoiceNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg

        self.frontends = nn.ModuleDict(
            {name: build_frontend(fe, cfg.audio) for name, fe in cfg.frontends.items()})

        self.heads = nn.ModuleDict()
        for name, br in cfg.branches.items():
            in_dim = sum(cfg.frontends[s].output_dim for s in br.sources)
            self.heads[name] = SEDHead(in_dim, br.head)

        # Training-only appendages. Both are deleted before packaging, so neither
        # may contribute to a submission column.
        self.distill_head = None
        if cfg.distill.enabled:
            widest = max(fe.output_dim for fe in cfg.frontends.values())
            self.distill_head = nn.Linear(widest, cfg.distill.embed_dim)
        self.separation_head = None
        if cfg.aux.separation_head:                     # candidate C-lite
            widest = max(fe.output_dim for fe in cfg.frontends.values())
            self.separation_head = nn.Linear(widest, 2 * cfg.aux.stft_bins)

    # ---------------------------------------------------------------- helpers

    def _branch_input(self, name: str, encoded: dict[str, tuple[Tensor, Tensor]]):
        """Gather a branch's sources onto one time base and concatenate."""
        br = self.cfg.branches[name]
        if len(br.sources) == 1:
            return encoded[br.sources[0]]

        target_frames = encoded[br.align_to][0].shape[1]
        feats, masks = [], []
        for src in br.sources:
            f, m = align_time(*encoded[src], target_frames)
            feats.append(f)
            masks.append(m)
        # A frame is valid only where every source has real content.
        mask = masks[0]
        for m in masks[1:]:
            mask = mask & m
        return torch.cat(feats, dim=-1), mask

    # ---------------------------------------------------------------- forward

    def forward(self, wav: Tensor, lengths: Tensor | None = None) -> dict:
        """``wav`` is (B, S) mono at 16 kHz. Returns one SEDOutput per branch,
        plus any training-only auxiliary outputs under ``_aux``."""
        encoded = {name: fe(wav, lengths) for name, fe in self.frontends.items()}

        out = {}
        for name in self.cfg.branches:
            feats, mask = self._branch_input(name, encoded)
            out[name] = self.heads[name](feats, mask)

        aux = {}
        if self.distill_head is not None or self.separation_head is not None:
            widest = max(self.cfg.frontends, key=lambda n: self.cfg.frontends[n].output_dim)
            feats, mask = encoded[widest]
            if self.distill_head is not None:
                # Mean over *valid* frames only, then project to the teacher's dim.
                denom = mask.sum(1, keepdim=True).clamp(min=1)
                pooled = (feats * mask.unsqueeze(-1)).sum(1) / denom
                aux["distill_emb"] = self.distill_head(pooled)
            if self.separation_head is not None:
                aux["separation"] = self.separation_head(feats)
        if aux:
            out["_aux"] = aux
        return out

    # ---------------------------------------------------------------- misc

    @property
    def columns(self) -> dict[str, str]:
        """branch name -> submission column."""
        return {name: br.column for name, br in self.cfg.branches.items()}

    def n_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "DeepVoiceNet":
        return cls(cfg)


def save_checkpoint(model: DeepVoiceNet, path: str | Path) -> None:
    """Store weights *and* the config that produced them.

    `script.py` rebuilds the model from the stored config before loading the
    state dict, which is what makes a strict load a real assertion rather than a
    coincidence (docs/architecture/07 §6).
    """
    torch.save({"config": dump_config(model.cfg),
                "state_dict": model.state_dict()}, Path(path))


def load_checkpoint(path: str | Path, map_location="cpu") -> DeepVoiceNet:
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "config" not in blob or "state_dict" not in blob:
        raise ValueError(
            f"{path}: not a DeepVoiceNet checkpoint (expected 'config' and 'state_dict')")
    model = DeepVoiceNet(_model_from_dict(blob["config"]))
    # strict=True raises on any missing or unexpected key. That is the point: the
    # config was rebuilt from the same blob, so a mismatch means the checkpoint is
    # not what it claims to be rather than that the architecture drifted.
    model.load_state_dict(blob["state_dict"], strict=True)
    return model.eval()
