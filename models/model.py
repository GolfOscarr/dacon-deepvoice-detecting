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

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from models.audio import bandpass
from models.config import ModelConfig, dump_config, _model_from_dict
from models.frontends import build_frontend
from models.heads import SEDHead
from models.outputs import branch_logit, to_probability
from models.utils import align_time

__all__ = ["DeepVoiceNet", "load_checkpoint", "save_checkpoint", "shipped_weights"]

#: Config fields the *model* deliberately does not read, with who owns them.
#: tests/test_model.py asserts that every other field is read somewhere, so a
#: knob cannot go quietly unimplemented (see `test_no_config_field_is_silently_ignored`).
CONSUMED_ELSEWHERE = {
    "audio.channels":        "models.audio.prepare_waveform, called by the data path",
    "audio.min_seconds":     "the data loader",
    "audio.max_seconds":     "the data loader",
    "runtime.precision":     "the inference script",
    "runtime.compile":       "the inference script",
    "runtime.batch_size":    "the inference script",
    "runtime.num_workers":   "the inference script",
    "segmentation.window_seconds": "the inference script, when tiling (the model refuses tiling)",
    "segmentation.hop_seconds":    "the inference script, when tiling",
    "aggregation.kind":      "models.outputs.aggregate_windows, when tiling",
    "aggregation.k":         "models.outputs.aggregate_windows, when tiling",
    "aggregation.quantile":  "models.outputs.aggregate_windows, when tiling",
    "frontends.layers":      "real frontends; the stub refuses it rather than ignoring it",
    "frontends.adapter.kind":    "real frontends; the stub refuses it",
    "frontends.adapter.rank":    "real frontends",
    "frontends.adapter.alpha":   "real frontends",
    "frontends.adapter.dropout": "real frontends",
    "frontends.adapter.targets": "real frontends",
}


class DeepVoiceNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg

        if cfg.segmentation.mode != "whole_file":
            raise NotImplementedError(
                f"segmentation.mode={cfg.segmentation.mode!r} is not implemented. The "
                "model consumes a whole waveform; windowing and cross-window pooling "
                "belong to the inference script, which does not exist yet. Silently "
                "running whole_file here would make a tiling ablation measure nothing.")

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
        fps_tgt = self.frontends[br.align_to].fps
        feats, masks = [], []
        for src in br.sources:
            f, m = align_time(*encoded[src], target_frames,
                              self.frontends[src].fps, fps_tgt)
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
        wav = bandpass(wav, self.cfg.audio, lengths)
        encoded = {name: fe(wav, lengths) for name, fe in self.frontends.items()}

        # 🔴 The stop-gradient of the borrowed distillation recipe: the branch
        # heads get *detached* features so the distillation loss owns the
        # backbone, rather than the two losses fighting over it. ⚠️ With a frozen
        # frontend there is little left for it to own -- see 09 B9, which flags
        # exactly that -- so this is wired to be measurable, not because the
        # borrowed number transfers unchanged.
        detach = self.distill_head is not None and self.cfg.distill.stop_gradient
        branch_src = ({k: (f.detach(), m) for k, (f, m) in encoded.items()}
                      if detach else encoded)

        out = {}
        for name in self.cfg.branches:
            feats, mask = self._branch_input(name, branch_src)
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

    def submission_probs(self, out: dict,
                         frame_masks: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """The five submission columns, as float64 probabilities.

        Honours ``file_head.mode``. G3 records the FILE construction as an open
        question with no prior art, so all three must actually differ -- an
        earlier version validated the field and then always produced `learned`,
        which would have made a three-way comparison return one answer.
        """
        fm = frame_masks or {}
        probs = {}
        by_col = {}
        for name, br in self.cfg.branches.items():
            z = branch_logit(out[name], br.head, fm.get(name))
            by_col[br.column] = z
            probs[br.column] = to_probability(z, self.cfg.output)

        mode = self.cfg.file_head.mode
        if mode != "learned":
            eps = self.cfg.output.clamp_eps
            v = probs["VOICE_FAKE_PROB"] * probs["VOICE_PRESENT_PROB"]
            m = probs["MUSIC_FAKE_PROB"] * probs["MUSIC_PRESENT_PROB"]
            if mode == "noisy_or":
                probs["FILE_FAKE_PROB"] = (1.0 - (1.0 - v) * (1.0 - m)).clamp(eps, 1.0 - eps)
            elif mode == "max":
                probs["FILE_FAKE_PROB"] = torch.maximum(v, m).clamp(eps, 1.0 - eps)
            else:
                raise ValueError(f"unknown file_head.mode {mode!r}")
        return probs

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


def shipped_weights(model_dir: str | Path) -> dict[str, str] | None:
    """``model_dir/weights/<frontend>/`` -> {frontend: dir}, or None if absent.

    The layout `script.py` ships: one directory per pretrained frontend, holding
    what that frontend needs to *construct* (the BEATs ``.pt``; the XLS-R
    snapshot with ``config.json``). Keys are frontend names from the config.
    """
    root = Path(model_dir) / "weights"
    if not root.is_dir():
        return None
    return {p.name: str(p) for p in sorted(root.iterdir()) if p.is_dir()} or None


def load_checkpoint(path: str | Path, map_location="cpu",
                    weights: Mapping[str, str | Path] | None = None) -> DeepVoiceNet:
    """Rebuild from the stored config, then load the state dict strictly.

    ``weights`` maps frontend name -> pretrained checkpoint directory, replacing
    the stored ``frontends.<name>.weights``. ⚠️ A pretrained frontend (BEATs,
    XLS-R) reads its checkpoint at *construction* for the architecture, so the
    stored path -- the training machine's absolute /data/... path -- must
    resolve wherever this runs. On the offline test server it will not;
    `script.py` passes `shipped_weights(model_dir)`. Every weight is then
    overwritten by the strict load below, so which copy of the pretrained file
    is read changes no number -- a test asserts that bitwise.
    """
    import dataclasses

    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "config" not in blob or "state_dict" not in blob:
        raise ValueError(
            f"{path}: not a DeepVoiceNet checkpoint (expected 'config' and 'state_dict')")
    cfg = _model_from_dict(blob["config"])
    if weights:
        unknown = sorted(set(weights) - set(cfg.frontends))
        if unknown:
            raise ValueError(f"weights names frontends {unknown} not in the checkpoint's "
                             f"config {sorted(cfg.frontends)}")
        cfg = dataclasses.replace(cfg, frontends={
            k: (dataclasses.replace(v, weights=str(weights[k])) if k in weights else v)
            for k, v in cfg.frontends.items()})
    model = DeepVoiceNet(cfg)
    # strict=True raises on any missing or unexpected key. That is the point: the
    # config was rebuilt from the same blob, so a mismatch means the checkpoint is
    # not what it claims to be rather than that the architecture drifted.
    model.load_state_dict(blob["state_dict"], strict=True)
    return model.eval()
