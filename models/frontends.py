"""Frontend wrappers.

Two families of pretrained encoder are in play and they do **not** share an
output shape (docs/architecture/02, 04 §2):

* wav2vec2 / XLS-R / WavLM emit ``(B, T, D)`` -- frequency already collapsed.
* BEATs / EAT / SSLAM are ViT-style over mel patches, so their tokens form an
  ``(F', T')`` grid that must be pooled over ``F'`` before the head sees it.

Every wrapper here normalises to one contract, so `models.heads` never needs to
know which family fed it:

    forward(wav, lengths) -> (features: (B, T, D), frame_mask: (B, T))

plus ``output_dim`` and ``fps``. ``fps`` is what lets two frontends with
different frame rates be aligned onto a common time base for the file branch.

`StubFrontend` is a randomly-initialised encoder with no pretrained weights. It
exists so that every shape, mask and invariance test runs *now* -- before any
checkpoint is downloaded, and before the licence questions in
docs/architecture/09 C1-C3 are answered.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.config import AudioConfig, FrontendConfig
from models.heads import FreqPool

__all__ = ["Frontend", "StubFrontend", "BEATsFrontend", "XLSRFrontend", "LoRALinear",
           "build_frontend", "frames_for", "hop_length"]


def _is_adapter_param(name: str) -> bool:
    """True for parameters this repo added on top of a pretrained checkpoint.

    Name-based on purpose: `_apply_freeze` runs over `named_parameters()`, and a
    name test is the one form of this check that a reader can verify against a
    `state_dict` listing without running anything.
    """
    return name.endswith("lora_a") or name.endswith("lora_b") or ".lora_" in name


def hop_length(sample_rate: int, fps: float) -> int:
    """Samples per frame."""
    return max(1, round(sample_rate / fps))


def frames_for(n_samples: Tensor | int, sample_rate: int, fps: float) -> Tensor | int:
    """How many frames a frontend emits for a given number of samples.

    Defined once, because the frame mask, the time alignment between two
    frontends, and the stub's own framing must all agree on it. 🔴 Frame *i*
    must correspond to a fixed sample range regardless of how long the rest of
    the batch is -- otherwise padding shifts a file's content and both the mask
    and rule-2.4 batch invariance become meaningless.
    """
    hop = hop_length(sample_rate, fps)
    if isinstance(n_samples, Tensor):
        return torch.clamp(-(-n_samples // hop), min=1)          # ceil division
    return max(1, -(-int(n_samples) // hop))


class Frontend(nn.Module):
    """Common contract. Subclasses implement `_encode`."""

    def __init__(self, cfg: FrontendConfig, audio: AudioConfig):
        super().__init__()
        self.cfg = cfg
        self.audio = audio
        self.fps = cfg.fps
        self.output_dim = cfg.output_dim
        self.freq_pool = FreqPool(cfg.freq_pool) if cfg.freq_pool.kind != "none" else None

    def _apply_freeze(self) -> None:
        """Freeze the pretrained encoder, but not our own pooling.

        Called at the end of a subclass's __init__, once its modules exist.
        `freq_pool`'s learnable GeM exponent is ours, not the checkpoint's, so
        freezing the frontend must not freeze it too.
        """
        if not self.cfg.freeze:
            return
        pool_params = set(id(p) for p in (self.freq_pool.parameters() if self.freq_pool else []))
        for name, p in self.named_parameters():
            # 🔴 `lora_` params are OURS, not the checkpoint's, exactly as
            # freq_pool's GeM exponent is. Freezing them would leave a frozen
            # frontend with an adapter that can never move -- which loads,
            # trains and reports a loss, while learning nothing in the frontend.
            if id(p) in pool_params or _is_adapter_param(name):
                continue
            p.requires_grad_(False)

    def _encode(self, wav: Tensor, lengths: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def _frames_for(self, n_samples: Tensor | int) -> Tensor | int:
        """Frames this frontend emits for `n_samples`.

        🔴 Overridable because the module-level `frames_for` is a *nominal*
        ceil(samples/hop), and a real encoder's framing is its own arithmetic.
        BEATs is off by one frame at almost every length (a 4 s file: nominal
        25, actual 24), and the base mask would then mark a frame valid that
        the encoder produced from the padding. That is a rule-2.4 leak of the
        exact shape docs/pipelines/05 records twice already.
        """
        return frames_for(n_samples, self.audio.sample_rate, self.fps)

    def enable_grad_checkpointing(self) -> int:
        """Checkpoint the transformer layers; returns how many. A frontend with
        none (the stub) returns 0."""
        layers = self._transformer_layers()
        return enable_layer_checkpointing(layers) if layers is not None else 0

    def _transformer_layers(self) -> nn.ModuleList | None:
        return None

    def forward(self, wav: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``wav`` is (B, S) at ``audio.sample_rate``; ``lengths`` is (B,) in samples."""
        if wav.dim() != 2:
            raise ValueError(f"frontend expects (B, S) mono audio, got {tuple(wav.shape)}")

        # 🔴 Zero everything past each row's own `lengths` before encoding.
        # `frames_for` rounds UP, so a row whose length is not a multiple of
        # `hop` has a last valid frame that is *part padding* -- and that frame
        # is masked IN, so whatever fills the pad reaches the score through it.
        # Measured on the stub: two pad fillings moved a submitted probability
        # by 1.35e-3 on rendered audio, and the same rows trimmed to whole
        # frames moved by exactly 0.
        #
        # ⚠️ A no-op on the shipped path: training and inference both pad with
        # zeros, so `wav * keep` changes no number we produce today. What it
        # changes is that "a row's features are a function of its own samples"
        # stops being incidental -- true only because everyone happens to pad
        # with zeros -- and becomes structural. Do not go looking for a metric
        # shift; there isn't one.
        #
        # It could not be seen before because every padding test in the suite
        # used `lengths = SR * 4`, an exact multiple of the 320-sample hop, so
        # no test had ever had a partial boundary frame. The pipeline draws
        # U(4, 60)s and renders arbitrary sample counts.
        if lengths is not None:
            keep = torch.arange(wav.shape[-1], device=wav.device)[None, :] < \
                lengths.to(wav.device)[:, None]
            wav = wav * keep.to(wav.dtype)
        feats = self._encode(wav, lengths)

        if self.freq_pool is not None:
            if feats.dim() != 4:
                raise ValueError(
                    f"{type(self).__name__}: freq_pool is configured but the encoder "
                    f"emitted {feats.dim()}-D output; expected (B, F, T, D)")
            feats = self.freq_pool(feats)
        elif feats.dim() != 3:
            raise ValueError(
                f"{type(self).__name__}: expected (B, T, D) with freq_pool disabled, "
                f"got {tuple(feats.shape)}")

        b, t, _ = feats.shape
        if lengths is None:
            mask = torch.ones(b, t, dtype=torch.bool, device=feats.device)
        else:
            valid = self._frames_for(lengths).to(feats.device)
            mask = torch.arange(t, device=feats.device)[None, :] < valid[:, None].clamp(max=t)
        return feats, mask


class StubFrontend(Frontend):
    """A small deterministic encoder with no pretrained weights.

    🔴 Frames are **absolutely positioned**: the waveform is cut into fixed
    `hop`-sample frames, so frame *i* always covers samples [i*hop, (i+1)*hop)
    no matter how long the rest of the batch is. An earlier version pooled
    adaptively to the frame count, which *stretched* a short file's content
    across the whole padded width -- that makes the frame mask describe the
    wrong frames and quietly breaks rule-2.4 batch invariance. Real SSL
    frontends are strided convolutions and behave the absolute way; the stub
    must too, or it tests a property the real model will not have.
    """

    def __init__(self, cfg: FrontendConfig, audio: AudioConfig):
        super().__init__(cfg, audio)
        self.n_freq = cfg.n_freq or 1
        self.hop = hop_length(audio.sample_rate, cfg.fps)
        width = 64
        self.frame = nn.Linear(self.hop, width)
        self.proj = nn.Sequential(nn.GELU(), nn.Linear(width, self.n_freq * cfg.output_dim))

        # ⚠️ Refuse to look like we honoured a knob we did not. The stub has no
        # transformer layers to truncate and no attention projections to adapt,
        # so silently accepting these would let an ablation "measure" a setting
        # that never took effect.
        if cfg.layers is not None:
            raise NotImplementedError(
                "frontends: `layers` (truncation depth) has no meaning for the stub "
                "encoder and is not applied. It becomes real when a checkpoint is "
                "wired; until then set `layers: null`.")
        if cfg.adapter.kind != "none":
            raise NotImplementedError(
                f"frontends: adapter.kind={cfg.adapter.kind!r} is not implemented for the "
                "stub encoder -- there are no attention projections to adapt. Set "
                "`adapter: {kind: none}` for stub configs.")
        self._apply_freeze()

    def _encode(self, wav: Tensor, lengths: Tensor | None = None) -> Tensor:
        b, n = wav.shape
        t = frames_for(n, self.audio.sample_rate, self.fps)
        pad = t * self.hop - n
        if pad:
            wav = F.pad(wav, (0, pad))
        h = self.frame(wav.view(b, t, self.hop))              # (B, T, width)
        h = self.proj(h)                                      # (B, T, F*D)
        if self.cfg.n_freq is None:
            return h
        b, t_, _ = h.shape
        # (B, T, F, D) -> (B, F, T, D), matching what a patch-grid frontend gives us
        return h.view(b, t_, self.n_freq, self.output_dim).permute(0, 2, 1, 3).contiguous()


class LoRALinear(nn.Module):
    """LoRA on one frozen `nn.Linear` (docs/architecture/06 §2).

    🔴 `lora_b` is zero-initialised, so at construction the wrapped layer is
    *bitwise* the original. That is what makes "load BEATs, adapt it" honest:
    an untrained adapter cannot silently perturb a pretrained encoder, and a
    test asserts the equality rather than trusting the initialiser.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"lora rank must be positive, got {rank}")
        self.base = base
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_a), self.lora_b)
        return self.base(x) + delta * self.scaling


def _apply_lora(module: nn.Module, targets: tuple[str, ...], rank: int,
                alpha: float, dropout: float, *, rename: dict[str, str] | None = None,
                where: str = "the encoder") -> int:
    """Wrap every `nn.Linear` in `module` whose attribute name is in `targets`.

    `rename` maps a config target onto the backbone's own attribute name (XLS-R
    calls BEATs' `fc1`/`fc2` `intermediate_dense`/`output_dense`), so one target
    list names the same projections in both trunks.

    🔴 **Every** target must match at least once. An earlier version raised only
    when the *total* was zero, so `[q_proj, fc1]` on a backbone without `fc1`
    adapted half of what the config said and trained without complaint.

    ⚠️ A target that is a module but not an `nn.Linear` is resolved explicitly:
    a GLU projection (`models.vendor.beats.modules.GLU_Linear`, used by BEATs
    checkpoints with `activation_fn: glu`) keeps its weight in an inner
    `.linear`, which is wrapped instead. Anything else raises rather than being
    skipped. The shipped BEATs checkpoint is `gelu`, so its `fc1` is a plain
    `nn.Linear` and this branch is not taken there.
    """
    rename = rename or {}
    counts = dict.fromkeys(targets, 0)
    for child in list(module.modules()):             # snapshot: we mutate below
        for t in targets:
            attr = rename.get(t, t)
            layer = getattr(child, attr, None)
            if not isinstance(layer, nn.Module) or isinstance(layer, LoRALinear):
                continue
            owner = child
            if not isinstance(layer, nn.Linear):
                inner = getattr(layer, "linear", None)
                if not isinstance(inner, nn.Linear):
                    raise ValueError(
                        f"adapter target {t!r} is a {type(layer).__name__} in {where}: "
                        f"not an nn.Linear and no inner .linear to adapt")
                owner, attr, layer = layer, "linear", inner
            setattr(owner, attr, LoRALinear(layer, rank, alpha, dropout))
            counts[t] += 1
    unmatched = [t for t, n in counts.items() if n == 0]
    if unmatched:
        raise ValueError(
            f"adapter.targets {unmatched} matched no nn.Linear in {where} -- a "
            f"silently-inert adapter is the failure "
            f"`test_no_config_field_is_silently_ignored` exists to prevent")
    return sum(counts.values())


def enable_layer_checkpointing(layers: nn.ModuleList) -> int:
    """Recompute each transformer layer in the backward pass instead of storing
    its activations (docs/training/07 §3: a 60 s clip held ~17 GiB of attention
    maps across both trunks, and batch 8 ran the H200 out of memory).

    The layer's ``forward`` is replaced on the INSTANCE, not wrapped in a new
    module, so parameter names -- and every checkpoint and state_dict -- are
    unchanged. Active only in training mode with grad enabled; evaluation and
    inference run the original forward. ``torch.utils.checkpoint`` restores
    the RNG state for the recompute, so dropout draws are the same.
    """
    from torch.utils.checkpoint import checkpoint

    n = 0
    for layer in layers:
        if "_uncheckpointed_forward" in layer.__dict__:
            continue
        orig = layer.forward

        def forward(*args, _orig=orig, _layer=layer, **kwargs):
            if _layer.training and torch.is_grad_enabled():
                return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
            return _orig(*args, **kwargs)

        layer.__dict__["_uncheckpointed_forward"] = orig
        layer.forward = forward
        n += 1
    return n


class BEATsFrontend(Frontend):
    """BEATs (microsoft/unilm, MIT), truncated and LoRA-adapted.

    The encoder itself is vendored verbatim under `models/vendor/beats/`; this
    class is the whole of our adaptation, so re-vendoring upstream stays a copy
    rather than a merge.

    Three things here are not the upstream defaults, and each is a decision:

    🔴 **`encoder_layerdrop` is forced to 0.** Upstream draws
    `np.random.random()` per layer per forward to decide whether to skip it.
    That is a generator outside our seeding, and `training/AGENTS.md` records
    the resume guarantee as *bitwise* with a complete state list -- a numpy
    global-RNG draw in the forward pass would silently make that false. The
    checkpoint ships `encoder_layerdrop: 0.05`; we do not use it.

    🔴 **Dropout is zeroed while the encoder is frozen.** `dropout`,
    `attention_dropout` and `dropout_input` are pretraining regularisers. On a
    frozen encoder they only inject torch RNG draws into a module we are not
    training, which costs reproducibility and buys nothing.

    🔴 **The patch grid is time-major, and which one it is was measured, not
    read.** Upstream feeds fbank as `(B, 1, T_mel, 128)`, so the conv2d output
    is `(B, C, T', F')` and `reshape(B, C, -1)` orders tokens `t * F' + f`.
    Recovering the grid therefore takes `view(B, T', F', D)` and a permute, not
    `view(B, F', T', D)`. Getting this backwards transposes frequency against
    time with no shape error at all -- `F'` is 8 and `T'` is ~24 at 4 s, so the
    view simply fails, but at any length where they coincide it would train and
    be silently wrong. `StubFrontend` already lays its grid out time-major, so
    the two agree; a test pins the ordering against a synthetic grid.
    """

    N_MELS = 128
    PATCH = 16
    MEL_FPS = 100.0                 # kaldi fbank at frame_shift=10 ms
    _FRAME_LENGTH = 400             # 25 ms at 16 kHz
    _FRAME_SHIFT = 160              # 10 ms at 16 kHz
    _FFT = 512                      # kaldi rounds the 400-sample window up to 2^9
    FBANK_MEAN = 15.41663           # upstream BEATs.preprocess defaults
    FBANK_STD = 6.55582

    def __init__(self, cfg: FrontendConfig, audio: AudioConfig):
        super().__init__(cfg, audio)
        from models.vendor.beats import TransformerEncoder

        if audio.sample_rate != 16000:
            raise ValueError(
                f"BEATs is a 16 kHz model; audio.sample_rate is {audio.sample_rate}")
        expected_fps = self.MEL_FPS / self.PATCH
        if abs(cfg.fps - expected_fps) > 1e-9:
            raise ValueError(
                f"BEATs emits {expected_fps} fps -- 128 mel bins at "
                f"{self.MEL_FPS:g} fps, patched {self.PATCH}x{self.PATCH} with "
                f"stride {self.PATCH} -- but the config says fps={cfg.fps}. "
                f"fps is what aligns two frontends onto a common time base, so a "
                f"wrong value misaligns evidence silently rather than failing.")
        expected_n_freq = self.N_MELS // self.PATCH
        if cfg.n_freq != expected_n_freq:
            raise ValueError(
                f"BEATs' patch grid is {expected_n_freq} bands "
                f"({self.N_MELS} mels / {self.PATCH}); config says n_freq={cfg.n_freq}")
        if cfg.freq_pool.kind == "none":
            raise ValueError(
                "BEATs emits a (F', T') patch grid, so freq_pool must not be 'none'; "
                "docs/architecture/03 records this as easy to get wrong when implementing")

        ckpt_cfg, state = self._load_checkpoint(cfg.weights)
        if ckpt_cfg["encoder_embed_dim"] != cfg.output_dim:
            raise ValueError(
                f"checkpoint encoder_embed_dim={ckpt_cfg['encoder_embed_dim']} but "
                f"config output_dim={cfg.output_dim}")
        if ckpt_cfg["input_patch_size"] != self.PATCH:
            raise ValueError(
                f"checkpoint input_patch_size={ckpt_cfg['input_patch_size']}, "
                f"expected {self.PATCH}")

        embed = ckpt_cfg["embed_dim"]
        self.patch_embedding = nn.Conv2d(1, embed, kernel_size=self.PATCH,
                                         stride=self.PATCH, bias=ckpt_cfg["conv_bias"])
        self.layer_norm = nn.LayerNorm(embed)
        self.post_extract_proj = (
            nn.Linear(embed, cfg.output_dim) if embed != cfg.output_dim else None)

        enc_cfg = _BEATsEncoderConfig(ckpt_cfg, freeze=cfg.freeze)
        self.encoder = TransformerEncoder(enc_cfg)

        self.n_layers = self._truncate(cfg.layers, ckpt_cfg["encoder_layers"])
        self._load_state(state)

        self.n_lora = 0
        if cfg.adapter.kind == "lora":
            self.n_lora = _apply_lora(self.encoder.layers, cfg.adapter.targets,
                                      cfg.adapter.rank, cfg.adapter.alpha,
                                      cfg.adapter.dropout, where="the BEATs encoder")
        elif cfg.adapter.kind != "none":
            raise ValueError(f"unsupported adapter kind {cfg.adapter.kind!r} for BEATs")

        self._apply_freeze()

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _load_checkpoint(weights: str | None):
        if weights is None:
            raise ValueError(
                "BEATsFrontend needs `weights`: a path to BEATs_iter3_plus_AS2M.pt. "
                "A randomly-initialised BEATs is a StubFrontend with a longer runtime "
                "-- use `name: stub` for shape work instead.")
        import os
        path = weights
        if os.path.isdir(path):
            path = os.path.join(path, "BEATs_iter3_plus_AS2M.pt")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if "cfg" not in blob or "model" not in blob:
            raise ValueError(f"{path} is not a BEATs checkpoint (keys: {list(blob)})")
        return blob["cfg"], blob["model"]

    def _transformer_layers(self) -> nn.ModuleList:
        return self.encoder.layers

    def _truncate(self, layers: int | None, available: int) -> int:
        """Keep the first `layers` transformer blocks and DELETE the rest.

        Deleted, not skipped: an unused block still costs parameters, optimizer
        state and checkpoint bytes, and docs/architecture/06 §1 argues
        truncation precisely as a way to make two frontends affordable.
        """
        if layers is None:
            return available
        if not 1 <= layers <= available:
            raise ValueError(
                f"layers={layers} out of range for a {available}-layer checkpoint")
        self.encoder.layers = nn.ModuleList(list(self.encoder.layers)[:layers])
        return layers

    def _load_state(self, state: dict) -> None:
        keep = {}
        for k, v in state.items():
            if k.startswith("predictor"):
                continue                      # fine-tuned head; absent from our model
            if k.startswith("encoder.layers."):
                idx = int(k.split(".")[2])
                if idx >= self.n_layers:
                    continue                  # truncated away
            keep[k] = v
        missing, unexpected = self.load_state_dict(keep, strict=False)
        # `freq_pool`'s GeM exponent and any adapter weights are ours, not the
        # checkpoint's -- the same distinction `_apply_freeze` draws. Everything
        # else missing means the checkpoint and the model really disagree, and
        # a silently half-loaded encoder is worse than a build failure.
        missing = [m for m in missing
                   if not _is_adapter_param(m) and not m.startswith("freq_pool.")]
        if missing or unexpected:
            raise ValueError(
                f"BEATs checkpoint did not match the model. missing={missing}, "
                f"unexpected={unexpected}")

    # -- framing --------------------------------------------------------------

    def _mel_frames(self, n_samples: Tensor | int) -> Tensor | int:
        """Kaldi fbank frame count with snip_edges=True (torchaudio's default)."""
        if isinstance(n_samples, Tensor):
            return torch.clamp(
                (n_samples - self._FRAME_LENGTH) // self._FRAME_SHIFT + 1, min=0)
        return max(0, (int(n_samples) - self._FRAME_LENGTH) // self._FRAME_SHIFT + 1)

    def _frames_for(self, n_samples: Tensor | int) -> Tensor | int:
        """Patch-grid width: mel frames floor-divided by the patch stride."""
        mel = self._mel_frames(n_samples)
        if isinstance(mel, Tensor):
            return torch.clamp(mel // self.PATCH, min=1)
        return max(1, mel // self.PATCH)

    # -- encoding -------------------------------------------------------------

    def _fbank(self, wav_1d: Tensor) -> Tensor:
        import torchaudio.compliance.kaldi as ta_kaldi
        # 🔴 Shorter than one analysis window and kaldi's fbank asserts rather
        # than returning an empty tensor ("choose a window size 400 that is
        # [2, 399]"), so `_frames_for`'s promise of at least one frame was a lie
        # below 400 samples -- 25 ms. Zero-pad up to one window: the row owns
        # every sample involved, so this is within-row and rule 2.4 is untouched.
        # `test_audio_shorter_than_one_patch_still_yields_a_frame` used n=1000,
        # which still has 4 mel frames, so it never reached this path.
        if wav_1d.shape[-1] < self._FRAME_LENGTH:
            wav_1d = F.pad(wav_1d, (0, self._FRAME_LENGTH - wav_1d.shape[-1]))
        # Upstream scales to int16 range before fbank; the checkpoint's
        # normalisation constants are defined against that scale.
        src = wav_1d.unsqueeze(0).to(torch.float32) * (2 ** 15)
        # 🔴 Never under autocast. At int16 scale the power spectrum reaches
        # ~1e14, and fbank's mel matmul autocasts to fp16 (max 65504): measured
        # under fp16 autocast, every BEATs output was NaN -- at 9 layers too, so
        # candidate A had it. bf16's range hid it. MUTATION: drop this context
        # and test_fbank_is_immune_to_fp16_autocast fails.
        with torch.autocast(device_type=src.device.type, enabled=False):
            fbank = ta_kaldi.fbank(src, num_mel_bins=self.N_MELS,
                                   sample_frequency=self.audio.sample_rate,
                                   frame_length=25, frame_shift=10)
        return (fbank - self.FBANK_MEAN) / (2 * self.FBANK_STD)

    def _encode(self, wav: Tensor, lengths: Tensor | None = None) -> Tensor:
        """(B, S) -> (B, F', T', D), frequency-major.

        🔴 Each row is encoded over **its own valid prefix**, one at a time,
        and the results are padded afterwards. Encoding the padded batch in one
        pass would be faster and wrong: BEATs flattens the patch grid into a
        single `F' * T'` token sequence before self-attention, so a row's valid
        tokens are *not* a prefix of that sequence -- they are `T'`-strided runs,
        one per frequency band. Every published rule-2.4 defect in this repo
        (docs/pipelines/05) came from an operation that read the padded extent,
        and this is the same shape of mistake with a harder-to-see mask.
        """
        b, s = wav.shape
        if lengths is None:
            lengths = torch.full((b,), s, dtype=torch.long, device=wav.device)
        host_lengths = [int(n) for n in lengths.tolist()]
        widths = [int(self._frames_for(n)) for n in host_lengths]
        t_max = max(widths)

        if self.cfg.window_patches is not None:
            if self.cfg.batched_tokens:
                return self._encode_windowed_batched(wav, lengths, host_lengths, widths,
                                                     t_max)
            return self._encode_windowed(wav, lengths, widths, t_max)
        out = wav.new_zeros(b, self.N_MELS // self.PATCH, t_max, self.output_dim)
        for i in range(b):
            n = int(lengths[i].item())
            fbank = self._fbank(wav[i, :n])                        # (T_mel, 128)
            usable = widths[i] * self.PATCH
            if fbank.shape[0] < usable:
                # `_frames_for` floors to 1 frame, so audio shorter than one
                # patch (2800 samples, 175 ms) has fewer mel frames than the
                # patch it must fill. Repeat the last frame rather than pad with
                # zeros: zeros are -1.18 after the checkpoint's normalisation,
                # i.e. loud silence, not silence. Within-row only, so rule 2.4
                # is untouched.
                tail = fbank[-1:].expand(usable - fbank.shape[0], -1)
                fbank = torch.cat([fbank, tail], dim=0)
            fbank = fbank[:usable].unsqueeze(0).unsqueeze(0)       # (1, 1, T_mel', 128)
            x = self.patch_embedding(fbank)                        # (1, C, T', F')
            x = x.reshape(1, x.shape[1], -1).transpose(1, 2)       # (1, T'*F', C)
            x = self.layer_norm(x)
            if self.post_extract_proj is not None:
                x = self.post_extract_proj(x)
            x, _ = self.encoder(x)                                 # (1, T'*F', D)
            grid = x.view(1, widths[i], self.N_MELS // self.PATCH,
                          self.output_dim)                         # (1, T', F', D)
            out[i, :, :widths[i], :] = grid[0].permute(1, 0, 2)    # -> (F', T', D)
        return out

    def _tokens(self, wav_1d: Tensor, width: int) -> Tensor:
        """One row's valid prefix -> its projected patch tokens, (T' * F', D),
        time-major: each patch column's F' frequency tokens are adjacent."""
        fbank = self._fbank(wav_1d)
        usable = width * self.PATCH
        if fbank.shape[0] < usable:
            fbank = torch.cat([fbank, fbank[-1:].expand(usable - fbank.shape[0], -1)], 0)
        x = self.patch_embedding(fbank[:usable].unsqueeze(0).unsqueeze(0))
        x = x.reshape(1, x.shape[1], -1).transpose(1, 2)
        x = self.layer_norm(x)
        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)
        return x[0]

    def _encode_windowed(self, wav: Tensor, lengths: Tensor, widths: list[int],
                         t_max: int) -> Tensor:
        """`FrontendConfig.window_patches`: each row's patch grid is cut into
        windows of W patch columns (the filterbank is computed over the whole
        valid prefix first, so frame counts and masks are unchanged). Every
        window is encoded on its own -- no window ever sees padding or another
        row, so rule 2.4 holds -- and windows of equal length share one
        encoder call across the batch, which is where the speed comes from.
        """
        f = self.N_MELS // self.PATCH
        w = int(self.cfg.window_patches)
        # Every window is padded to w columns and the encoder is told which
        # tokens are padding: it zeroes them before its positional convolution
        # (which zero-pads at a sequence end anyway) and masks them out of
        # attention, so a valid token sees exactly what it would unpadded. One
        # encoder call for the whole batch: per-length calls made the step
        # launch-bound (3.7 s CPU for 1.05 s of GPU time on a mixed batch).
        toks, place = [], []
        for i in range(wav.shape[0]):
            tok = self._tokens(wav[i, :int(lengths[i].item())], widths[i])
            for start in range(0, widths[i], w):
                cols = min(w, widths[i] - start)
                piece = tok[start * f:(start + cols) * f]
                if cols < w:
                    piece = torch.cat([piece, piece.new_zeros((w - cols) * f, piece.shape[1])])
                toks.append(piece)
                place.append((i, start, cols))
        x = torch.stack(toks)
        pad = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
        for k, (_, _, cols) in enumerate(place):
            pad[k, cols * f:] = True
        y, _ = self.encoder(x, padding_mask=pad if bool(pad.any()) else None)
        grid = y.view(len(place), w, f, self.output_dim)
        out = wav.new_zeros(wav.shape[0], f, t_max, self.output_dim)
        for (i, start, cols), g in zip(place, grid):
            out[i, :, start:start + cols, :] = g[:cols].permute(1, 0, 2).to(out.dtype)
        return out


    # -- batched tokens (FrontendConfig.batched_tokens) ----------------------

    def _fbank_consts(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """The povey window, the (128, 257) mel matrix and kaldi's epsilon, as
        `ta_kaldi.fbank` builds them on every call -- built once per device."""
        cache = self.__dict__.setdefault("_fbank_cache", {})
        if device not in cache:
            import torchaudio.compliance.kaldi as ta_kaldi
            window = ta_kaldi._feature_window_function(
                ta_kaldi.POVEY, self._FRAME_LENGTH, 0.42, device, torch.float32)
            banks, _ = ta_kaldi.get_mel_banks(self.N_MELS, self._FFT,
                                              float(self.audio.sample_rate), 20.0,
                                              0.0, 100.0, -500.0, 1.0)
            banks = F.pad(banks, (0, 1)).to(device=device, dtype=torch.float32)
            cache[device] = (window, banks,
                             ta_kaldi._get_epsilon(device, torch.float32))
        return cache[device]

    def _fbank_batch(self, wav: Tensor, n_frames: int) -> Tensor:
        """`_fbank` for every row at once: (B, S) -> (B, n_frames, 128).

        Frame j of row i reads samples [160 j, 160 j + 400) of row i only, and
        every step after the framing is per frame, so row i's first
        `_mel_frames(lengths[i])` frames are `_fbank(wav[i, :lengths[i]])`'s
        frames -- the same operations in the same order as
        `ta_kaldi.fbank` with `_fbank`'s arguments (dither 0, DC removal,
        pre-emphasis 0.97, povey window, 512-point power spectrum, log mel).
        The raw-energy column kaldi also computes is not used and not computed.
        Frames past a row's own count read its zeroed padding; the caller never
        keeps them.
        """
        window, banks, eps = self._fbank_consts(wav.device)
        x = wav.to(torch.float32) * (2 ** 15)
        if x.shape[-1] < self._FRAME_LENGTH:
            x = F.pad(x, (0, self._FRAME_LENGTH - x.shape[-1]))
        with torch.autocast(device_type=x.device.type, enabled=False):
            frames = x.unfold(-1, self._FRAME_LENGTH, self._FRAME_SHIFT)[:, :n_frames]
            frames = frames - frames.mean(dim=-1, keepdim=True)
            prev = F.pad(frames, (1, 0), mode="replicate")[..., :-1]
            frames = (frames - 0.97 * prev) * window
            frames = F.pad(frames, (0, self._FFT - self._FRAME_LENGTH))
            power = torch.fft.rfft(frames).abs().pow(2.0)
            fbank = torch.max(torch.matmul(power, banks.T), eps).log()
        return (fbank - self.FBANK_MEAN) / (2 * self.FBANK_STD)

    def _encode_windowed_batched(self, wav: Tensor, lengths: Tensor,
                                 host_lengths: list[int], widths: list[int],
                                 t_max: int) -> Tensor:
        """`_encode_windowed` without the Python loops over rows and windows.

        Run 1's step was launch-bound (GPU ~53 % busy): a filterbank, a patch
        embedding and a window slice per row, a pad-mask write per window and a
        scatter per window back out. Here each is one batched op. The encoder
        sees the same windows, in the same (row, start) order, with the same
        zero padding and padding mask, so everything from the encoder on is
        unchanged; the filterbank and patch embedding run on a batch-shaped
        tensor, so kernel choice may round differently (fp32: ~1e-6).
        """
        b = wav.shape[0]
        f = self.N_MELS // self.PATCH
        w = int(self.cfg.window_patches)
        n_mel = [max(1, int(self._mel_frames(max(n, self._FRAME_LENGTH))))
                 for n in host_lengths]
        usable = t_max * self.PATCH
        fbank = self._fbank_batch(wav, max(n_mel))                   # (B, M, 128)
        # 🔴 Per-row sizes come from `lengths` on the device, never from
        # `torch.tensor(list, device=...)`: a host-to-device copy of a Python
        # list blocks until the GPU has drained its queue, and the row loop's
        # per-row copies (the mel matrix, `lengths[i].item()`) were exactly that
        # -- ~35 stalls a forward.
        dev_len = lengths.to(wav.device)
        if any(m < wd * self.PATCH for m, wd in zip(n_mel, widths)):
            # A row shorter than one patch repeats its last frame (`_encode`'s
            # tail rule): frame j reads min(j, n_mel - 1) of its own row.
            last = self._mel_frames(dev_len.clamp(min=self._FRAME_LENGTH)).clamp(min=1) - 1
            idx = torch.arange(usable, device=wav.device)[None, :].minimum(last[:, None])
            fbank = fbank.gather(1, idx[..., None].expand(-1, -1, self.N_MELS))
        else:
            fbank = fbank[:, :usable]
        x = self.patch_embedding(fbank.unsqueeze(1))                 # (B, C, T', F')
        x = x.reshape(b, x.shape[1], -1).transpose(1, 2)             # (B, T'*F', C)
        x = self.layer_norm(x)
        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)

        n_win = -(-t_max // w)
        span = n_win * w
        width_t = self._frames_for(dev_len)
        col_ok = torch.arange(span, device=wav.device)[None, :] < width_t[:, None]
        grid = F.pad(x.view(b, t_max, f, -1), (0, 0, 0, 0, 0, span - t_max))
        grid = grid.masked_fill(~col_ok[:, :, None, None], 0.0)
        # The windows that hold any valid column, in (row, start) order. Indexed
        # by a list built on the host and copied without blocking: a boolean
        # mask index would need `nonzero`, i.e. a device sync, to size its output.
        flat = [i * n_win + j for i, wd in enumerate(widths) for j in range(-(-wd // w))]
        flat = _index_to(flat, wav.device)
        toks = grid.view(b * n_win, w * f, -1).index_select(0, flat)   # (K, w*F', D)
        any_pad = any(wd % w for wd in widths)
        pad = (~col_ok.view(b * n_win, w)).index_select(0, flat).repeat_interleave(f, dim=1)
        y, _ = self.encoder(toks, padding_mask=pad if any_pad else None)
        out = y.new_zeros(b * n_win, w * f, self.output_dim, dtype=wav.dtype)
        out = out.index_copy(0, flat, y.to(wav.dtype))
        out = out.view(b, span, f, self.output_dim)[:, :t_max]
        out = out.masked_fill(~col_ok[:, :t_max, None, None], 0.0)
        return out.permute(0, 2, 1, 3).contiguous()


def _index_to(values: list[int], device: torch.device) -> Tensor:
    """A host list as a device index tensor, without a device sync: through
    pinned memory, the copy is queued on the stream instead of waited for."""
    t = torch.tensor(values, dtype=torch.long)
    if device.type != "cuda":
        return t
    return t.pin_memory().to(device, non_blocking=True)


class _BEATsEncoderConfig:
    """Adapter from a checkpoint's `cfg` dict to what `TransformerEncoder` reads."""

    def __init__(self, cfg: dict, *, freeze: bool):
        self.__dict__.update(cfg)
        # See BEATsFrontend's docstring: both of these are deliberate.
        self.encoder_layerdrop = 0.0
        if freeze:
            self.dropout = 0.0
            self.attention_dropout = 0.0
            self.activation_dropout = 0.0
            self.dropout_input = 0.0


class XLSRFrontend(Frontend):
    """XLS-R 300M (`facebook/wav2vec2-xls-r-300m`, Apache-2.0), truncated and
    LoRA-adapted, on `transformers.Wav2Vec2Model`.

    Loaded from a local directory only (`local_files_only=True`): the test
    server has no network, and a build that silently reached for the Hub would
    pass here and fail there.

    The encoder loop is ours, not `Wav2Vec2Model.forward`, because the upstream
    forward gets in the way of two of the three decisions below:

    🔴 **No hidden generator.** Upstream draws `torch.rand([])` per layer on
    *every* forward -- in eval and at `layerdrop: 0` too -- for LayerDrop, and
    applies SpecAugment (`mask_time_prob: 0.075`, drawn with numpy) in train
    mode. The first shifts the global torch stream by one draw per layer per
    batch; the second is a generator outside our seeding. Same reasoning as
    `BEATsFrontend`'s `encoder_layerdrop = 0`, and a test pins it.

    🔴 **The input normalisation is over the valid prefix.** Wav2Vec2 expects
    zero-mean unit-variance per utterance (`Wav2Vec2FeatureExtractor`,
    `do_normalize: true`). Taking the statistics over the padded row would make
    a file's features depend on how long its batch neighbours are -- rule 2.4.
    The pad is zeroed *after* normalisation, so it contributes nothing.

    ⚠️ **The final encoder LayerNorm is kept on the truncated stack.**
    `do_stable_layer_norm` is pre-LN: the residual stream is un-normalised
    between blocks and the checkpoint's `encoder.layer_norm` was trained on
    block 24's output. Applying it after block 12 is what HF itself does for a
    12-layer config, and it keeps the stream's scale bounded under fp16 --
    which is the reason it is kept, not a claim that it is the best readout.

    The CNN feature extractor is always frozen (even with `freeze: false`): it is
    the standard wav2vec2 fine-tuning recipe.
    """

    #: Config target names -> Wav2Vec2 attribute names, so one LoRA target list
    #: names the same projections in BEATs and XLS-R.
    TARGET_ALIASES = {"fc1": "intermediate_dense", "fc2": "output_dense"}
    _MIN_SAMPLES = 400              # the conv stack's receptive field: one frame

    def __init__(self, cfg: FrontendConfig, audio: AudioConfig):
        super().__init__(cfg, audio)
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        if cfg.weights is None:
            raise ValueError(
                "XLSRFrontend needs `weights`: a local directory holding "
                "facebook/wav2vec2-xls-r-300m (config.json + weights). A randomly-"
                "initialised XLS-R is a StubFrontend with a longer runtime.")
        if cfg.freq_pool.kind != "none" or cfg.n_freq is not None:
            raise ValueError(
                "XLS-R emits (B, T, D) with no frequency axis: set freq_pool "
                "{kind: none} and n_freq: null")
        if audio.sample_rate != 16000:
            raise ValueError(
                f"XLS-R is a 16 kHz model; audio.sample_rate is {audio.sample_rate}")
        hf_cfg = Wav2Vec2Config.from_pretrained(cfg.weights, local_files_only=True)
        hop = 1
        for s in hf_cfg.conv_stride:
            hop *= s
        expected_fps = audio.sample_rate / hop
        if abs(cfg.fps - expected_fps) > 1e-9:
            raise ValueError(
                f"XLS-R emits {expected_fps:g} fps (conv strides multiply to {hop} "
                f"samples at {audio.sample_rate} Hz) but the config says fps={cfg.fps}. "
                f"fps is what aligns two frontends onto a common time base, so a "
                f"wrong value misaligns evidence silently rather than failing.")
        if hf_cfg.hidden_size != cfg.output_dim:
            raise ValueError(
                f"checkpoint hidden_size={hf_cfg.hidden_size} but config "
                f"output_dim={cfg.output_dim}")
        if hf_cfg.feat_extract_norm != "layer":
            # A group-norm feature extractor normalises across TIME, so the padded
            # extent would reach the valid frames. XLS-R is `layer`; refuse the rest.
            raise ValueError(
                f"feat_extract_norm={hf_cfg.feat_extract_norm!r}: only 'layer' is "
                f"padding-invariant (group norm pools over time)")

        # See the class docstring. Dropouts are pretraining regularisers; on a
        # frozen encoder they only add RNG draws to a module we are not training.
        hf_cfg.layerdrop = 0.0
        hf_cfg.apply_spec_augment = False
        hf_cfg.mask_time_prob = 0.0
        hf_cfg.mask_feature_prob = 0.0
        if cfg.freeze:
            for k in ("hidden_dropout", "attention_dropout", "activation_dropout",
                      "feat_proj_dropout", "final_dropout"):
                setattr(hf_cfg, k, 0.0)

        model, info = Wav2Vec2Model.from_pretrained(
            cfg.weights, config=hf_cfg, local_files_only=True,
            attn_implementation="sdpa", output_loading_info=True)
        if info["missing_keys"] or info.get("mismatched_keys"):
            raise ValueError(
                f"XLS-R checkpoint did not match the model: missing="
                f"{info['missing_keys']}, mismatched={info.get('mismatched_keys')}")
        self.model = model

        self.n_layers = self._truncate(cfg.layers, len(model.encoder.layers))

        self.n_lora = 0
        if cfg.adapter.kind == "lora":
            self.n_lora = _apply_lora(self.model.encoder.layers, cfg.adapter.targets,
                                      cfg.adapter.rank, cfg.adapter.alpha,
                                      cfg.adapter.dropout, rename=self.TARGET_ALIASES,
                                      where="the XLS-R encoder")
        elif cfg.adapter.kind != "none":
            raise ValueError(f"unsupported adapter kind {cfg.adapter.kind!r} for XLS-R")

        self._apply_freeze()
        for p in self.model.feature_extractor.parameters():
            p.requires_grad_(False)

    def _transformer_layers(self) -> nn.ModuleList:
        return self.model.encoder.layers

    def _truncate(self, layers: int | None, available: int) -> int:
        """Keep the first `layers` blocks and DELETE the rest (as BEATs does)."""
        if layers is None:
            return available
        if not 1 <= layers <= available:
            raise ValueError(
                f"layers={layers} out of range for a {available}-layer checkpoint")
        self.model.encoder.layers = nn.ModuleList(list(self.model.encoder.layers)[:layers])
        self.model.config.num_hidden_layers = layers
        return layers

    # -- framing --------------------------------------------------------------

    def _frames_for(self, n_samples: Tensor | int) -> Tensor | int:
        """The conv stack's own arithmetic, via the model's helper.

        🔴 Not the nominal ceil(samples / 320): the stack is `floor((L - k)/s) + 1`
        per layer, one frame short of nominal at almost every length (4 s:
        nominal 200, actual 199). Audio shorter than one receptive field is
        padded up to it in `_encode`, so the floor here is 1, not 0.
        """
        if isinstance(n_samples, Tensor):
            n = n_samples.clamp(min=self._MIN_SAMPLES)
            return self.model._get_feat_extract_output_lengths(n).clamp(min=1)
        n = torch.tensor(max(int(n_samples), self._MIN_SAMPLES))
        return max(1, int(self.model._get_feat_extract_output_lengths(n)))

    # -- encoding -------------------------------------------------------------

    @staticmethod
    def _normalise(wav: Tensor, lengths: Tensor) -> Tensor:
        """Zero-mean unit-variance per row over ``[0, lengths)``, zeros after.

        Matches `Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm` (population
        variance, eps 1e-7). Always float32: under fp16 a 60 s row's sum of
        squares overflows.
        """
        x = wav.float()
        valid = (torch.arange(x.shape[-1], device=x.device)[None, :]
                 < lengths[:, None]).float()
        n = lengths.clamp(min=1).float()[:, None]
        mean = (x * valid).sum(-1, keepdim=True) / n
        var = (((x - mean) * valid) ** 2).sum(-1, keepdim=True) / n
        return (x - mean) / torch.sqrt(var + 1e-7) * valid

    def _encode(self, wav: Tensor, lengths: Tensor | None = None) -> Tensor:
        """(B, S) -> (B, T, 1024) at 50 fps.

        One padded pass, unlike BEATs: wav2vec2's token sequence *is* time, so a
        row's valid frames are a prefix and a key-padding mask is exact. What
        makes it exact, each pinned by tests/test_frontends_xlsr.py:

        * the CNN only sees valid samples in valid frames (`_frames_for` is its
          own arithmetic) and its `layer` norm is per frame, not over time;
        * padded frames are zeroed before the positional conv, which is what a
          lone row sees there anyway -- the conv's own zero padding;
        * attention masks padded keys in every block.
        """
        b, s = wav.shape
        if lengths is None:
            lengths = torch.full((b,), s, dtype=torch.long, device=wav.device)
        lengths = lengths.to(wav.device)
        x = self._normalise(wav, lengths)
        if s < self._MIN_SAMPLES:
            x = F.pad(x, (0, self._MIN_SAMPLES - s))
        m = self.model
        feats = m.feature_extractor(x).transpose(1, 2)          # (B, T, 512)
        hidden, _ = m.feature_projection(feats)                 # (B, T, 1024)

        t = hidden.shape[1]
        valid = self._frames_for(lengths)
        frame_mask = torch.arange(t, device=hidden.device)[None, :] < valid[:, None]
        hidden = hidden * frame_mask[..., None].to(hidden.dtype)
        enc = m.encoder
        attn_mask = enc._update_full_mask(frame_mask.long(), hidden)
        hidden = hidden + enc.pos_conv_embed(hidden)
        hidden = enc.dropout(hidden)
        for layer in enc.layers:
            hidden = layer(hidden, attention_mask=attn_mask)[0]
        return enc.layer_norm(hidden)


def build_frontend(cfg: FrontendConfig, audio: AudioConfig) -> Frontend:
    """Construct a frontend from config.

    `stub`, `beats` and `xlsr_300m` are implemented. The remaining names in
    docs/architecture/02 stay unwired on purpose: the licences for SSLAM, EAT
    and W2V-BERT 2.0 are unverified (09 C1-C3), and a licence that forbids
    third-party provision makes a checkpoint unusable *at all* here rather than
    merely unshippable. Downloading first and checking later is the wrong order.

    BEATs is exempt from that gate because its licence is not in question --
    MIT, read at origin -- which is exactly why 02 calls it "the licence-safe
    floor" and why candidate A was specified against it. XLS-R 300M is exempt
    for the same reason: Apache-2.0, read on the facebook/wav2vec2-xls-r-300m card.
    """
    if cfg.name == "stub":
        return StubFrontend(cfg, audio)
    if cfg.name == "beats":
        return BEATsFrontend(cfg, audio)
    if cfg.name == "xlsr_300m":
        return XLSRFrontend(cfg, audio)
    raise NotImplementedError(
        f"frontend {cfg.name!r} is not wired yet. 'stub', 'beats' and 'xlsr_300m' are "
        f"implemented; "
        f"the other checkpoints are gated on the licence verification in "
        f"docs/architecture/09-open-questions.md (C1-C3). Set `name: stub` with "
        f"`weights: null` to build and test the architecture without weights.")
