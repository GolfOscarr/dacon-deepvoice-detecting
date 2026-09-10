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

__all__ = ["Frontend", "StubFrontend", "BEATsFrontend", "LoRALinear",
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
                alpha: float, dropout: float) -> int:
    """Wrap every `nn.Linear` in `module` whose attribute name is in `targets`."""
    wrapped = 0
    for child in module.modules():
        for attr in targets:
            layer = getattr(child, attr, None)
            if isinstance(layer, nn.Linear):
                setattr(child, attr, LoRALinear(layer, rank, alpha, dropout))
                wrapped += 1
    return wrapped


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
                                      cfg.adapter.dropout)
            if self.n_lora == 0:
                raise ValueError(
                    f"adapter.targets={cfg.adapter.targets} matched no nn.Linear in "
                    f"the BEATs encoder -- a silently-inert adapter is the failure "
                    f"`test_no_config_field_is_silently_ignored` exists to prevent")
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
        # Upstream scales to int16 range before fbank; the checkpoint's
        # normalisation constants are defined against that scale.
        src = wav_1d.unsqueeze(0).to(torch.float32) * (2 ** 15)
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
        widths = [int(self._frames_for(int(n))) for n in lengths.tolist()]
        t_max = max(widths)

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


def build_frontend(cfg: FrontendConfig, audio: AudioConfig) -> Frontend:
    """Construct a frontend from config.

    `stub` and `beats` are implemented. The remaining names in
    docs/architecture/02 stay unwired on purpose: the licences for SSLAM, EAT
    and W2V-BERT 2.0 are unverified (09 C1-C3), and a licence that forbids
    third-party provision makes a checkpoint unusable *at all* here rather than
    merely unshippable. Downloading first and checking later is the wrong order.

    BEATs is exempt from that gate because its licence is not in question --
    MIT, read at origin -- which is exactly why 02 calls it "the licence-safe
    floor" and why candidate A was specified against it.
    """
    if cfg.name == "stub":
        return StubFrontend(cfg, audio)
    if cfg.name == "beats":
        return BEATsFrontend(cfg, audio)
    raise NotImplementedError(
        f"frontend {cfg.name!r} is not wired yet. 'stub' and 'beats' are implemented; "
        f"the other checkpoints are gated on the licence verification in "
        f"docs/architecture/09-open-questions.md (C1-C3). Set `name: stub` with "
        f"`weights: null` to build and test the architecture without weights.")
