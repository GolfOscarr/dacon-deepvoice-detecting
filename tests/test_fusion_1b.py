"""The scaled model (docs/training/13 M1-M3, 14 §5): `xlsr_1b`, the learnable layer
fusion and the partial init from a run-2 checkpoint.

CPU only, on a tiny random wav2vec2 saved to disk (the XLS-R wrapper only loads
from a directory), so the real 1B snapshot is not needed. The partial-init tests
need the BEATs checkpoint, because refusing an init that loads nothing of BEATs is
part of the contract.
"""

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.config import (AdapterConfig, ConfigError, FusionConfig, dump_config,
                           load_model_config, _model_from_dict)
from models.frontends import build_frontend
from models.model import DeepVoiceNet, _WeightedLayerSum, save_checkpoint

REPO = Path(__file__).resolve().parents[1]
SR = 16_000
BEATS = os.environ.get("DACON_BEATS_WEIGHTS", "/data/project/private/dacon-weights/beats")
needs_beats = pytest.mark.skipif(
    not os.path.exists(os.path.join(BEATS, "BEATs_iter3_plus_AS2M.pt")),
    reason=f"no BEATs checkpoint under {BEATS}")
#: the last commit before the fusion existed: fusion-off must still compute its numbers
PRE_FUSION = "9131764"


def _tiny_dir(root: Path, width: int, depth: int = 4) -> str:
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    torch.manual_seed(width)
    cfg = Wav2Vec2Config(hidden_size=width, num_hidden_layers=depth, num_attention_heads=4,
                         intermediate_size=2 * width, conv_dim=(8,) * 7,
                         num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=4,
                         do_stable_layer_norm=True, feat_extract_norm="layer")
    path = root / f"w2v-{width}"
    Wav2Vec2Model(cfg).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("w2v")
    return {w: _tiny_dir(root, w) for w in (16, 32)}


def _stub_audio():
    return load_model_config(REPO / "configs" / "b_stub.yaml").frontends["audio"]


def _cfg(weights, *, width=32, layers=3, fused=True, name="xlsr_1b", audio=None):
    """c_1b_fusion.yaml with a tiny speech trunk and (by default) a stub audio one."""
    cfg = load_model_config(REPO / "configs" / "c_1b_fusion.yaml")
    speech = dataclasses.replace(
        cfg.frontends["speech"], name=name, weights=weights, output_dim=width, layers=layers,
        adapter=dataclasses.replace(cfg.frontends["speech"].adapter, rank=2),
        fusion=FusionConfig(kind="weighted" if fused else "none"))
    cfg = dataclasses.replace(cfg, frontends={"audio": audio or _stub_audio(),
                                              "speech": speech})
    return _model_from_dict(dump_config(cfg))       # validated, as a YAML load would be


def _ragged(seed=0):
    g = torch.Generator().manual_seed(seed)
    n0, n1 = SR * 3 + 123, SR * 2 + 77
    wav = torch.zeros(2, n0)
    wav[0] = torch.randn(n0, generator=g) * 0.1
    wav[1, :n1] = torch.randn(n1, generator=g) * 0.3
    return wav, torch.tensor([n0, n1])


def _perturb(model, seed=1):
    """LoRA B and the fusion logits start at zero; move them so they matter."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_b" in n or n.startswith("fusion."):
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return model


# --------------------------------------------------------------------------- #
# xlsr_1b and the config

def test_xlsr_1b_takes_its_width_from_the_checkpoint(tiny):
    cfg = _cfg(tiny[32])
    fe = build_frontend(cfg.frontends["speech"], cfg.audio)
    assert fe.output_dim == 32 and fe.n_layers == 3
    with pytest.raises(ValueError, match="hidden_size"):
        build_frontend(dataclasses.replace(cfg.frontends["speech"], output_dim=1280), cfg.audio)


def test_c_1b_fusion_is_c_run2_with_the_1b_trunk_and_fusion():
    a = dump_config(load_model_config(REPO / "configs" / "c_run2.yaml"))
    b = dump_config(load_model_config(REPO / "configs" / "c_1b_fusion.yaml"))
    sa, sb = a["frontends"].pop("speech"), b["frontends"].pop("speech")
    assert (sb["name"], sb["layers"], sb["output_dim"]) == ("xlsr_1b", 24, 1280)
    assert sb["fusion"] == {"kind": "weighted"} and sa["fusion"] == {"kind": "none"}
    assert sb["adapter"] == sa["adapter"]
    for k in ("name", "layers", "output_dim", "fusion"):
        sa.pop(k), sb.pop(k)
    assert sa == sb
    a.pop("name"), b.pop("name")
    assert a == b


@pytest.mark.parametrize("mutate, match", [
    (lambda c: {**c, "fusion": {"kind": "attentive"}}, "fusion.kind"),
    (lambda c: {**c, "name": "stub", "weights": None, "layers": None,
                "adapter": {"kind": "none"}, "fusion": {"kind": "weighted"}}, "fusion is implemented"),
])
def test_a_bad_fusion_config_is_refused(mutate, match):
    d = dump_config(load_model_config(REPO / "configs" / "c_1b_fusion.yaml"))
    d["frontends"]["speech"] = mutate(d["frontends"]["speech"])
    with pytest.raises(ConfigError, match=match):
        _model_from_dict(d)


def test_fusion_with_distill_is_refused():
    d = dump_config(load_model_config(REPO / "configs" / "c_1b_fusion.yaml"))
    d["distill"]["enabled"] = True
    with pytest.raises(ConfigError, match="distill"):
        _model_from_dict(d)


# --------------------------------------------------------------------------- #
# the fusion

def test_fused_frontend_hands_over_every_kept_layer_with_the_same_mask(tiny):
    on = build_frontend(_cfg(tiny[32]).frontends["speech"], _cfg(tiny[32]).audio).eval()
    off = build_frontend(_cfg(tiny[32], fused=False).frontends["speech"],
                         _cfg(tiny[32]).audio).eval()
    off.load_state_dict(on.state_dict())
    wav, lengths = _ragged()
    with torch.no_grad():
        layers, m_on = on(wav, lengths)
        last, m_off = off(wav, lengths)
    assert isinstance(layers, tuple) and len(layers) == 3
    assert all(h.shape == last.shape for h in layers)
    assert torch.equal(m_on, m_off)
    # the unfused output is the final LayerNorm of the last layer
    assert torch.equal(off.model.encoder.layer_norm(layers[-1]), last)
    assert not any(p.requires_grad for p in on.model.encoder.layer_norm.parameters())


def test_fusion_at_init_is_the_uniform_mean_of_the_normalised_layers(tiny):
    model = DeepVoiceNet(_cfg(tiny[32])).eval()
    wav, lengths = _ragged()
    with torch.no_grad():
        layers, _ = model.frontends["speech"](wav, lengths)
        for branch in ("voice", "file"):
            got = model.fusion(branch, "speech", layers)
            want = torch.stack([F.layer_norm(h, h.shape[-1:], eps=1e-5) for h in layers]).mean(0)
            assert torch.allclose(got, want, atol=1e-6), branch
    assert sorted(n for n, _ in model.named_parameters() if n.startswith("fusion.")) == \
        ["fusion.logits.file.speech", "fusion.logits.voice.speech"]


def test_model_output_shapes_match_the_unfused_model(tiny):
    wav, lengths = _ragged()
    with torch.no_grad():
        a = DeepVoiceNet(_cfg(tiny[32])).eval()(wav, lengths)
        b = DeepVoiceNet(_cfg(tiny[32], fused=False)).eval()(wav, lengths)
    assert a.keys() == b.keys()
    for k in a:
        for f in ("clip_logits", "frame_logits"):
            assert a[k][f].shape == b[k][f].shape, (k, f)


def test_a_fused_file_scores_the_same_alone_and_inside_a_padded_batch(tiny):
    """Rule 2.4 through the fusion: the per-layer norm is per frame, so padded
    frames cannot reach a valid one."""
    model = _perturb(DeepVoiceNet(_cfg(tiny[32]))).eval()
    wav, lengths = _ragged(2)
    with torch.no_grad():
        batched = model.submission_probs(model(wav, lengths))
        for i in range(2):
            n = int(lengths[i])
            solo = model.submission_probs(model(wav[i:i + 1, :n], lengths[i:i + 1]))
            for col in solo:
                assert abs(float(batched[col][i] - solo[col][0])) < 1e-5, (i, col)


def test_the_streaming_backward_is_the_autograd_backward():
    torch.manual_seed(0)
    hs = [torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    p = torch.softmax(torch.randn(3, dtype=torch.float64), 0).requires_grad_()
    assert torch.autograd.gradcheck(lambda p, *h: _WeightedLayerSum.apply(p, *h), (p, *hs))


def test_gradients_reach_the_fusion_and_the_speech_lora(tiny):
    model = _perturb(DeepVoiceNet(_cfg(tiny[32]))).train()
    wav, lengths = _ragged()
    out = model(wav, lengths)
    sum(out[k]["clip_logits"].sum() for k in out if not k.startswith("_")).backward()
    for n, p in model.fusion.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, n
    lora = [p for n, p in model.frontends["speech"].named_parameters() if "lora_" in n]
    assert lora and all(p.grad is not None and p.grad.abs().sum() > 0 for p in lora)
    # every trainable parameter got a gradient: DDP needs no find_unused_parameters
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not unused, unused


def test_fusion_trains_with_the_heads_not_the_frontends(tiny):
    from training.loop import LoopConfig, _param_groups
    from training.stages import stage_plan, trainable_parameters
    model = DeepVoiceNet(_cfg(tiny[32]))
    plan = stage_plan("joint", model.cfg)
    params = trainable_parameters(model, plan, plan.branch_groups[0])
    ids = {id(p) for p in params}
    assert all(id(p) in ids for p in model.fusion.parameters())
    assert ids == {id(p) for p in model.parameters() if p.requires_grad}
    groups = _param_groups(model, params, 1e-4, LoopConfig())
    heads = {id(p) for p in groups[0]["params"]}
    assert groups[0]["lr"] == 1e-4 and all(id(p) in heads for p in model.fusion.parameters())


def test_grad_checkpointing_gives_the_same_fusion_gradients(tiny):
    wav, lengths = _ragged()

    def grads(ckpt):
        model = _perturb(DeepVoiceNet(_cfg(tiny[32]))).train()
        if ckpt:
            assert model.frontends["speech"].enable_grad_checkpointing() == 3
        out = model(wav, lengths)
        sum(out[k]["clip_logits"].sum() for k in ("voice", "file")).backward()
        return {n: p.grad.clone() for n, p in model.named_parameters()
                if p.grad is not None and (n.startswith("fusion.") or "lora_" in n)}

    a, b = grads(False), grads(True)
    assert a.keys() == b.keys() and a
    for n in a:
        assert torch.allclose(a[n], b[n], atol=1e-6, rtol=1e-5), n


_PRE_FUSION_SCRIPT = """
import json, sys, torch
from models.config import _model_from_dict
from models.model import DeepVoiceNet
cfg = _model_from_dict(json.load(open(sys.argv[1])))
torch.manual_seed(0)
model = DeepVoiceNet(cfg).eval()
model.load_state_dict(torch.load(sys.argv[2]))
wav, lengths = torch.load(sys.argv[3])
with torch.no_grad():
    out = model(wav, lengths)
torch.save({k: v["clip_logits"] for k, v in out.items() if not k.startswith("_")}, sys.argv[4])
"""


def test_fusion_off_is_bitwise_the_pre_fusion_code(tiny, tmp_path):
    """The same config, weights and input through the code as it was before the
    fusion (commit PRE_FUSION) and through today's with ``fusion: none``."""
    ok = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", PRE_FUSION],
                        capture_output=True).returncode == 0
    if not ok:
        pytest.skip(f"commit {PRE_FUSION} not available")
    old = tmp_path / "old"
    old.mkdir()
    subprocess.run(f"git -C {REPO} archive {PRE_FUSION} models metrics | tar -x -C {old}",
                   shell=True, check=True)
    cfg = _cfg(tiny[32], fused=False, name="xlsr_300m")     # xlsr_1b was not wired then
    model = _perturb(DeepVoiceNet(cfg)).eval()
    assert model.fusion is None
    d = dump_config(cfg)
    for fe in d["frontends"].values():
        assert fe.pop("fusion") == {"kind": "none"}
    (tmp_path / "cfg.json").write_text(json.dumps(d))
    torch.save(model.state_dict(), tmp_path / "sd.pt")
    wav, lengths = _ragged(5)
    torch.save((wav, lengths), tmp_path / "in.pt")
    (tmp_path / "run.py").write_text(_PRE_FUSION_SCRIPT)
    subprocess.run([sys.executable, str(tmp_path / "run.py"), str(tmp_path / "cfg.json"),
                    str(tmp_path / "sd.pt"), str(tmp_path / "in.pt"), str(tmp_path / "out.pt")],
                   check=True, cwd=old, env={**os.environ, "PYTHONPATH": str(old)})
    theirs = torch.load(tmp_path / "out.pt")
    with torch.no_grad():
        ours = model(wav, lengths)
    assert theirs.keys() == {k for k in ours if not k.startswith("_")}
    for k, v in theirs.items():
        assert torch.equal(ours[k]["clip_logits"], v), k


# --------------------------------------------------------------------------- #
# partial init

@pytest.fixture(scope="module")
def train():
    spec = importlib.util.spec_from_file_location("train_script", REPO / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _beats():
    fe = load_model_config(REPO / "configs" / "c_run2.yaml").frontends["audio"]
    return dataclasses.replace(fe, weights=BEATS, layers=1)


@needs_beats
def test_partial_init_loads_beats_and_its_heads_and_leaves_the_rest_fresh(
        tiny, train, tmp_path, capsys):
    torch.manual_seed(0)
    # "run-2 T7": a narrower speech trunk, no fusion
    src = _perturb(DeepVoiceNet(_cfg(tiny[16], width=16, layers=2, fused=False,
                                     name="xlsr_300m", audio=_beats())), seed=7)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(0.01)                    # nothing may coincide with a fresh init
    save_checkpoint(src, tmp_path / "scored.pt")
    torch.manual_seed(1)
    dst = DeepVoiceNet(_cfg(tiny[32], audio=_beats()))
    fresh = {k: v.clone() for k, v in dst.state_dict().items()}
    report = train.init_partial(dst, str(tmp_path / "scored.pt"))
    s, d = src.state_dict(), dst.state_dict()

    for k in d:
        loaded = k.startswith(("frontends.audio.", "heads.music.", "heads.v_pres.",
                               "heads.m_pres."))
        assert torch.equal(d[k], s[k] if loaded else fresh[k]), k
        assert (k in report["loaded"]) == loaded, k
    # the voice head's later layers match T7's shapes, but the head stays fresh whole
    same_shape = [k for k in d if k.startswith("heads.voice.") and k in s
                  and s[k].shape == d[k].shape]
    assert same_shape and all(k in report["fresh"] for k in same_shape)
    assert all(k in report["fresh"] for k in d if k.startswith(("fusion.", "frontends.speech.")))
    out = capsys.readouterr().out
    assert "loaded:" in out and "fusion.logits" in out and "heads.voice" in out


@needs_beats
def test_partial_init_refuses_a_checkpoint_with_no_beats(tiny, train, tmp_path):
    src = DeepVoiceNet(_cfg(tiny[16], width=16, layers=2, fused=False, name="xlsr_300m",
                            audio=_beats()))
    sd = {k: v for k, v in src.state_dict().items() if not k.startswith("frontends.audio.")}
    torch.save({"config": {}, "state_dict": sd}, tmp_path / "nobeats.pt")
    with pytest.raises(RuntimeError, match="BEATs"):
        train.init_partial(DeepVoiceNet(_cfg(tiny[32], audio=_beats())),
                           str(tmp_path / "nobeats.pt"))


def test_init_partial_needs_init_weights(train, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["train.py", "--manifest-dir", str(tmp_path),
                                      "--out", str(tmp_path), "--init-partial"])
    with pytest.raises(SystemExit, match="--init-weights"):
        train.main()


def test_adapter_config_is_untouched_by_fusion():
    """The fusion is its own field: turning it on changes no other speech knob."""
    fe = load_model_config(REPO / "configs" / "c_1b_fusion.yaml").frontends["speech"]
    assert fe.adapter == AdapterConfig(kind="lora", rank=16, alpha=32.0,
                                       targets=("q_proj", "k_proj", "v_proj", "out_proj",
                                                "fc1", "fc2"))
