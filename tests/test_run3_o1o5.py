"""Run 3's two flag-gated features (docs/training/13 §3).

O1 -- the one-class (OC-Softmax) auxiliary loss on the voice and file heads. It is
training-only: with the flag off nothing may change, and with it on the submitted
probabilities may not change either.

O5 -- `file_head.mode = "max3"`: max(learned FILE, v * vp, m * mp), per file.
"""

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

from models.config import ConfigError, LossConfig, load_model_config, load_train_config
from models.heads import SEDHead, SEDOutput
from models.config import SEDHeadConfig
from models.losses import multitask_loss, oc_softmax_loss
from models.model import DeepVoiceNet, save_checkpoint

SR = 16_000
REPO = Path(__file__).resolve().parents[1]
OC_BRANCHES = ("voice", "file")


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_script",
                                                  REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def train():
    return _load_script("train")


def _cfg(oc=False, mode=None, name="b_stub"):
    cfg = load_model_config(REPO / "configs" / f"{name}.yaml")
    if oc:
        cfg = dataclasses.replace(cfg, branches={
            k: (dataclasses.replace(b, head=dataclasses.replace(b.head, oc=True))
                if k in OC_BRANCHES else b)
            for k, b in cfg.branches.items()})
    if mode is not None:
        cfg = dataclasses.replace(cfg, file_head=dataclasses.replace(cfg.file_head, mode=mode))
    return cfg


def _pair():
    """A flag-off model and a flag-on one carrying exactly its weights."""
    torch.manual_seed(0)
    off = DeepVoiceNet(_cfg()).eval()
    on = DeepVoiceNet(_cfg(oc=True)).eval()
    res = on.load_state_dict(off.state_dict(), strict=False)
    assert sorted(res.missing_keys) == [f"heads.{b}.oc_center" for b in sorted(OC_BRANCHES)]
    assert not res.unexpected_keys
    return off, on


def _targets(vp, mp, vf, mf):
    vp, mp, vf, mf = (torch.tensor(x, dtype=torch.float32) for x in (vp, mp, vf, mf))
    return {"voice_present": vp, "music_present": mp, "voice_fake": vf,
            "music_fake": mf, "file_fake": ((vp * vf) + (mp * mf)).clamp(max=1)}


TG = _targets([1, 1, 0, 1], [0, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1])


# --------------------------------------------------------------------------- #
# O1: the loss itself

def _sp(x):
    return math.log1p(math.exp(x))


def test_oc_softmax_matches_hand_values():
    # s = 1 (parallel), s = 0 (orthogonal); the scale of either vector is irrelevant
    emb = torch.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 0.5], [0.0, 0.5]])
    center = torch.tensor([2.0, 0.0])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    got = oc_softmax_loss(emb, center, labels, 20.0, 0.9, 0.2)
    want = torch.tensor([_sp(20 * (0.9 - 1)), _sp(20 * (1 - 0.2)),
                         _sp(20 * (0.9 - 0)), _sp(20 * (0 - 0.2))])
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
    # a cosine of 0.6 with other margins/alpha
    emb = torch.tensor([[0.6, 0.8], [0.6, 0.8]])
    got = oc_softmax_loss(emb, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), 5.0, 0.7, 0.1)
    torch.testing.assert_close(got, torch.tensor([_sp(5 * 0.1), _sp(5 * 0.5)]),
                               rtol=1e-6, atol=1e-6)


def test_oc_softmax_is_fp32_under_half_inputs():
    emb = torch.randn(4, 16).to(torch.bfloat16)
    got = oc_softmax_loss(emb, torch.randn(16), torch.tensor([0., 1., 0., 1.]), 20, .9, .2)
    assert got.dtype == torch.float32


def test_loss_config_rejects_nonsense_oc_settings():
    with pytest.raises(ConfigError):
        LossConfig(oc_weight=-1.0)
    with pytest.raises(ConfigError):
        LossConfig(oc_m_real=0.1, oc_m_fake=0.2)
    with pytest.raises(ConfigError):
        LossConfig(oc_alpha=0.0)


# --------------------------------------------------------------------------- #
# O1: the head

def test_head_embedding_is_the_attention_pooled_hidden_over_valid_frames():
    torch.manual_seed(0)
    head = SEDHead(8, SEDHeadConfig(hidden=16, oc=True)).eval()
    x = torch.randn(2, 10, 8)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1, 6:] = False
    out = head(x, mask)
    h = head.dense(x)                                    # (B, T, hidden)
    want = (out["attention"].unsqueeze(-1) * h).sum(1)
    torch.testing.assert_close(out["embedding"], want)
    # padded frames do not reach it
    x2 = x.clone()
    x2[1, 6:] = 1e3
    torch.testing.assert_close(head(x2, mask)["embedding"][1], out["embedding"][1])
    assert out["oc_center"] is head.oc_center


def test_flag_off_head_has_no_new_parameter_or_output():
    torch.manual_seed(0)
    head = SEDHead(8, SEDHeadConfig(hidden=16))
    assert not hasattr(head, "oc_center")
    assert "oc_center" not in dict(head.named_parameters())
    out = head(torch.randn(2, 5, 8))
    assert set(out) == {"clip_logits", "frame_logits", "attention", "mask"}
    assert out.get("embedding") is None


# --------------------------------------------------------------------------- #
# O1: the multitask loss

def _wav():
    torch.manual_seed(7)
    return torch.randn(4, SR * 2)


def test_flag_off_loss_is_bitwise_unchanged_by_any_oc_setting():
    """🔴 Run 2's objective: no oc head -> oc_weight/alpha/margins change nothing."""
    off, _ = _pair()
    with torch.no_grad():
        out = off(_wav())
    t0, p0 = multitask_loss(out, TG, off.cfg, LossConfig())
    t1, p1 = multitask_loss(out, TG, off.cfg, LossConfig(oc_weight=0.7, oc_alpha=3.0,
                                                         oc_m_real=0.5, oc_m_fake=-0.5))
    assert torch.equal(t0, t1)
    assert p0 == p1
    assert not any(k.endswith("/oc") for k in p0)


def test_oc_model_at_weight_zero_has_run2_total_and_outputs():
    off, on = _pair()
    wav = _wav()
    with torch.no_grad():
        a, b = off(wav), on(wav)
    for k in a:
        if k.startswith("_"):
            continue
        for f in ("clip_logits", "frame_logits", "attention", "mask"):
            assert torch.equal(a[k][f], b[k][f]), (k, f)
    ta, pa = multitask_loss(a, TG, off.cfg, LossConfig())
    tb, pb = multitask_loss(b, TG, on.cfg, LossConfig())
    assert torch.equal(ta, tb)
    assert {k: v for k, v in pb.items() if not k.endswith("/oc")} == pa
    assert set(pb) - set(pa) == {f"{x}/oc" for x in OC_BRANCHES}


def test_oc_term_is_masked_like_the_bce_and_enters_the_total():
    _, on = _pair()
    with torch.no_grad():
        out = on(_wav())
    lc = LossConfig(oc_weight=0.5)
    t0, p0 = multitask_loss(out, TG, on.cfg, LossConfig())
    t1, p1 = multitask_loss(out, TG, on.cfg, lc)
    for b in OC_BRANCHES:
        br = on.cfg.branches[b]
        y = TG[{"voice": "voice_fake", "file": "file_fake"}[b]]
        per = oc_softmax_loss(out[b]["embedding"], out[b]["oc_center"], y, 20.0, .9, .2)
        keep = (torch.ones(4, dtype=torch.bool) if br.masked_by is None
                else TG[br.masked_by].bool())
        assert p1[f"{b}/oc"] == pytest.approx(float(per[keep].mean()), rel=1e-6)
    # voice is masked by voice_present: file 2 has no voice, so its embedding is free
    assert TG["voice_present"][2] == 0
    out2 = {k: (SEDOutput(v) if k == "voice" else v) for k, v in out.items()}
    emb = out2["voice"]["embedding"].clone()
    emb[2] = -emb[2] * 100
    out2["voice"]["embedding"] = emb
    _, p2 = multitask_loss(out2, TG, on.cfg, lc)
    assert p2["voice/oc"] == p1["voice/oc"]
    want = float(t0) + 0.5 * sum(p1[f"{b}/oc"] for b in OC_BRANCHES)
    assert float(t1) == pytest.approx(want, rel=1e-6)
    # label smoothing does not reach the OC term
    _, p3 = multitask_loss(out, TG, on.cfg, dataclasses.replace(lc, label_smoothing=0.2))
    assert all(p3[f"{b}/oc"] == p1[f"{b}/oc"] for b in OC_BRANCHES)


def test_oc_gradient_reaches_the_centre_and_the_head():
    _, on = _pair()
    on.train()
    out = on(_wav())
    total, _ = multitask_loss(out, TG, on.cfg, LossConfig(oc_weight=0.1))
    total.backward()
    for b in OC_BRANCHES:
        assert on.heads[b].oc_center.grad is not None
        assert on.heads[b].oc_center.grad.abs().sum() > 0


def test_oc_weight_zero_keeps_the_centre_in_the_graph():
    """DDP deadlocks on a parameter that gets no gradient at all."""
    _, on = _pair()
    on.train()
    total, _ = multitask_loss(on(_wav()), TG, on.cfg, LossConfig())
    total.backward()
    assert all(on.heads[b].oc_center.grad is not None for b in OC_BRANCHES)


def test_tensor_parts_carry_the_oc_diagnostic():
    _, on = _pair()
    with torch.no_grad():
        out = on(_wav())
    _, pf = multitask_loss(out, TG, on.cfg, LossConfig(oc_weight=0.1))
    _, pt = multitask_loss(out, TG, on.cfg, LossConfig(oc_weight=0.1), tensor_parts=True)
    for b in OC_BRANCHES:
        assert float(pt[f"{b}/oc"]) == pf[f"{b}/oc"]


# --------------------------------------------------------------------------- #
# O1: inference is untouched

@pytest.mark.parametrize("mode", ["learned", "max3"])
def test_submission_probs_identical_with_oc_on_and_off(mode):
    off, on = _pair()
    off.cfg = dataclasses.replace(off.cfg, file_head=dataclasses.replace(
        off.cfg.file_head, mode=mode))
    on.cfg = dataclasses.replace(on.cfg, file_head=dataclasses.replace(
        on.cfg.file_head, mode=mode))
    wav = _wav()
    lengths = torch.tensor([SR * 2, SR + 321, SR * 2 - 5, SR])
    with torch.no_grad():
        a = off.submission_probs(off(wav, lengths))
        b = on.submission_probs(on(wav, lengths))
    assert a.keys() == b.keys()
    for c in a:
        assert torch.equal(a[c], b[c]), c


# --------------------------------------------------------------------------- #
# O1: init from a run-2 checkpoint

def test_init_from_loads_a_run2_state_dict_leaving_only_oc_center(train, tmp_path, capsys):
    off, _ = _pair()
    ck = tmp_path / "scored.pt"
    save_checkpoint(off, ck)
    torch.manual_seed(123)
    on = DeepVoiceNet(_cfg(oc=True))
    centres = {b: on.heads[b].oc_center.detach().clone() for b in OC_BRANCHES}
    train.init_from(on, str(ck))
    sd_off, sd_on = off.state_dict(), on.state_dict()
    assert set(sd_on) - set(sd_off) == {f"heads.{b}.oc_center" for b in OC_BRANCHES}
    for k, v in sd_off.items():
        assert torch.equal(sd_on[k], v), k
    for b in OC_BRANCHES:
        assert torch.equal(on.heads[b].oc_center.detach(), centres[b])
    assert "oc_center" in capsys.readouterr().out


def test_init_from_still_fails_on_real_mismatches(train, tmp_path):
    off, _ = _pair()
    sd = off.state_dict()
    # a missing non-oc key
    missing = {k: v for k, v in sd.items() if not k.startswith("heads.voice.cla")}
    torch.save({"config": {}, "state_dict": missing}, tmp_path / "a.pt")
    with pytest.raises(RuntimeError, match="heads.voice.cla"):
        train.init_from(DeepVoiceNet(_cfg(oc=True)), str(tmp_path / "a.pt"))
    # an unexpected key
    torch.save({"config": {}, "state_dict": {**sd, "heads.voice.bogus": torch.zeros(1)}},
               tmp_path / "b.pt")
    with pytest.raises(RuntimeError, match="bogus"):
        train.init_from(DeepVoiceNet(_cfg()), str(tmp_path / "b.pt"))
    # an oc checkpoint into a flag-off model: oc_center is unexpected there
    _, on = _pair()
    save_checkpoint(on, tmp_path / "c.pt")
    with pytest.raises(RuntimeError, match="oc_center"):
        train.init_from(DeepVoiceNet(_cfg()), str(tmp_path / "c.pt"))
    # a shape mismatch
    bad = dict(sd)
    bad["heads.voice.cla.weight"] = torch.zeros(1, 3, 1)
    torch.save({"config": {}, "state_dict": bad}, tmp_path / "d.pt")
    with pytest.raises(RuntimeError, match="size mismatch"):
        train.init_from(DeepVoiceNet(_cfg(oc=True)), str(tmp_path / "d.pt"))


# --------------------------------------------------------------------------- #
# O1: the run-3 configs

def _yaml(name):
    return yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))


def test_c_run2_oc_is_c_run2_plus_oc_on_voice_and_file():
    a, b = _yaml("c_run2.yaml"), _yaml("c_run2_oc.yaml")
    assert b["branches"]["voice"].pop("head") == {"oc": True}
    assert b["branches"]["file"]["head"].pop("oc") is True
    a.pop("name"), b.pop("name")
    assert a == b
    cfg = load_model_config(REPO / "configs" / "c_run2_oc.yaml")
    assert {k for k, br in cfg.branches.items() if br.head.oc} == set(OC_BRANCHES)


def test_train_run3_oc_is_train_run2_plus_the_oc_loss():
    a, b = _yaml("train_run2.yaml"), _yaml("train_run3_oc.yaml")
    oc = {k: b["loss"].pop(k) for k in ("oc_weight", "oc_alpha", "oc_m_real", "oc_m_fake")}
    assert oc == {"oc_weight": 0.02, "oc_alpha": 20.0, "oc_m_real": 0.9, "oc_m_fake": 0.2}
    assert a == b
    assert load_train_config(REPO / "configs" / "train_run3_oc.yaml").loss.oc_weight == 0.02


# --------------------------------------------------------------------------- #
# O5: max3

def _fake_out(file_z, voice_z, vp_z, music_z, mp_z):
    def so(z):
        z = torch.tensor(z, dtype=torch.float32)
        return SEDOutput(clip_logits=z, frame_logits=z[:, None].repeat(1, 3),
                         attention=torch.full((len(z), 3), 1 / 3),
                         mask=torch.ones(len(z), 3, dtype=torch.bool))
    return {"file": so(file_z), "voice": so(voice_z), "v_pres": so(vp_z),
            "music": so(music_z), "m_pres": so(mp_z)}


def _probs(out, mode):
    return DeepVoiceNet(_cfg(mode=mode)).submission_probs(out)


def test_max3_is_the_max_of_its_three_inputs():
    torch.manual_seed(3)
    z = [torch.randn(64).mul(3).tolist() for _ in range(5)]
    out = _fake_out(*z)
    learned, mx, m3 = (_probs(out, m)["FILE_FAKE_PROB"] for m in ("learned", "max", "max3"))
    p = _probs(out, "learned")
    v = p["VOICE_FAKE_PROB"] * p["VOICE_PRESENT_PROB"]
    m = p["MUSIC_FAKE_PROB"] * p["MUSIC_PRESENT_PROB"]
    assert m3.dtype == torch.float64
    assert (m3 >= learned).all() and (m3 >= v).all() and (m3 >= m).all()
    assert torch.equal(m3, torch.maximum(learned, mx))
    # every input wins somewhere, so it is not a relabelled two-way max
    assert (m3 == learned).any() and (m3 > learned).any()


def test_max3_equals_learned_when_the_component_heads_are_below():
    out = _fake_out([0.5, -1.0, 2.0], [-3.0, -4.0, 5.0], [5.0, 5.0, -9.0],
                    [-6.0, -5.0, 4.0], [3.0, 3.0, -8.0])
    learned = _probs(out, "learned")["FILE_FAKE_PROB"]
    assert torch.equal(_probs(out, "max3")["FILE_FAKE_PROB"], learned)


def test_max3_other_columns_are_untouched():
    torch.manual_seed(4)
    out = _fake_out(*[torch.randn(8).tolist() for _ in range(5)])
    a, b = _probs(out, "learned"), _probs(out, "max3")
    for c in a:
        if c != "FILE_FAKE_PROB":
            assert torch.equal(a[c], b[c])


def test_max3_is_per_file():
    """Rule 2.4: a file's score must not depend on what is batched with it."""
    torch.manual_seed(5)
    z = [torch.randn(6).mul(2).tolist() for _ in range(5)]
    together = _probs(_fake_out(*z), "max3")["FILE_FAKE_PROB"]
    for i in range(6):
        alone = _probs(_fake_out(*[[c[i]] for c in z]), "max3")["FILE_FAKE_PROB"]
        assert torch.equal(alone[0], together[i])
    # and end to end through a real model with ragged lengths
    model = DeepVoiceNet(_cfg(mode="max3")).eval()
    torch.manual_seed(6)
    wav = torch.randn(3, SR * 3)
    lengths = torch.tensor([SR * 3, SR * 2 + 1, SR + 7])
    with torch.no_grad():
        both = model.submission_probs(model(wav, lengths))["FILE_FAKE_PROB"]
        for i in range(3):
            n = int(lengths[i])
            one = model.submission_probs(model(wav[i:i + 1, :n], lengths[i:i + 1]))
            torch.testing.assert_close(one["FILE_FAKE_PROB"][0], both[i], rtol=0, atol=1e-6)


def test_max3_is_accepted_by_the_config_and_unknown_modes_are_not():
    from models.config import validate_model_config
    validate_model_config(_cfg(mode="max3"))
    with pytest.raises(ConfigError):
        validate_model_config(_cfg(mode="max4"))


# --------------------------------------------------------------------------- #
# O5: packaging

def test_package_submission_file_mode_round_trip(tmp_path, monkeypatch):
    pkg = _load_script("package_submission")
    torch.manual_seed(0)
    model = DeepVoiceNet(_cfg(name="a_stub")).eval()
    run = tmp_path / "run"
    run.mkdir()
    save_checkpoint(model, run / "scored.pt")
    (run / "processing.json").write_text('{"chain": 1}', encoding="utf-8")
    out = tmp_path / "sub"
    fe = next(iter(model.cfg.frontends))
    monkeypatch.setattr(sys, "argv", [
        "package_submission.py", "--out", str(out), "--scored", str(run / "scored.pt"),
        "--weights", f"{fe}={tmp_path}", "--no-zip", "--file-mode", "max3"])
    assert pkg.main() == 0
    from models.model import load_checkpoint
    shipped = load_checkpoint(out / "model" / "scored.pt")
    assert shipped.cfg.file_head.mode == "max3"
    meta = json.loads((out / "model" / "members.json").read_text())
    assert meta["file_mode"] == {"trained": "learned", "shipped": "max3"}
    assert meta["scored"] == [str((run / "scored.pt").resolve())]
    # weights untouched; only the FILE column differs from the trained mode
    for k, v in model.state_dict().items():
        assert torch.equal(shipped.state_dict()[k], v), k
    wav = torch.randn(2, SR * 2)
    with torch.no_grad():
        a = model.submission_probs(model(wav))
        b = shipped.submission_probs(shipped(wav))
        c = pkg.with_file_mode(model, "max3").submission_probs(model(wav))
    for col in a:
        assert torch.equal(b[col], c[col]), col
        if col != "FILE_FAKE_PROB":
            assert torch.equal(a[col], b[col]), col


def test_package_submission_refuses_file_stack_with_a_file_mode(tmp_path, monkeypatch):
    pkg = _load_script("package_submission")
    monkeypatch.setattr(sys, "argv", [
        "package_submission.py", "--out", str(tmp_path / "x"), "--scored", "a.pt",
        "--weights", "a=b", "--file-stack", "s.json", "--file-mode", "max3"])
    with pytest.raises(SystemExit, match="file-stack"):
        pkg.main()
