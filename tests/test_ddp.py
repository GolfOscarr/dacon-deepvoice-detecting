"""Data-parallel training (docs/training/14): CPU, gloo, 2 processes, stub model.

What is claimed, and what here would go red if it stopped being true:

* T1 -- one process, or a process group of ONE, is the historical loop bitwise
  (`test_a_world_of_one_is_the_single_process_loop_bitwise`; and the whole of
  tests/test_loop.py, which runs unchanged).
* T2 -- the ranks' slices of every global batch are that global batch
  (`rank_micro_batches`), and every rank draws the same corpus.
* T3 / T3b -- the gradient DDP steps on equals the single-process gradient on the
  whole global batch, including when one rank (or one micro-batch) carries no
  file for a masked head, which is exactly where a per-rank masked mean differs.
  Paired with the mutation: the naive per-slice mean measurably differs.
* T4 -- a DDP run stopped mid-pass and resumed reproduces the uninterrupted DDP
  run bitwise, with dropout and the EMA on; a resume at another world is refused.
* T5 -- only rank 0 writes, and what it writes loads strictly into a bare model.
* T6 -- a multi-group stage is refused under DDP before any collective.
* T7 -- `frontend_hold_steps` freezes the frontend without touching the heads'
  steps, then starts the frontend's own warm-up.
* T8 -- every parameter that requires grad reaches the optimizer, including a
  new top-level module (a layer fusion).
* T9 -- `scripts/train.py` under torchrun: both ranks exit 0, rank 0 writes the
  scored weights.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from loop_fixtures import (_dataset, _flat, _max_abs_diff, _model, _same,
                           _train_cfg, corpus, model_cfg)  # noqa: F401 (fixtures)
from models.model import DeepVoiceNet
from training.collate import collate
from training.distributed import SINGLE, Dist, current
from training.loop import (LoopConfig, forward_backward, rank_micro_batches,
                           train_stage)
from training.stages import stage_plan, trainable_parameters

REPO = Path(__file__).resolve().parents[1]
WORLD = 2


# --------------------------------------------------------------------------- #
# spawn plumbing


def _join(rank: int, world: int, init: str) -> None:
    torch.set_num_threads(1)
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", init_method=f"file://{init}", rank=rank,
                            world_size=world)


def _spawn(fn, tmp_path: Path, *args, world: int = WORLD) -> None:
    init = tmp_path / f"pg-{fn.__name__}-{len(list(tmp_path.iterdir()))}"
    mp.spawn(fn, args=(world, str(init), *args), nprocs=world, join=True)


def _no_dropout(cfg):
    return dataclasses.replace(cfg, branches={
        k: dataclasses.replace(b, head=dataclasses.replace(b.head, dropout_in=0.0,
                                                           dropout_out=0.0))
        for k, b in cfg.branches.items()})


# --------------------------------------------------------------------------- #
# T2 -- the slicing


def test_rank_slices_partition_every_global_batch():
    batches = [list(range(i * 8, i * 8 + 8)) for i in range(3)]
    for world, per_rank, accum in ((2, 4, 1), (2, 2, 2), (4, 1, 2), (1, 4, 2)):
        per = [rank_micro_batches(batches, rank=r, world=world, per_rank=per_rank,
                                  accum=accum) for r in range(world)]
        for step, b in enumerate(batches):
            union = [i for r in range(world) for m in per[r][step] for i in m]
            assert union == b, (world, per_rank, accum)
            assert all(len(m) == per_rank for r in range(world) for m in per[r][step])


def test_a_short_global_batch_is_refused_rather_than_padded():
    with pytest.raises(ValueError, match="global batch of 3"):
        rank_micro_batches([[0, 1, 2]], rank=0, world=2, per_rank=2, accum=1)


# --------------------------------------------------------------------------- #
# T3 / T3b -- the gradient


def _global_batch(corpus, n=4):
    ds = _dataset(corpus, n=n)
    batch = collate([ds[i] for i in range(n)])
    # One rank (and one micro-batch) with NO voice file, and music split unevenly:
    # the case where a per-slice masked mean and the global one disagree.
    batch["targets"]["voice_present"] = torch.tensor([1, 1, 0, 0])
    batch["targets"]["music_present"] = torch.tensor([1, 0, 0, 1])
    return batch, ds.ship


def _slice(batch, idx):
    idx_t = torch.tensor(idx)
    return {"wav": batch["wav"][idx_t], "lengths": batch["lengths"][idx_t],
            "targets": {k: v[idx_t] for k, v in batch["targets"].items()}}


def _grads(cfg, batch, ship, micros_idx, *, d=SINGLE, accum=1, net_wrap=False):
    torch.manual_seed(0)
    model = DeepVoiceNet(cfg)
    model.train()
    net = model
    if net_wrap:
        from torch.nn.parallel import DistributedDataParallel
        net = DistributedDataParallel(model, broadcast_buffers=False)
    train_cfg = _train_cfg(stage="joint")
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    forward_backward(net, [_slice(batch, m) for m in micros_idx], ship_cfg=ship,
                     device=torch.device("cpu"), precision="fp32",
                     loss_cfg_model=cfg, loss_cfg=train_cfg.loss, scaler=scaler,
                     d=d, accum=accum)
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def _w_grads(rank, world, init, cfg, batch, ship, per_rank, accum, out):
    _join(rank, world, init)
    try:
        d = current()
        micros = rank_micro_batches([list(range(per_rank * world * accum))], rank=rank,
                                    world=world, per_rank=per_rank, accum=accum)[0]
        g = _grads(cfg, batch, ship, micros, d=d, accum=accum, net_wrap=True)
        if rank == 0:
            torch.save(g, out)
    finally:
        dist.destroy_process_group()


def _close(a, b, atol=1e-6):
    assert a.keys() == b.keys()
    worst = max(float((a[k] - b[k]).abs().max()) for k in a)
    assert worst <= atol, f"max |grad diff| {worst:.3e}"


@pytest.mark.parametrize("per_rank,accum", [(2, 1), (1, 2)])
def test_ddp_gradient_is_the_single_process_gradient(corpus, model_cfg, tmp_path,
                                                     per_rank, accum):
    cfg = _no_dropout(model_cfg)
    batch, ship = _global_batch(corpus)
    ref = _grads(cfg, batch, ship, [[0, 1, 2, 3]])
    out = tmp_path / "g.pt"
    _spawn(_w_grads, tmp_path, cfg, batch, ship, per_rank, accum, out)
    _close(torch.load(out), ref)


def test_accumulation_alone_is_the_full_batch_gradient(corpus, model_cfg):
    cfg = _no_dropout(model_cfg)
    batch, ship = _global_batch(corpus)
    ref = _grads(cfg, batch, ship, [[0, 1, 2, 3]])
    acc = _grads(cfg, batch, ship, [[0, 1], [2, 3]], accum=2)
    _close(acc, ref)


def test_the_naive_per_slice_mean_is_not_the_global_gradient(corpus, model_cfg):
    """The mutation T3 exists for: averaging each half's own masked mean (what DDP
    does without `GlobalNorm`) moves the voice/music heads' gradient."""
    cfg = _no_dropout(model_cfg)
    batch, ship = _global_batch(corpus)
    ref = _grads(cfg, batch, ship, [[0, 1, 2, 3]])
    a = _grads(cfg, batch, ship, [[0, 1]])
    b = _grads(cfg, batch, ship, [[2, 3]])
    naive = {k: (a[k] + b[k]) / 2 for k in ref}
    worst = max(float((naive[k] - ref[k]).abs().max()) for k in ref if k.startswith("heads.voice"))
    assert worst > 1e-4, "the fixture no longer separates the two normalisations"


# --------------------------------------------------------------------------- #
# T1, T4, T5, T6 -- whole stages under DDP


def _loop(out, **kw):
    base = dict(out_dir=out, n_buckets=1, ema_decay=0.99)
    return LoopConfig(**{**base, **kw})


def _w_stage(rank, world, init, corpus, cfg, out, loop_kw, stage, resume, result_path):
    _join(rank, world, init)
    try:
        model = _model(cfg)
        torch.manual_seed(1234)
        rank_out = Path(out) / f"r{rank}"
        try:
            res = train_stage(model, _dataset(corpus, n=6),
                              train_cfg=_train_cfg(stage=stage, epochs=2, batch_size=1),
                              loop_cfg=_loop(rank_out, **loop_kw), stage=stage,
                              resume_from=resume)
        except ValueError as e:
            if rank == 0:
                torch.save({"error": str(e)}, result_path)
            return
        if rank == 0:
            torch.save({"weights": _flat(model),
                        "ema": res.ema.state_dict_for(model) if res.ema else None,
                        "steps": res.steps, "digests": res.pass_digests,
                        "ckpts": [str(c.path) for c in res.checkpoints],
                        "sampler": [c.sampler for c in res.checkpoints]}, result_path)
    finally:
        dist.destroy_process_group()


def _w_world_one(rank, world, init, corpus, cfg, out, result_path, join):
    # `join=False` is the plain single-process loop, run in the same kind of
    # process (one torch thread) so the two differ in nothing but the group.
    if join:
        _join(rank, world, init)
    else:
        torch.set_num_threads(1)
    try:
        assert current() == SINGLE
        model = _model(cfg)
        torch.manual_seed(1234)
        train_stage(model, _dataset(corpus, n=6),
                    train_cfg=_train_cfg(stage="joint", epochs=2),
                    loop_cfg=_loop(Path(out)))
        torch.save(_flat(model), result_path)
    finally:
        if join:
            dist.destroy_process_group()


def test_a_world_of_one_is_the_single_process_loop_bitwise(corpus, model_cfg, tmp_path):
    plain, grouped = tmp_path / "plain.pt", tmp_path / "w1.pt"
    _spawn(_w_world_one, tmp_path, corpus, model_cfg, str(tmp_path / "p"), plain, False,
           world=1)
    _spawn(_w_world_one, tmp_path, corpus, model_cfg, str(tmp_path / "g"), grouped, True,
           world=1)
    assert _same(torch.load(plain), torch.load(grouped))


def _stage_run(tmp_path, corpus, cfg, name, *, stage="joint", resume=None, **loop_kw):
    path = tmp_path / f"{name}.pt"
    _spawn(_w_stage, tmp_path, corpus, cfg, str(tmp_path / name), loop_kw, stage,
           resume, path)
    return torch.load(path, weights_only=False)


def test_a_ddp_resume_mid_pass_reproduces_the_uninterrupted_run_bitwise(
        corpus, model_cfg, tmp_path):
    # dropout ON (the stub heads' defaults) and the EMA on: the per-rank RNG and
    # the rank-0 EMA are what a resume has to restore.
    assert model_cfg.branches["voice"].head.dropout_in > 0
    full = _stage_run(tmp_path, corpus, model_cfg, "full")
    assert full["steps"] == 6                  # 6 specs / global batch 2, 2 passes
    part = _stage_run(tmp_path, corpus, model_cfg, "part", max_steps=2)
    stopped = part["sampler"][-1]
    assert stopped.pass_index == 0 and stopped.batch_index == 2, stopped   # mid-pass
    resumed = _stage_run(tmp_path, corpus, model_cfg, "resumed", resume=part["ckpts"][-1])
    assert resumed["steps"] == full["steps"]
    assert _same(full["weights"], resumed["weights"]), (
        f"diverged by {_max_abs_diff(full['weights'], resumed['weights']):.3e}")
    assert _same(full["ema"], resumed["ema"])
    # T2 on real draws: the ranks trained on the corpus a 1-process run draws
    one = _model(model_cfg)
    res1 = train_stage(one, _dataset(corpus, n=6),
                       train_cfg=_train_cfg(stage="joint", epochs=2, batch_size=2),
                       loop_cfg=_loop(tmp_path / "one"))
    assert full["digests"] == res1.pass_digests


def test_a_ddp_checkpoint_refuses_a_resume_at_another_world(corpus, model_cfg, tmp_path):
    part = _stage_run(tmp_path, corpus, model_cfg, "part", max_steps=2)
    with pytest.raises(ValueError, match="world 2"):
        train_stage(_model(model_cfg), _dataset(corpus, n=6),
                    train_cfg=_train_cfg(stage="joint", epochs=2, batch_size=2),
                    loop_cfg=_loop(tmp_path / "one"), resume_from=part["ckpts"][-1])


def test_only_rank_zero_writes_and_its_weights_load_strictly(corpus, model_cfg, tmp_path):
    run = _stage_run(tmp_path, corpus, model_cfg, "w", checkpoint_every=1, log_every=1)
    assert not (tmp_path / "w" / "r1").exists(), "rank 1 wrote files"
    r0 = tmp_path / "w" / "r0"
    assert (r0 / "train_log.jsonl").exists() and run["ckpts"]
    blob = torch.load(run["ckpts"][-1], map_location="cpu", weights_only=False)
    assert not any(k.startswith("module.") for k in blob["state_dict"])
    DeepVoiceNet(model_cfg).load_state_dict(blob["state_dict"], strict=True)
    info = blob["extra"]["dist"]
    assert (info["world"], info["global_batch"], info["grad_accum"]) == (2, 2, 1)
    assert len(info["rng_by_rank"]) == 2
    logs = [json.loads(line) for line in (r0 / "train_log.jsonl").read_text().splitlines()]
    assert len(logs) == run["steps"] and logs[0]["world"] == 2


def test_a_multi_group_stage_is_refused_under_ddp(corpus, model_cfg, tmp_path):
    got = _stage_run(tmp_path, corpus, model_cfg, "indep", stage="independent")
    assert "branch groups" in got.get("error", ""), got


# --------------------------------------------------------------------------- #
# T7 -- the encoder hold (one process: it is not a DDP property)


def _frontend(model):
    return {k: v.detach().clone() for k, v in model.frontends.state_dict().items()
            if v.is_floating_point()}


def _heads(model):
    return {k: v.detach().clone() for k, v in model.heads.state_dict().items()}


def _hold_run(corpus, cfg, out, *, freeze_frontend=False, **loop_kw):
    model = _model(cfg)
    if freeze_frontend:
        for p in model.frontends.parameters():
            p.requires_grad_(False)
    init = _frontend(model)
    torch.manual_seed(1234)
    train_stage(model, _dataset(corpus, n=8),
                train_cfg=_train_cfg(stage="joint", epochs=2),
                loop_cfg=_loop(out, **loop_kw))
    return model, init


def test_the_frontend_hold_freezes_the_frontend_and_leaves_the_heads_alone(
        corpus, model_cfg, tmp_path):
    assert any(p.requires_grad for p in _model(model_cfg).frontends.parameters()), \
        "the stub must have a trainable frontend parameter (GeM) for this to test anything"
    held, init = _hold_run(corpus, model_cfg, tmp_path / "held",
                           frontend_hold_steps=2, max_steps=2)
    assert _same(_frontend(held), init)
    frozen, _ = _hold_run(corpus, model_cfg, tmp_path / "frozen", freeze_frontend=True,
                          max_steps=2)
    # the held gradients stayed out of the clip norm and the optimizer entirely
    assert _same(_heads(held), _heads(frozen))
    after, _ = _hold_run(corpus, model_cfg, tmp_path / "after", frontend_hold_steps=2,
                         max_steps=3)
    assert not _same(_frontend(after), init), "the frontend never started"


def test_after_the_hold_the_frontend_warms_up_from_its_own_start(corpus, model_cfg,
                                                                  tmp_path):
    out = tmp_path / "lr"
    _hold_run(corpus, model_cfg, out, frontend_hold_steps=2, lr_schedule="cosine",
              warmup_steps=3, log_every=1)
    rows = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    by_step = {r["step"]: r for r in rows}
    assert by_step[1]["lr_factor_frontend"] == 0.0 and by_step[2]["lr_factor_frontend"] == 0.0
    # step index 2 (logged as step 3) is the frontend's warm-up step 0: 1/3, not
    # the heads' factor, which is already past its warm-up
    assert by_step[3]["lr_factor_frontend"] == pytest.approx(1 / 3)
    assert by_step[3]["lr_factor"] == 1.0


# --------------------------------------------------------------------------- #
# T8 -- nothing that requires grad is left out of the optimizer


def test_every_trainable_parameter_reaches_the_optimizer(model_cfg):
    model = _model(model_cfg)
    model.fusion = torch.nn.Linear(4, 4)          # a new top-level module (14 D6)
    plan = stage_plan("joint", model.cfg)
    got = {id(p) for p in trainable_parameters(model, plan, plan.branch_groups[0])}
    want = {id(p) for p in model.parameters() if p.requires_grad}
    missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in got]
    assert got == want, f"never optimised: {missing}"


# --------------------------------------------------------------------------- #
# T9 -- scripts/train.py under torchrun


@pytest.fixture(scope="module")
def integration_corpus(tmp_path_factory):
    """The layout `scripts/train.py` reads -- tests/test_train_integration.py's
    `corpus` fixture, built the same way (imported names would clash with the
    loop fixtures' `corpus`)."""
    import yaml
    from processing.config import load_processing_config
    from processing.splits import build_and_check, write_outputs
    from training.folds import FoldConfig
    from training.synthetic import synthetic_manifest, write_synthetic_corpus
    root = tmp_path_factory.mktemp("ddp-integration")
    manifest = synthetic_manifest(n_per_pool=60, n_whole_file=30, seed=0,
                                  duration_range=(4.5, 9.0))
    a = manifest.index[manifest.pool == "A"]
    manifest.loc[a, "speaker_ref_id"] = [f"{s}_spk{i % 3}"
                                         for i, s in enumerate(manifest.loc[a, "source_name"])]
    write_synthetic_corpus(manifest, root / "corpus", seed=0)
    mdir = root / "manifests"
    mdir.mkdir()
    manifest.to_parquet(mdir / "manifest.parquet", index=False)
    fcfg = FoldConfig(n_folds=2, probe_share=0.1, assigned_at="2026-01-01T00:00:00+00:00",
                      probe_min_real_hours=0.0, probe_min_real_atoms=1,
                      caveat_min_role_hours=0.0)
    plan, report, summary = build_and_check(manifest, fcfg)
    write_outputs(mdir, plan, report, summary)
    raw = yaml.safe_load((REPO / "configs" / "processing_v1.yaml").read_text())
    raw["draw"]["duration_range"] = [4.0, 8.0]
    raw["draw"]["augments"] = [x for x in raw["draw"]["augments"] if x["name"] != "rir"]
    raw["render"]["root"] = str(root / "corpus")
    raw["render"]["cache_root"] = None
    raw["folds"]["n_folds"] = 2
    raw["folds"]["probe_min_real_hours"] = 0.0
    raw["folds"]["probe_min_real_atoms"] = 1
    raw["folds"]["assigned_at"] = "2026-01-01T00:00:00+00:00"
    raw["loop"]["device"] = "cpu"
    cfg_path = root / "processing_test.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    load_processing_config(cfg_path)
    return root, mdir, cfg_path


def test_train_py_under_torchrun_exits_clean_and_rank_zero_writes(integration_corpus,
                                                                   tmp_path):
    root, mdir, cfg_path = integration_corpus
    out = tmp_path / "ddp"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nproc_per_node", "2", str(REPO / "scripts" / "train.py"),
           "--manifest-dir", str(mdir), "--processing", str(cfg_path),
           "--model", "configs/a_stub.yaml", "--out", str(out),
           "--folds", "0", "--stages", "joint", "--epochs", "1", "--batch-size", "2",
           "--draws", "48", "--eval-n", "24", "--max-steps", "2",
           "--device", "cpu", "--allow-unquotable"]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env,
                       timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    assert "global batch 4" in r.stdout
    scored = out / "fold0" / "scored.pt"
    assert scored.exists() and (out / "fold0" / "ledger_row.json").exists()
    blob = torch.load(scored, map_location="cpu", weights_only=False)
    assert not any(k.startswith("module.") for k in blob["state_dict"])
    # 48 draws over 4 duration buckets fill enough global batches of 4 that the
    # stage really steps, and --max-steps 2 truncates it: rank 0 wrote the
    # truncation checkpoint, at world 2, after two real optimizer steps
    ck = torch.load(out / "fold0" / "joint-truncated.pt", map_location="cpu",
                    weights_only=False)
    assert ck["extra"]["dist"]["world"] == 2 and ck["global_step"] == 2


def _torchrun_refusal(integration_corpus, tmp_path, *extra):
    root, mdir, cfg_path = integration_corpus
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nproc_per_node", "2", str(REPO / "scripts" / "train.py"),
           "--manifest-dir", str(mdir), "--processing", str(cfg_path),
           "--model", "configs/a_stub.yaml", "--out", str(tmp_path / "x"),
           "--device", "cpu", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=600)


def test_train_py_refuses_several_folds_under_ddp(integration_corpus, tmp_path):
    r = _torchrun_refusal(integration_corpus, tmp_path, "--folds", "0,1", "--stages", "joint")
    assert r.returncode != 0 and "one fold or --all-data" in (r.stdout + r.stderr)


def test_train_py_refuses_several_stages_under_ddp(integration_corpus, tmp_path):
    r = _torchrun_refusal(integration_corpus, tmp_path, "--folds", "0",
                          "--stages", "joint,codec_aware")
    assert r.returncode != 0 and "one stage" in (r.stdout + r.stderr)


def test_dist_defaults_describe_one_process():
    assert current() == SINGLE and not SINGLE.enabled and SINGLE.main
    assert Dist(rank=1, world=2).enabled and not Dist(rank=1, world=2).main


# --------------------------------------------------------------------------- #
# The epoch draw, sharded across ranks (docs/training/14 §8: 13 min per pass)


def _w_draw(rank, world, init, corpus, n, out):
    _join(rank, world, init)
    try:
        from training.dataset import draw_epoch
        from training.sampler import Sampler
        from loop_fixtures import DRAW
        manifest, _, _ = corpus
        specs = draw_epoch(Sampler(manifest, DRAW), n, epoch=3, seed=7)
        if rank == 1:                   # a non-zero rank must hold the full list too
            torch.save([s.to_dict() for s in specs], out)
    finally:
        dist.destroy_process_group()


def test_the_sharded_epoch_draw_is_the_serial_draw(corpus, tmp_path):
    from training.sampler import Sampler
    from loop_fixtures import DRAW
    manifest, _, _ = corpus
    n = 11                               # not a multiple of the world size
    serial = [s.to_dict() for s in Sampler(manifest, DRAW).epoch_specs(n, epoch=3, seed=7)]
    out = tmp_path / "draw.pt"
    _spawn(_w_draw, tmp_path, corpus, n, out)
    assert torch.load(out, weights_only=False) == serial


def test_set_epoch_to_the_current_epoch_keeps_the_same_list(corpus):
    ds = _dataset(corpus, n=6)
    before = ds.specs
    ds.set_epoch(0)
    assert ds.specs is before            # no redraw: it is already epoch 0's draw
    ds.set_epoch(1)
    assert ds.specs != before


def test_a_world_change_resume_is_allowed_only_opted_in_and_at_the_same_global_batch(
        corpus, model_cfg, tmp_path, monkeypatch):
    part = _stage_run(tmp_path, corpus, model_cfg, "part", max_steps=2)   # world 2 x 1
    ck = part["ckpts"][-1]
    monkeypatch.setenv("DDP_ALLOW_WORLD_CHANGE", "1")
    with pytest.raises(ValueError, match="world 2"):        # global batch 4 != 2: refused
        train_stage(_model(model_cfg), _dataset(corpus, n=6),
                    train_cfg=_train_cfg(stage="joint", epochs=2, batch_size=4),
                    loop_cfg=_loop(tmp_path / "g4"), resume_from=ck)
    res = train_stage(_model(model_cfg), _dataset(corpus, n=6),     # world 1 x 2 = 2: allowed
                      train_cfg=_train_cfg(stage="joint", epochs=2, batch_size=2),
                      loop_cfg=_loop(tmp_path / "g2"), resume_from=ck)
    assert res.steps == 6
