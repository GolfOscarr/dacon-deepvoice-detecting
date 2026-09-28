# 14 · DDP training for the scaled model (XLS-R-1B@24 + layer fusion)

*Branch `feat/ddp-1b`, cut from `feat/training` at `c6e57d2`. Written 2026-09-26 ~18:30 KST.
Implementation plan for the owner (who implements DDP and the 1B model; doc 13 §9, HANDOFF_RUN3
§4). Every code reference was read at `c6e57d2`. Figures marked **ESTIMATE** are extrapolated
from the encoder benchmark (`docs/training/12` §7), not measured on this model. Rev 2 folds in an
independent review against the code (1 blocker, 4 major, 9 minor; all addressed in place).*

## 0 · Brief

- **One Slurm job, `torchrun --standalone`, N GPUs on the debug node**, one process per GPU.
  Parameters are replicated (DDP), not sharded: only LoRA + heads + GeM + fusion train, so
  gradients are small, and the frozen trunk fits on every H200.
- **The data plan does not change; it is only sliced.** Every rank builds the same pass plan at
  the **global** batch size (`per-rank batch × world`) and takes its own contiguous slice of
  each global batch. Renders are pure functions of their spec, so a W-GPU run trains on exactly
  the batches a 1-GPU run at batch `W·B` would. `SamplerState` keeps its meaning (index of the
  global batch), and bitwise resume carries over.
- **The loss is normalised globally.** `_masked_mean` divides by the *rank's* present count. Under
  DDP's gradient averaging that silently reweights the masked heads (music, voice). The present
  count gets all-reduced so the DDP gradient equals the single-process one.
- **Two traps the review found:** a fusion module outside `model.frontends` is silently never
  optimised unless `trainable_parameters` learns about it (D6), and gradient accumulation must
  not divide the globally normalised masked heads again (D5).
- **Rank 0 owns every side effect:** EMA, checkpoints (with every rank's RNG state), the
  `train_log.jsonl`, evaluation and `scored.pt`. The other ranks leave the process group after
  training, before rank 0 evaluates, so no NCCL timeout can fire during a long evaluation.
- **Scope: stage `joint` only** (the only stage run 2 and run 3 use). `independent` changes the
  trainable set between branch passes, which a DDP wrapper built once cannot follow, so it is
  refused under DDP.
- **Tests first on CPU (gloo, 2 processes, stub model), then a 2-GPU smoke on the known 300M@12
  config, then a 1B memory/throughput probe.** Only then the 1B main run.

| # | piece | where | size |
|---|---|---|---|
| D0 | process-group helpers, device, seeding | new `training/distributed.py` | S |
| D1 | global plan → rank slice | `training/loop.py` (`train_stage`, `_iter_batches`) | S |
| D2 | DDP wrap, unwrapped model for everything else | `training/loop.py` | S |
| D3 | globally normalised masked mean | `models/losses.py` (`_masked_mean`, `multitask_loss`) | S |
| D4 | rank-0 I/O, per-rank RNG in checkpoints, barriers, eval hand-off | `training/loop.py`, `training/checkpoint.py`, `scripts/train.py` | M |
| D5 | optional gradient accumulation (same global batch on 2 or 8 GPUs) | `training/loop.py` | S |
| D6 | encoder hold (frontend grads dropped for the first K steps, then its own warm-up; DDP-safe) | `training/loop.py` (`lr_factor` per group) | S |
| D7 | launcher | new `scripts/train_ddp.sbatch` | S |
| D8 | tests | new `tests/test_ddp.py` | M |
| D9 | GPU smoke, memory probe, resume on GPU, scaling | runs only | — |

## 1 · What the current code assumes (and where DDP breaks it)

| # | assumption today | where | consequence under DDP |
|---|---|---|---|
| A1 | one GPU per invocation, by design | `scripts/train.py:37` (docstring) | docstring and `--device` handling change: device becomes `cuda:{LOCAL_RANK}` |
| A2 | the pass plan is built at `train_cfg.batch_size` and every batch is trained | `training/loop.py:456` (`pass_plan`), `:473` (`total_steps`) | must be built at the global size and sliced per rank (D1); `total_steps` then counts global steps, so the cosine schedule stays right |
| A3 | the render pool renders whole batches | `training/loop.py:263` (`_iter_batches`), `:426` (pool) | each rank renders only its slice; one pool per rank |
| A4 | `model` is the object stepped, EMA'd, checkpointed and scored | `training/loop.py:372`, `:520`, `:582`; `scripts/train.py:242`, `:265`, `:311` | forward/backward goes through the DDP wrapper; **everything else must use the unwrapped module**, or state-dict keys gain a `module.` prefix and `init_from`/`load_checkpoint`/packaging fail strictly |
| A5 | masked mean over the *local* batch | `models/losses.py:70-98` | DDP averages rank gradients, so the effective weight of a masked head becomes the mean of per-rank masked means, not the global masked mean (§3) |
| A6 | a zero-present batch contributes `per_sample.sum() * 0.0` | `models/losses.py:96-98` | already DDP-safe: the graph stays connected, so no unused-parameter hang. Keep it |
| A7 | O1's `oc_center` is kept in the graph at `oc_weight 0` | `models/losses.py:255-266` | already DDP-safe; keep it |
| A8 | a new optimizer per branch group; the trainable set changes between S1 passes | `training/loop.py:435`, `training/stages.py:219` | DDP registers gradient hooks for the parameters that require grad at wrap time. Stage `joint` has one group, so wrap once; **refuse `independent` under DDP** |
| A9 | one RNG state per checkpoint | `training/checkpoint.py:151-176` | heads use dropout (0.25 in / 0.5 out, `models/heads.py:137-140`), so each rank's CUDA/CPU generator is state; save all ranks' (D4) |
| A10 | `torch.manual_seed(cfg.seed)` before building the model | `models/model.py:61`, `scripts/train.py:142` | identical init on every rank (good; DDP also broadcasts at wrap). **After** the wrap, reseed per rank (`seed + rank`) so dropout masks differ across ranks |
| A11 | layer checkpointing wraps layers with `checkpoint(..., use_reentrant=False)` | `models/frontends.py:295-320` | the non-reentrant form is the DDP-compatible one; no change. It stays off unless memory needs it (§5) |
| A12 | no BatchNorm anywhere in `models/`, `models/vendor` included (grep: none); XLS-R-1B uses `feat_extract_norm: layer` | — | no SyncBatchNorm; wrap with `broadcast_buffers=False` |

## 2 · Design, piece by piece

### D0 · `training/distributed.py` (new)

```python
@dataclass(frozen=True)
class Dist:
    rank: int = 0
    world: int = 1
    local_rank: int = 0
    @property
    def enabled(self) -> bool: return self.world > 1
    @property
    def main(self) -> bool: return self.rank == 0

def init_from_env(timeout_min: int = 30) -> Dist        # reads RANK/WORLD_SIZE/LOCAL_RANK; nccl on cuda, gloo on cpu
def barrier(d: Dist) -> None
def all_reduce_sum(t: Tensor, d: Dist) -> Tensor        # no-op when not enabled
def all_gather_object(obj, d: Dist) -> list
def teardown(d: Dist) -> None
```

- `world == 1` (no `torchrun`) must take **exactly today's code path**, with no process group
  and no wrapper. That is how the 8-task arrays keep running unchanged, and test T1 (§4) pins it.
- Device: `torch.cuda.set_device(local_rank)`; `LoopConfig.device = f"cuda:{local_rank}"`.
- Env in the launcher: `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, a process-group timeout of 30 min
  (a pass boundary writes a ~5 GB checkpoint on rank 0 while the others wait at a barrier).

### D1 · global plan, rank slice

In `train_stage`:

```python
B = train_cfg.batch_size                      # per-rank (memory-shaped); unchanged meaning on 1 GPU
G = B * dist.world                            # the global batch
specs, batches = pass_plan(dataset.specs, plan, batch_size=G, ...)   # identical on every rank
# mid-pass resume: bucket_batches(..., G, ...) likewise
mine = [b[dist.rank * B:(dist.rank + 1) * B] for b in batches]
```

- Each rank pads its slice to the slice's own longest clip, not the global batch's. The data
  equivalence therefore also needs the encoders to be padding-invariant per row: pinned for
  XLS-R (`tests/test_frontends_xlsr.py`) and BEATs
  (`tests/test_beats_windowed.py::test_a_long_row_does_not_depend_on_its_batch`).
- `bucket_batches` draws a global batch from **one duration bucket** (`training/collate.py:265`),
  so every rank's slice has similar lengths and the ranks finish their forward passes together.
- It drops the remainder (`drop_last=True`), so **every rank runs the same number of steps**, a
  hard requirement. Assert it: `len(b) == G` for every batch.
- `SamplerState.batch_index` indexes the global plan, so its meaning and the resume logic
  (`training/loop.py:395-417` load, `:451-470` per-pass plan and mid-pass batch seed) do not change. The `batch_seed` rule is untouched.
- `result.pass_digests` stays the digest of the global spec list, so it matches the 1-GPU
  digest at `batch_size = G` (test T2).
- Log `global_batch = G` and `world` in `train_log.jsonl` and the ledger row. **Run lengths are
  stated in global steps × G** (doc 13 R-len).

### D2 · the wrapper

```python
net = model
if dist.enabled:
    net = DDP(model, device_ids=[dist.local_rank], broadcast_buffers=False,
              find_unused_parameters=False, gradient_as_bucket_view=True)
    if resume_from is None:      # per-rank dropout; a resume restores rng_by_rank[rank] instead
        torch.manual_seed(train_cfg.seed + 1000 * dist.rank)
...
out = net(wav, lengths)          # forward/backward only
ema.update(model)                # unwrapped
_checkpoint(result, model, ...)  # unwrapped
```

- The reseed is **inside** `if dist.enabled` and skipped on resume. At world 1 an unconditional
  reseed would reset the generator after the model's init draws and break T1's bitwise
  requirement; on resume it would clobber `_set_rng_state` (`training/loop.py:405`).
- `find_unused_parameters=False`: A6/A7 keep the graph connected, and the frozen trunk does not
  require grad, so DDP does not reduce its gradients. (DDP still **broadcasts every parameter
  once at wrap**, frozen trunk included: ~2 GB, one time.) If a hang appears, the first suspect is a *new* module (the
  fusion) that some batch skips. Diagnose with `TORCH_DISTRIBUTED_DEBUG=DETAIL`, not by turning
  on `find_unused_parameters`.
- Wrap **after** `init_from` / partial init and after `model.to(device)`.
- Wrap **before** building the optimizer. The optimizer takes `model`'s parameters, which the
  wrapper shares.
- Gradient clipping (`training/loop.py:508-516`) runs on the averaged gradients after
  `backward`, so it is correct unchanged. With `bf16` the scaler is disabled, so `unscale_` is a
  no-op.

### D3 · globally normalised masked mean

Problem: rank r computes `L_r = Σ_i∈r m_i ℓ_i / n_r` (with `n_r` its present count); DDP steps on
`mean_r ∇L_r`. The single-process loss on the same global batch is `Σ m_i ℓ_i / Σ n_r`.
They differ whenever the present counts differ across ranks. A rank with `n_r = 0` still counts
in the average, so a rare component's head is **down-weighted by the fraction of empty ranks**.
The error is small for per-rank batches ≥ 8, but it is a silent objective change that grows as
the per-rank batch shrinks.

Fix (no config flag: pass the global counts into `multitask_loss` as an argument; at world 1
the formula is bitwise today's `x · 1 / denom`):

```python
denom_local = mask.sum()
denom = all_reduce_sum(denom_local.detach(), dist)            # global present count
loss = (per_sample * mask).sum() * dist.world / denom.clamp(min=1)
```

- Multiplying by `world` cancels DDP's `1/world` average. The resulting gradient **equals** the
  single-process gradient at batch `G` (test T3, up to reduction order).
- A globally zero count still returns `per_sample.sum() * 0.0` (A6).
- Apply it to the O1 term (`models/losses.py:262-264`) too; it uses the same mask.
- The logged `p_c` / `w_eff` parts should also use the global count. They are per-batch
  diagnostics that doc 02 §4 reads.
- Unmasked heads (`file`, presences) need nothing. Their plain mean over a fixed per-rank `B` is
  already exact under averaging, because every rank has the same `B` (D1's assert).
- One all-reduce of one scalar per masked head per step: negligible. Better: all-reduce the
  per-head present counts **once per global batch from `batch["targets"]`** before the forward
  (this is also what D5 needs).
- The ranking term is **not** DDP-exact (`pairwise_ranking_loss` loses cross-rank pairs). It is
  0 in every config today; **refuse `ranking_weight > 0` under DDP**.
- Checked by review: exact under grad clipping (clipping sees the averaged, i.e. single-process,
  gradient), for the O1 term, and under bf16 (BCE runs in fp32 under autocast).

### D4 · rank-0 side effects, RNG, eval hand-off

- **EMA:** built and updated on rank 0 only (weights are identical across ranks after every
  step). Memory: a float32 shadow of every float tensor, frozen trunk included. For 1B@24 + BEATs
  that is about 580 M params ≈ 2.3 GB, fine on an H200. ESTIMATE: ~1–2 ms per step for the
  foreach update.
- **Checkpoints** (`_checkpoint`, `training/loop.py:582`): rank 0 writes. Before that, gather
  every rank's `_rng_state()` with `all_gather_object` and store it as `rng_by_rank` (keep `rng`
  = rank 0's, for back-compat with 1-GPU loaders). Put a `barrier` after the write. On resume
  each rank restores `rng_by_rank[rank]` and refuses a checkpoint whose `world`, global batch
  `G` or `grad_accum` differs, because its slices and RNG would not line up. *(As built, 781433c: a
  different world is allowed when `DDP_ALLOW_WORLD_CHANGE=1` and the global batch is unchanged.)*
  - Size ESTIMATE: 1B@24 state ~2 GB, plus EMA ~2.3 GB, plus AdamW over the trainable set. So
    ≈ 5 GB per checkpoint.
  - At pass boundaries × ~30 passes (§4) that is ~150 GB; prune all but the last few + EMA. `/data/project/private` has 78 TB free
    (df, 18:25), so disk is fine.
  - Still, keep `checkpoint_every: 0` (pass boundaries) unless mid-pass resume is wanted.
- **`train_log.jsonl`**: rank 0 writes.
  - Before writing, all-reduce the logged loss parts (mean) and the per-rank `data_wait_s`
    (**max**). The max is what tells whether one rank's render pool is starving the step.
  - Add `world`, `global_batch` and `max_mem_gib`.
  - `result.loss_history` (pass means, `training/loop.py:543`) is rank-local and carries D3's
    `× world` scale: all-reduce it the same way.
- **After training** (in `scripts/train.py` **`main()`**, not inside `train_fold`: `main` unpacks
  its tuple at `scripts/train.py:427`, so an early `return 0` there crashes ranks > 0, and
  torchrun then kills rank 0 mid-evaluation):
  1. every rank `barrier`s;
  2. ranks > 0 `teardown` and exit 0 from `main()`;
  3. rank 0 tears the process group down too, then runs `select_weights`, `save_checkpoint`,
     `evaluate` and the ledger single-process exactly as today (`scripts/train.py:231-273`,
     `:309-315`).

  This avoids an NCCL timeout during a long rank-0 evaluation, and leaves the evaluation code
  untouched. **Only rank 0's exit code carries the quotable status** (HANDOFF_RUN3 #26).
  - Under DDP, **refuse more than one fold (or `--folds all`) per invocation**: after the
    teardown the next fold has no process group. One fold or `--all-data` per job.
- **stdout**: prefix lines with `[r{rank}]`, or silence ranks > 0 except for errors, so Slurm
  logs stay readable.

### D5 · gradient accumulation (optional, recommended)

`LoopConfig.grad_accum: int = 1`.

- Each global batch is split into `grad_accum` micro-batches per rank. Use `net.no_sync()` on
  all but the last micro-batch, and divide the loss by `grad_accum`.
- The global batch becomes `B × world × grad_accum`.
- Why: the owner's 2-GPU smoke (tonight) and the 8-GPU main run (Sunday) can then run the **same
  global batch and LR**, so the smoke's loss curve predicts the main run's.
- **Do not divide the masked heads by `grad_accum`.** D3's count is taken over the whole global
  batch (all ranks × all micro-batches), all-reduced once from `batch["targets"]` before the first
  micro-forward; each micro-batch's masked term is `Σ m_i ℓ_i · world / N_global`, which already
  sums to the right total across micro-batches. Only the **unmasked** means are divided by
  `grad_accum`. Dividing everything (the naive recipe) down-weights masked heads by `grad_accum`;
  normalising per micro-batch reintroduces the bias D3 removes.

### D6 · encoder hold (DDP-safe warm-up for the cold 1B LoRA)

Doc 13 §7 open question 4: train the new parts (heads + fusion) first, with the encoder LoRA
frozen for 1–2k steps.

- Toggling `requires_grad` mid-run breaks a DDP wrapper built once (A8).
- Instead, add `LoopConfig.frontend_hold_steps: int = 0`. For `step < hold`, after `backward`,
  **set every frontend-group `p.grad = None`** before clipping and `optimizer.step`. AdamW skips
  params without a gradient, so the frontend does not move (no update, no weight decay, no Adam
  moments), and the held gradients do not inflate the clip norm and shrink the heads' steps.
  (LR 0 alone would also freeze them bitwise, but the clip coupling remains.)
- After the hold the frontend group needs **its own warm-up**: use
  `lr_factor(step - hold, total - hold, cfg)` for that group. Today's `lr_factor`
  (`training/loop.py:301`) warms up from step 0, so with a hold longer than `warmup_steps` the
  frontend would jump from 0 to the full cosine factor.
- It is a pure function of the global step, so resume needs no new state.
- The cost is computing LoRA gradients that are not applied.
- **Which params are "frontend"** is decided by `_param_groups` (`training/loop.py`,
  `model.frontends.parameters()`). If the fusion weights live inside the XLS-R frontend module,
  they are held too. **Put the fusion module outside `model.frontends` (e.g. `model.fusion`) if
  it must train during the hold.**
- 🔴 **And then add it to `trainable_parameters`** (`training/stages.py:219-245`), which collects
  only `heads`, `frontends`, `distill_head` and `separation_head`. A `model.fusion` left out gets
  gradients but is never in AdamW: the layer weights stay at their uniform init, **silently**.
  Test T8 guards this.

### D7 · launcher `scripts/train_ddp.sbatch` (new; `train_run2.sbatch` untouched)

```bash
#SBATCH --job-name=eval-hyeonseop
#SBATCH --partition=debug
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:2             # #SBATCH lines do not expand variables: override with sbatch --gres=gpu:8
#SBATCH --cpus-per-task=...       # 12 per GPU (node: 96 CPUs / 8 H200)
#SBATCH --mem=...                 # ~200G per GPU (run 2 used 200G per 1-GPU task)
NGPU="${SLURM_GPUS_ON_NODE:?}"   # derived from the allocation, never set by hand
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID CUDA_CACHE_MAXSIZE=4294967296   # NOT on EFS (HANDOFF_TRAINING)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec "$DACON_VENV/bin/torchrun" --standalone --nproc_per_node="$NGPU" scripts/train.py \
  --manifest-dir "$MDIR" --processing ... --model configs/<1b>.yaml --train ... \
  --weights audio=.../beats,speech=/data/project/private/dacon-weights/xlsr-1b \
  --stages joint --out "$RUNS_ROOT/$RUN" --device cuda --draws ... "$@"
```

- `torchrun` sets `RANK`, `WORLD_SIZE` and `LOCAL_RANK`; `train.py` calls `init_from_env()`
  first.
- Per rank, `render_workers` stays 7 (the 1-GPU measurement). With 12 CPUs per GPU there is
  headroom; hyperthreads bought nothing (HANDOFF_TRAINING).
- Tonight the owner's job must coexist with the R1 array on the same node: `--gres=gpu:2` and no
  `--exclusive`. For the main run, `--gres=gpu:8`, launched once R2 has finished.
- `$DACON_VENV/bin/torchrun` exists (torch 2.7.1+cu128; NCCL and gloo available; checked
  18:40).

### D8 · tests (`tests/test_ddp.py`, CPU + gloo, stub model, `torch.multiprocessing.spawn`)

| id | asserts | why |
|---|---|---|
| T1 | world 1 through the new code = today's `train_stage`, **bitwise** (weights, losses, pass digests) | the 8-task arrays and every existing test stay valid |
| T2 | world 2 × B: every rank sees the same global plan and pass digest, and the union of its slices = the 1-process plan at batch 2B | the data claim in §0 |
| T3 | world 2 × B, one step: averaged gradients = the 1-process gradients at 2B (atol 1e-6, fp32), **with a batch in which one rank has zero present for a masked head** | D3; that case is where the naive version differs |
| T3b | `grad_accum 2` × world 2 vs the 1-process gradient at the full global batch, **with micro-batches whose present counts differ** | D5 (the double-normalisation trap) |
| T4 | world 2, **dropout on, EMA on, stop mid-pass** (`checkpoint_every`, so the `bucket_batches(G)` resume path runs): resume, and weights + EMA at step n equal an uninterrupted run's bitwise; a resume at a different world / `G` / `grad_accum` is refused | the resume guarantee of `training/loop.py:6-20`, which the existing `tests/test_loop.py` pins for 1 process |
| T5 | only rank 0 writes files; the saved `state_dict` has no `module.` keys and loads strictly into a fresh model | A4 |
| T6 | `--stages independent` under world 2 raises before training | A8 |
| T7 | `frontend_hold_steps k`: frontend params bitwise unchanged for k steps, then move; the heads' updates during the hold equal a run with the frontend frozen (no clip coupling); the frontend LR at step k is its warm-up start, not the full factor | D6 |
| T8 | every parameter with `requires_grad` is in exactly one `optimizer.param_groups` entry (run on the stub **and** on the owner's 1B config, CPU, no weights needed if the stub path allows; else on GPU in the probe) | D6's `model.fusion` trap |
| T9 | end-to-end: `torchrun --nproc_per_node 2` (gloo, CPU) on `scripts/train.py` with a stub config and `--max-steps` + `--allow-unquotable`: both exit 0, only rank 0 wrote files, `scored.pt` exists and loads strictly | D4's exit path; any collective called only under `if main:` hangs here, not in production |

- The gradient tests (T3, T3b) need the stub model with dropout **off**. Dropout is where ranks'
  RNGs differ by design; T4 needs it **on**, since that is what the per-rank RNG restore is for.
- Gates: `ruff check` and the full `pytest` suite stay green (`dacon311` venv).

## 3 · Sequence and gates (with the clock: LB closes Tue 10:00)

| when (KST) | step | gate to pass before the next |
|---|---|---|
| Sat evening | D0–D4 + T1–T6, T9 on CPU | all green; T1 bitwise |
| Sat ~19:30 (run 3a frees GPUs; owner has 2) | **smoke A**: 2 GPUs, **300M@12 `c_run2.yaml`**, init from `run2-T7`'s `scored.pt`, 300 steps, per-rank 8 | `train_log` loss tracks a 1-GPU batch-16 run from the same init (same global batch, same data): within noise over the first 300 steps; no hang; `max data_wait` < 10 % of the step |
| Sat night | **smoke B**: resume on GPU; kill at step 150, resume, compare with smoke A | bitwise equal to smoke A at step 300 (bf16 on GPU: allow exact-or-tiny; state which) |
| Sat night | **probe 1B**: owner's 1B@24 + fusion config, 2 GPUs, per-rank B ∈ {8, 12, 16}, grad checkpointing off/on, 50 steps each | pick the largest B with peak ≤ ~125 GiB of 141; record s/step and data wait |
| Sat night → Sun AM | D5 (accum), D6 (hold), T3b/T7/T8; **1B fold-1 short check** on the 2 GPUs if time allows (doc 13 §7: proxy → 1B transfer is otherwise unmeasured) | fold-1 VAL on run 1's v3 specs via `post_table.sbatch`-style scoring vs run 2 task T1 (fold 1) |
| Sun ~12:00 | **main**: 8 GPUs, 1B@24 + fusion + strategy-v5 (+ O1 if R1 kept it), all-data, `--select ema` | first 30 min: throughput within 15 % of `8 × (1-rank step rate)` ESTIMATE; loss falling; no rank starving |
| Mon early | rank 0 `scored.pt` → package (`--file-mode max3`) → server-mirror timing | real L4 timing ≤ 30 min (the 6.0 min figure is an ESTIMATE, doc 12 §7) |

**Go/no-go (doc 13 §7 open question 8, restated):** if the smoke gates are not green by Sun
~11:00, the main run does not start on DDP. The R2 **300M@24 + fusion** all-data model is the
Monday submission, and 1B can still run single-GPU as an extra if a GPU is free.

## 4 · Numbers to set (ESTIMATES, to be replaced by the smoke/probe measurements)

- **Memory.** 1B@24 trained at 24.6 GiB for batch 4 in the bench, against 14.5 for 300M@12
  (doc 12 §7). 300M@12 at batch 16 in a real run peaked at ~99–103 GiB (`configs/processing_run2.yaml`
  comment). Scaling crudely, **1B@24 at per-rank 16 without checkpointing ≈ 140 GiB: does not
  fit. Per-rank 8 ≈ 70 GiB fits**; per-rank 16 needs layer checkpointing (+10–15 % step,
  measured for 300M in run 2). The probe decides.
- **Step time.** Bench encoder step ratio 1B@24 / 300M@12 = 4.03× (doc 12 §7). Run 2's
  300M@12 step: 0.86–1.0 s at batch 16 on 1 GPU. The encoder is not the whole step, so
  **ESTIMATE ~1.5–2 s per step at per-rank 8**. Gradient all-reduce is over the trainable set only
  (LoRA + heads; tens of M params), negligible on NVLink.
- **Global batch and LR.**
  - Per-rank 8 × 8 GPUs = **64** (× `grad_accum 2` = 128, the doc 13 R-len figure).
  - Run 2 used lr 1e-4 at batch 16 for a *warm* fine-tune. A cold 1B LoRA + cold voice/file
    heads want more: **ESTIMATE heads 2e-4 (√-scaled from 1e-4@16 to 64), `frontend_lr_scale`
    0.5** (run 2's 0.2 was for a warm LoRA).
  - Warmup: 500 global steps after a `frontend_hold_steps` of 1000.
  - Owner's call; the smoke A/B loss curves are the evidence.
- **Length.** ESTIMATE 16 h window × 3600 / ~1.8 s ≈ 32k global steps × 64 ≈ 2 M samples. That
  is 6× run 2's 320k (10 passes × 32k draws). With `--draws 64000` a pass is 1000 global steps
  at G = 64, i.e. **~30 min per pass and ~32 passes** in the window. For hourly checkpoints use
  `--draws 128000` (~16 passes); either way prune old pass checkpoints (§D4).

## 5 · Interfaces with the 1B model work (owner)

DDP does not care what the model is, but these three points touch it:

1. **Partial init** (doc 13 M3; HANDOFF #24). `init_from` is strict except `INIT_MAY_MISS`
   (`scripts/train.py:112-133`). The 1B model needs these skipped:
   - the width-mismatched voice/file heads' first layers;
   - the new fusion weights;
   - all XLS-R LoRA.

   `load_state_dict(strict=False)` still **raises on shape mismatches**, and every 300M trunk key
   mismatches the 1B trunk, so filter the `run2-T7` blob **by key and shape** before loading and allow
   the whole speech trunk (plus its LoRA) to be missing; report the dropped keys.
   Do it on every rank before the wrap (identical result on every rank; DDP's broadcast is a
   safety net, not the mechanism). Print the loaded/skipped key lists **on rank 0**.
2. **Fusion over hidden layers.** `XLSRFrontend._encode` (`models/frontends.py:919-953`) keeps
   only the last layer. Fusion needs every layer's output, which is 24 × (B, T, 1280) of
   activations kept for backward. That is part of what the memory probe measures; it is also why
   layer checkpointing may become necessary. Every fusion weight must be used in every forward
   (a softmax over all layers is), or DDP needs `find_unused_parameters`.
3. **Packaging.** `save_checkpoint` must receive the unwrapped model (A4). `package_submission.py`
   and `script.py` then need the 1B weights directory; the zip grows by ~3.9 GB
   (`pytorch_model.bin` is 3,862,391,172 bytes), **plus** a larger `scored.pt`: a full 1B@24
   fp32 state_dict is ~2.3 GB ESTIMATE. Truncation to 24 layers happens at load, so the
   shipped weights could be pre-truncated to save space. That is a packaging decision; the limit
   is 10 GB (doc 13 E6).

## 6 · Risks

| risk | sign | response |
|---|---|---|
| a rank's render pool starves the step | `max data_wait_s` ≫ rank 0's | more workers on that job (12 CPUs/GPU available); check the long-clip bucket |
| hang at the first backward | no log line after `init` | `TORCH_DISTRIBUTED_DEBUG=DETAIL`; usually a parameter unused in some forward (§5.2) |
| NCCL timeout at a checkpoint barrier | rank-0 write > timeout | 30-min timeout; checkpoint at pass boundaries only |
| silent objective change vs 1 GPU | loss curve of smoke A off the 1-GPU curve | D3 + T3 exist for this |
| per-rank dropout identical | — | reseed `seed + 1000·rank` after the wrap (A10) |
| EFS CUDA-cache stall with 8 processes | slow first steps | `CUDA_CACHE_PATH=/tmp/...` (launcher) |
| main run cannot start at noon (R2 holds 6 GPUs until ~12:00) | squeue | the main run is one 8-GPU job; queue it to start after R2 (`--dependency=afterany:<R2 job>`) |

## 7 · Standing rules that apply

- Every Slurm job `-J eval-hyeonseop`, partition `debug`, per-job `CUDA_CACHE_PATH` on `/tmp`.
- Never kill jobs this session did not start; never `pkill -f`.
- Commit on `feat/ddp-1b` with explicit paths (never `git add -A`); no push or PR without the
  owner's go. `ruff check` + `pytest` green before each commit.
- English + Korean data only; the draw config and audits are unchanged by DDP (D1 slices the
  same plan).

## 8 · Implementation status (2026-09-26 ~23:00 KST)

D0–D8 are implemented on `feat/ddp-1b`, in the worktree `/home/hyeonseop.shin/workspace/dacon-ddp-1b`.
D9 is done for the 300M@12 model. The 1B probe is **not** done: `xlsr_1b` is not wired in
`models/frontends.py:build_frontend` yet (the owner's model work).

| piece | where | as planned? |
|---|---|---|
| D0 | `training/distributed.py` (`Dist`, `init_from_env`, `current`, collectives) | yes |
| D1 | `training/loop.py::rank_micro_batches`; the plan built at the global batch | yes; a short global batch is refused |
| D2 | DDP wrap inside `train_stage`; `model` stays unwrapped | yes; reseed only under DDP and not on resume |
| D3 | `models/losses.py::GlobalNorm`, passed as `multitask_loss(..., norm=)` | yes (no config flag); counts all-reduced once per step from `batch["targets"]` |
| D4 | `_checkpoint` (gathers `rng_by_rank`, rank 0 writes, barrier), `_combine_rows`, `scripts/train.py::_leave_group` | yes; ranks > 0 exit from `main()` |
| D5 | `LoopConfig.grad_accum`, `forward_backward` (`no_sync` on all but the last micro-batch) | yes; masked heads are not divided by accum |
| D6 | `LoopConfig.frontend_hold_steps`: frontend `grad = None` during the hold, then its own warm-up | yes; `trainable_parameters` now collects **any** other top-level module (the fusion trap), de-duplicated |
| D7 | `scripts/train_ddp.sbatch` | yes. **The node exposes 64 usable CPUs** (`CPUEfctv`), not 96: use 8 per GPU (`--cpus-per-task=16` for 2 GPUs, 64 for 8) |
| D8 | `tests/test_ddp.py` (18 tests, CPU/gloo) | T1–T9, plus the naive-mean mutation and both refusals (several folds, several stages) |

**Refused under DDP:**
- stages with several branch groups (`independent`);
- more than one `--stages` entry or fold per invocation;
- `ranking_weight > 0`;
- distillation;
- a resume at a different world, global batch or `grad_accum`.

### Verification

- **CPU (`tests/test_ddp.py`):**
  - the DDP gradient equals the single-process gradient on the global batch, including a rank
    and a micro-batch with no voice present (atol 1e-6);
  - the naive per-slice mean differs by more than 1e-4 (the mutation);
  - DDP resume in the middle of a pass is **bitwise**, with dropout and EMA on;
  - a world-1 process group is bitwise the plain loop;
  - `torchrun` over `scripts/train.py` exits 0 on both ranks, and rank 0 writes `scored.pt` and
    the truncation checkpoint.
- **Independent review:** the single-process path compared against `HEAD` gives the same
  weights, EMA, resume and loss history bitwise.
- **Existing suites:** `test_loop`, `test_losses`, `test_checkpoint`, `test_loop_pool`,
  `test_stages` and `test_train_integration` pass.
- **GPU smoke A** (job 221087; 2×H200; 300M@12 `c_run2.yaml`; fold 1; init run-2 T1; per-rank 8 →
  global 16; 300 steps):
  - voice `p_c` equals the single-GPU run 3-R1 T0 (same draw, batch 16) to 4 decimals at every
    logged step, so the data is the same;
  - the total loss agrees within 0.001 at every step;
  - 0.52 s/step (T0 on 1 GPU, on the shared node: ~1.9–2.3 s/step);
  - 51.8 GiB per rank;
  - exit 0, with rank 0 evaluating after the teardown.
- **GPU smoke B** (job 221097): resumed A at step 150, reached step 300 with the identical sampler
  state, and the step-300 loss agrees to 6e-7. The weights are **not** bitwise: max |Δ| 3.2e-3.
  - **Control** (job 221099): two *fresh* identical runs already differ at step 150 (max |Δ| 3.1e-4,
    the same 319 trainable tensors). So GPU training is nondeterministic, and the gap is not a
    resume defect; CPU resume is bitwise (T4).

### What remains (owner)

> **Done (2026-09-29):** the 1B wiring landed (fef750e) and the main run trained on 7 then 8 GPUs
> (jobs 221505 → 221709); outcomes in `docs/training/17-final-runs.md` on `feat/training`.


- wire `xlsr_1b`, the fusion and the partial init (§5);
- then the 1B memory probe on 2 GPUs (per-rank 8/12/16, grad checkpointing off/on);
- then the 8-GPU main run, with `sbatch --gres=gpu:8 --cpus-per-task=64 --mem=1600G
  scripts/train_ddp.sbatch <run> all:0 --batch-size <B> ...`.
