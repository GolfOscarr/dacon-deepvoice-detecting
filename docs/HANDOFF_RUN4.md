# Handoff — run 4: the 1B main run is training; package, submit and pick by Tue 10:00

> **Superseded (2026-09-29):** this is the Mon 01:30 snapshot. The 1B run finished (pass 15 was its
> best); runs A–D, strategy-v6/v6b/v6c, the soups, ensembles and final packages are in
> [training/17-final-runs.md](training/17-final-runs.md). Corrections to this snapshot: a pass is
> **1,140** steps (27,360 in all), not 1,142; diagB *was* submitted; runs A and D used the `normal`
> partition (node 68) with the owner's approval.

*Written 2026-09-28 ~01:30 KST by the lead session (its context is full). Every state claim was
measured at writing time; the command that shows it is next to it. **UNVERIFIED** marks what was
not. Read with `docs/training/16-lessons-learned.md` (what we learned, all LB results) and
`docs/HANDOFF_RUN3.md` §8 (pitfalls 17–26).*

**Clock:** the leaderboard closes **Tue 2026-09-29 10:00 KST**. 3 submissions per day (reset at
midnight KST). Private = Public. **Monday's 3 slots are unused at writing time.**

## 1 · Environment

| what | value |
|---|---|
| main repo | `/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting`, branch **`feat/training`**, HEAD `5ba7736`, clean |
| 1B / DDP worktree | `/home/hyeonseop.shin/workspace/dacon-ddp-1b`, branch **`feat/ddp-1b`**, HEAD `781433c`, clean. **All 1B training and 1B packaging run from here** (it has feat/training merged + DDP + xlsr_1b) |
| python | `/data/project/private/dacon-venvs/dacon311/bin/python` (no bare `python`); server mirror `/data/project/private/dacon-venvs/server-mirror/bin/python` |
| data | `/data/project/private/dacon-corpus/manifests/strategy-v5` (current, 590,316 rows, 3,324 h); v3/v4 never rewrite |
| runs | `/data/project/private/dacon-runs/` — 1B main: `main-1b-v5b/all_data_seed0/` |
| submissions | local `/data/project/private/dacon-submissions/<name>{,.zip}`; S3 `s3://hyeonseop-s3/dacon-deepfake-detection/submissions/` |
| ops scripts | `/data/project/private/dacon-runs/_ops/` (`health.sh`, `probe_watch.sh`, `pkg_1b.sh`, `pkg_v5.sh`) — persisted copies; the old session scratch dies with the session |

Nothing is pushed; no push or PR without the owner's go.

```bash
cd /home/hyeonseop.shin/workspace/dacon-ddp-1b && git status --short --branch && git log --oneline -1
squeue -u $USER -o "%.10i %.3t %.10M %b"
tail -1 /data/project/private/dacon-runs/main-1b-v5b/all_data_seed0/train_log.jsonl | cut -c1-120
cat /data/project/private/dacon-runs/_probe_1b/probe_table.tsv
```

## 2 · Goal of the next session

**The highest final LB by Tue 10:00.** Concretely:
1. Keep the 1B main run healthy to its end (~12:20 Mon) and keep the 15-min health check going.
2. Score its checkpoints on PROBE and submit the best 1B package(s) on Monday (3 slots).
3. Tuesday morning: final pick / safety resubmission.

## 3 · State

### 3.1 Leaderboard

| submission | LB | ADS | CPS |
|---|---|---|---|
| run 2 T7 | 0.81144 | 0.79302 | 0.97730 |
| T7 fine-tuned on v5, learned file head (`run5-T7v5-learned`) | 0.82487 | 0.80575 | 0.99698 |
| **T7 on v5, max3** (`run5-T7v5-max3`) | **0.83046** (best) | 0.81196 | 0.99698 |
| leader | 0.89871 | 0.88756 | 0.99907 |

- **max3 is +0.0056 on the LB**, which is 6× its VAL gain. **Package everything with
  `--file-mode max3`.**
- From diagA, T7's test file EER is 0.20; the voice and music EERs come to ≈ 0.21 each if equal.
  The gap is generalization on every head.

### 3.2 The 1B main run (RUNNING)

- **Slurm job 221709**, 8×H200, `-J eval-hyeonseop`, started 15:41 Sun.
  - Resumed from `joint-pass4.pt` of the cancelled 7-GPU job 221505.
- **Model and data:**
  - `configs/c_1b_fusion.yaml`: XLS-R-1B@24 + per-branch layer fusion + BEATs.
  - `configs/processing_1b_main.yaml`: the strategy-v5 draw, grad checkpointing, LoRA hold 1000,
    frontend LR ×0.5.
  - `configs/train_1b_main.yaml`: LR 3e-4, cosine.
- **Batching and length:** all-data, per-rank 14 × 8 = **global batch 112**, `--draws 128000`,
  `--epochs 24` → **27,408 steps** (1,142 per pass).
- **Checkpoints:** every pass, at `main-1b-v5b/all_data_seed0/joint-pass<N>.pt` (4.9 GB each,
  with EMA). Pass `N` ends at step `(N+1)*1142`. At the end: `scored.pt` (EMA) + `processing.json`.
- **At writing:** step ~15,900, loss ~0.02–0.03, LR factor 0.43.
- **Pace:** ~3.4 s/step including pass switches and PROBE scoring.
- **ETAs:**

  | pass | ETA (KST) |
  |---|---|
  | 15 | ~03:40 |
  | 18 | ~07:00 |
  | 21 | ~10:10 |
  | 23, final | ~12:20; the job then exits |

- **Launch command (for a crash restart).** Resume from the latest `joint-pass<N>.pt`, from the
  worktree:
  ```bash
  cd /home/hyeonseop.shin/workspace/dacon-ddp-1b
  MANIFEST_DIR=/data/project/private/dacon-corpus/manifests/strategy-v5 PROC=configs/processing_1b_main.yaml \
  TRAIN=configs/train_1b_main.yaml MODEL_CFG=configs/c_1b_fusion.yaml \
  WEIGHTS=audio=/data/project/private/dacon-weights/beats,speech=/data/project/private/dacon-weights/xlsr-1b \
  INIT=- DRAWS=128000 sbatch --gres=gpu:8 --cpus-per-task=64 --mem=1600G scripts/train_ddp.sbatch \
    main-1b-v5b all:0 --batch-size 14 --epochs 24 --resume <joint-passN.pt>
  ```
  - The same world (8) resumes bitwise-state.
  - A different GPU count needs `DDP_ALLOW_WORLD_CHANGE=1` and the **same global batch 112**.

### 3.3 PROBE per checkpoint (EMA weights; `_probe_1b/probe_table.tsv`)

| pass | step | PROBE | file | voice | music |
|---|---|---|---|---|---|
| 2 | 3,420 | 0.9251 | 0.0987 | 0.0512 | 0.0788 |
| 6 | 7,980 | 0.9293 | 0.0980 | 0.0371 | 0.0739 |
| 9 | 11,400 | 0.9257 | 0.0947 | 0.0468 | 0.0862 |
| **12** | 14,820 | **0.9411** | 0.0780 | 0.0380 | 0.0627 |

For reference: 300M-v5 (LB 0.83046) scores 0.9205 and T7 0.9103.
- The v5 gain was larger on the LB than on PROBE (+0.0134 vs +0.010).
- PROBE ranks checkpoints; differences below ~0.003 are noise.

**Watcher:** `_ops/probe_watch.sh`, PID in `_ops/probe_watch.pid`, log `_probe_1b/watch.log`.
- **What it scores:** passes **15, 18, 21** as they land, via `srun --jobid=221709 --overlap` on
  GPU 0 (~30 min each; it slows training ~5 min per score).
- **Where results go:** each score is appended to the table.
- **Pass 23 is NOT scored by it,** because the job exits after the last pass. The next agent
  must score pass 23 (or `scored.pt`) on a free GPU after the job ends (§4 step 4).
- **If the watcher dies:** restart it with
  `nohup /data/project/private/dacon-runs/_ops/probe_watch.sh > .../_probe_1b/watch.log 2>&1 &`.
  It skips passes already in the table. **Stop it by PID only.**

### 3.4 Packaged and ready

- **`1b-v5-p12-max3.zip` (4.08 GB)** is in S3. It is 1B pass-12 EMA with max3.
  - **Server-mirror CPU check:** 8 rows, 0 fallback, 168 s for 8 clips on CPU.
  - **L4 GPU timing UNVERIFIED** (estimate ~6 min per 1,200 clips; limit 30 min). This
    submission is also the timing test.
  - **→ Monday slot 1** (the owner submits).
- **Packaging any 1B pass:**
  `bash /data/project/private/dacon-runs/_ops/pkg_1b.sh <pass>`. It:
  1. builds the EMA scored.pt plus a `processing.json` from `processing_1b_main.yaml` in
     `dacon-runs/_pkg_1b/pass<N>/`;
  2. runs `package_submission.py --file-mode max3` from the worktree;
  3. runs the server-mirror CPU check;
  4. uploads to S3.

  The output is named `1b-v5-p<N>-max3`, and it refuses to overwrite.
- **For the final model:** package `main-1b-v5b/all_data_seed0/scored.pt` directly with the same
  flags. Its `processing.json` is written beside it by train.py.

### 3.5 Health check (session-only; RE-CREATE IT)

The 15-min cron dies with the session. Re-create it:
- **Schedule:** CronCreate `4,19,34,49 * * * *`.
- **Prompt:** "15-min GPU health check (owner request): run
  /data/project/private/dacon-runs/_ops/health.sh and judge it. Flag anything unhealthy: a
  FAILED/TIMEOUT/OOM job, a job not named eval-hyeonseop, an idle GPU that should be busy, a
  stalled training step since the last check, errors in logs. If unhealthy, act and send a
  PushNotification. Report to the owner in 2-5 lines only on progress or a problem."

`health.sh` prints jobs, non-ok ends, GPU use and the latest train_log lines.
- **Filter out old noise:** its "ended non-ok" list still shows old run-2/3 array tasks that
  "failed" by design, pitfall #26 (the not-quotable exit).

## 4 · What to do next (in order)

| # | when (KST) | step |
|---|---|---|
| 1 | now | re-create the health cron (§3.5); check the watcher PID is alive; confirm training advances |
| 2 | Mon morning | owner submits **`1b-v5-p12-max3.zip`** (slot 1). Record its LB (and whether it ran within time). If it times out, the fix is inference-side (fp16, batch); fall back to `run5-T7v5-max3` quality until fixed |
| 3 | ~04:20 / ~07:40 / ~10:50 | read PROBE for passes 15 / 18 / 21 as the watcher appends them; if a pass beats 12 by > ~0.003, package it with `pkg_1b.sh <pass>` right away |
| 4 | ~12:20 | job ends → score pass 23 / `scored.pt` on PROBE on a free GPU (`scripts/eval_probe.py` from the worktree, weights `audio=…/beats,speech=…/xlsr-1b`, manifest strategy-v3); consider a **checkpoint soup** of the top 2–3 passes (`package_submission.py` takes repeated `--scored` = uniform soup; PROBE it first via `eval_probe.py --scored a --scored b --soup`) |
| 5 | Mon afternoon | **slot 2:** the best 1B by PROBE (final, best pass, or soup), max3. **slot 3:** the best remaining idea — `--file-mode max` on the best 1B (max3 helped 6×, the learned file head may be the weak part), or a 1B + 300M-v5 ensemble (needs ensemble packaging support: check `package_submission.py`; UNVERIFIED that it can mix architectures) |
| 6 | Tue 00:00–10:00 | final pick: resubmit/choose the best; the private score = public, so the best LB submission is the pick. Keep one slot as a safety |

GPUs free after ~12:20 Monday can also train a 300M-v5 second seed for an ensemble (~3 h,
`scripts/train_run2.sbatch` with `TASK_TABLE`, as in doc 16 §7), only if the ensemble path is
verified.

## 5 · Decisions and standing rules

- **Primary goal: the score.**
- **Settled:** data first (strategy-v5 done); max3 everywhere; the one-class loss is dropped;
  WaveFake stays ×0.25; no diagB.
- **Decided (owner, 2026-09-28):** OuteTTS + Higgs (Llama 3.2 licence) STAY, and we COMPLY:
  every model trained on v5+ data is named with a leading "Llama" (e.g. "Llama-DeepVoice-Detector")
  in the 2nd-stage reports, with "Built with Llama" attribution.
- **Decided (owner, 2026-09-28):** ND-licensed music (fma, mtg-jamendo `derivatives_barred`) is
  used under talkboard A5: the 2nd-stage package ships ORIGINAL files + deterministic code
  (cut/convert scripts), never the derived pieces.
- **Slurm:**
  - Every job `-J eval-hyeonseop`, partition `debug`, per-job `CUDA_CACHE_PATH=/tmp/...`.
  - The node exposes **64 usable CPUs** (8 per GPU).
  - Never cancel jobs you did not start.
  - **Never `pkill -f`**: it killed its own shell this session. Stop background processes by PID.
- English + Korean data only. Nobody moves files under `dacon-corpus/` outside their own new dir.
- **Commits:**
  - Commit to the branch whenever there is a solid advance.
  - Stage explicit paths; never `git add -A`.
  - Keep the touched tests green (`$PY -m pytest -q -p no:cacheprovider <files>`; lint with
    `$PY -m pyflakes`; there is no ruff).
  - The commit message ends with the two attribution lines used in the log.

## 6 · Pitfalls found this phase (continue HANDOFF_RUN3 numbering)

| # | what | rule |
|---|---|---|
| 27 | The epoch draw is 6 ms/spec: 13 min per 128k-spec pass, on every rank | fixed: the draw is sharded across ranks, and the same-epoch redraw is skipped (`3a24e6b`) |
| 28 | 7 → 8 GPUs gained only ~5 % (the step waits on the slowest rank) | length-balanced rank slicing is a possible ~1 h speed-up, not done |
| 29 | `package_submission.py` needs `processing.json` beside the scored.pt; mid-run checkpoints have none | `pkg_1b.sh` writes it from `processing_1b_main.yaml` |
| 30 | PROBE via `srun --overlap` inside the training job shares GPU 0 and CPUs: ~30 min per score, training −25 % meanwhile | score every 3rd pass, not every pass |
| 31 | The DDP resume refuses a GPU-count change | `DDP_ALLOW_WORLD_CHANGE=1` + the same global batch (per-rank RNG reseeded; not bitwise) |
| 32 | v5 audit I1c voice_fake 0.597 (gate 0.60): repetition asymmetry, fake repeats .44 vs real .36 | passes; tune it down in any future rebuild |

You can use oh-my-claudecode skills if needed.
