# Handoff — training: run 1 done, round-2 data in flight, run 2 next

*Written 2026-09-26 00:10 KST by the session that built the training pipeline, ran run 1 and
started the round-2 data. Every number here was measured in that session, and the command or
file that produced it is named. Anything not measured is marked **UNVERIFIED**.*

**The competition clock.** The leaderboard closes **2026-09-29 10:00 KST**. It allows 3
submissions per day, and Private = Public. A full training run takes ~20 h on the 8 GPUs.

## 1 · Environment

| what | where |
|---|---|
| repo | `/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting`, **branch `feat/training`** (from `main` @ `3d8e1b3`, PR #8). One worktree. **Not pushed** |
| python | `/data/project/private/dacon-venvs/dacon311/bin/python`. There is no bare `python`; the venv has no pip, so use `/home/hyeonseop.shin/.local/bin/uv pip install --python <venv>/bin/python …` |
| test-server mirror | `/data/project/private/dacon-venvs/server-mirror`: torch 2.7.1+cu128, numpy 1.26.4, pandas 2.0.3, transformers 4.57.6 — the DACON L4 server's preinstalls. Run `script.py` under it before any submission |
| weights | BEATs `/data/project/private/dacon-weights/beats/BEATs_iter3_plus_AS2M.pt`; XLS-R-300M `/data/project/private/dacon-weights/xlsr-300m/` (revision in its README.txt). Pass them as `--weights audio=…/beats,speech=…/xlsr-300m` |
| corpus root | `/data/project/private/dacon-corpus` (`interim/<source>/…`, `raw/…`) |
| decode cache | `/data/project/private/dacon-corpus/cache16k` (16 kHz int16). Covers strategy-v3; `scripts/build_cache.py` extends it idempotently |
| manifests | `/data/project/private/dacon-corpus/manifests/strategy-v2/` (the EDA/processing PR), `strategy-v3/` (run 1). `_smoke-v3/` and `strategy-v1/` are not for use (a smoke copy; the pre-review corpus). **Never re-run a builder into an existing dir.** v4 goes to a new `strategy-v4/` |
| run outputs | `/data/project/private/dacon-runs/<run>/`, symlinked as `runs/<run>`. Run 1 is `first-v3/` (`fold0..3/`, `all_data_seed0..3/`, `probe/`) |
| submissions | `/data/project/private/dacon-submissions/` (`first-v3-seed0.zip` = the recommended first submission) |
| credentials | AWS is the EC2 instance role (none to handle). HF token at `~/.cache/huggingface/token` (account `hyeonseop-upstage`, gated-repo read enabled; Emilia is accepted) |

**Slurm.** Everything GPU goes through Slurm, partition `debug`, node `Slurm-GPU-Node-42`: this
host, 8× H200, 64 allocatable CPUs.

- **Owner rule: every job is named `eval-hyeonseop`** (`-J eval-hyeonseop`). The repo's
  `.sbatch` files already set it.
- **Every GPU job sets `CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID`** (§6 mistake 5).
- Find training PIDs with `nvidia-smi --query-compute-apps=pid`, not `pgrep -f` (§6).
- `kernel.yama.ptrace_scope=1`: py-spy cannot attach to a non-child. Use
  `ps -L -o tid,stat,wchan:28 -p PID` and `/proc/PID/fd`.

**Checking you are in the right place:**

```bash
cd /home/hyeonseop.shin/workspace/dacon-deepvoice-detecting && git status --short --branch && git log --oneline -1
squeue -u $USER -o "%.10i %.3t %.9M %.60o"
ls /data/project/private/dacon-runs/first-v3/ /data/project/private/dacon-corpus/manifests/
```

**Gates.** The suites touched this phase were run green after each change; the counts are in the
commit messages. **UNVERIFIED:** the full suite in its two halves on HEAD (last done on the EDA
branch). Run it before merging `feat/training`:

```bash
V=/data/project/private/dacon-venvs/dacon311/bin/python
H1=$(ls tests/test_*.py | grep -v -E "test_render.py|test_codec_roundtrip.py|test_loop.py" | awk 'NR<=24' | tr '\n' ' ')
H2=$(ls tests/test_*.py | grep -v -E "test_render.py|test_codec_roundtrip.py|test_loop.py" | awk 'NR>24' | tr '\n' ' ')
CUDA_VISIBLE_DEVICES="" $V -m pytest -o addopts="" -q $H1 2>&1 | grep -E "^FAILED|passed|failed"; echo "EXIT=${PIPESTATUS[0]}"
CUDA_VISIBLE_DEVICES="" $V -m pytest -o addopts="" -q $H2 tests/test_render.py 2>&1 | grep -E "^FAILED|passed|failed"; echo "EXIT=${PIPESTATUS[0]}"
CUDA_VISIBLE_DEVICES="" $V -m pytest -o addopts="" -q tests/test_loop.py 2>&1 | tail -1
$V scripts/check_links.py
```

Always read `${PIPESTATUS[0]}`. Lint is pyflakes plus ≤ 100 columns in `processing/`; ruff is
not installed.

## 2 · Goal of the next session

The owner's next step, in their words: *"investigate run 1's result and investigate how we
get much better."* Then finish round 2 and train run 2 in time for **several submissions
before 2026-09-29 10:00**.

Checkable outcomes:

1. **A diagnosis of run 1**, with per-family and per-cell numbers and the matched-pair gap (§3.2).
2. **strategy-v4 built and audited** with `bash scripts/build_strategy_v4.sh` (§5).
3. **Run 2 trained**, beating run 1 on the two numbers that matter:
   - the held-out Korean-clone EER (run 1: **0.177**);
   - fold 0's matched-pair voice gap (run 1: **0.153**).
4. **A run-2 submission** packaged and verified like run 1's (§5, step 6).

First step: report your understanding back to the owner, then §3.3's live jobs (the dispatcher
dies with the old session).

## 3 · Progress and diagnosis

### 3.1 What exists (all committed, `git log main..HEAD`, 23 commits)

| area | state | doc |
|---|---|---|
| plan + as-built record for run 1 | done | `docs/training/07-first-run-plan.md` (§6 as built, §7 results) |
| round-2 data plan, owner decisions, timeline | done | `docs/training/09-data-synthesis-plan.md` |
| round-1 Korean synthesis report | done | `docs/training/08-ko-synth-report.md` |
| model `configs/c_first_run.yaml` | BEATs 12 layers, LoRA r16 on 6 projections, 62-column (9.92 s) windows; XLS-R-300M 12 of 24 layers; XLS-R 50 fps pooled onto 6.25 fps; 264M params, 9.7M trainable | 07 §2, §6 |
| training loop | pooled rendering (bitwise = inline), cosine + warmup, frontend LR group, per-step log with `data_wait_s`/`compute_s`, layer checkpointing, capped cuFFT plan cache, `--all-data`, `--resume`, `--init-weights`, `--seed` | 07 §6 |
| draw | `lang_shares` with the both-sides rule (a language one side lacks is dropped) | 07 §6 |
| tools | `scripts/eval_checkpoint.py` (mid-run VAL), `scripts/eval_probe.py` (models + soup on PROBE), `scripts/package_submission.py`, `scripts/strategy/audit_fold.py`, `scripts/extend_manifest.py`, `scripts/build_strategy_v4.sh`, `scripts/train_first_run.sbatch` (8-task array; env `PROCESSING`, `MODEL_CFG`, `TRAIN_CFG`, `MANIFEST_DIR`) | |

### 3.2 Run 1 — measured (07 §7)

**What it is.** strategy-v3 (400,944 rows): 18 passes × 48,000 draws, batch 16, 54,000 steps.
It ran 8 tasks (4 fold + 4 all-data), ~21 h, EMA 0.999.

| fold | score | EER file | voice | music | held-out fake voice |
|---|---|---|---|---|---|
| 0 | 0.928 | 0.101 | 0.057 | 0.061 | LJ-voice block (WaveFake vocoders) |
| 1 | 0.916 | 0.101 | **0.179** | 0.024 | **Korean XTTS clones** (+ JSUT, dropped from draws) |
| 2 | 0.990 | 0.011 | 0.023 | 0.002 | MLAAD families |
| 3 | 0.966 | 0.044 | 0.012 | 0.043 | other families |

The mean is **0.950**. Per-fold detail is in `runs/first-v3/fold<k>/ledger_row.json` and
`val_predictions.parquet`. **Every fold task shows Slurm FAILED; that is the "not quotable"
exit (train.py exits 1), not a crash.** Read the ledger.

**Diagnoses — where to start "how do we get much better":**

- **D1 — same-speaker discrimination is weak.** On fold 0, the voice EER on matched pairs
  (the same LJ utterance, real vs vocoded) is **0.209**, against 0.057 pooled. The gap is
  **0.153**, over VG4's 0.10 gate.
  - Measured with `training.validate.t3_gap(pd.read_parquet(".../fold0/val_predictions.parquet"), head="voice")`.
  - The pooled voice number is carried partly by speaker/corpus identity.
  - Only fold 0 has T3 pairs; the other folds report VG4 "na".
- **D2 — unseen Korean zero-shot cloning is the weakest family anywhere.** Fold 1's `ko-synth/xtts`
  reads EER **0.177**. That is the test's threat (Korean voice phishing).
- **D3 — the music-side cells are the weak cells on fold 0.** Cell 6 (real voice + fake music)
  reads 0.180 and cell 5 0.150. The unseen family `fakemusiccaps/stable_audio_open` reads 0.143.
- **D4 — presence is trivial in our composed mixtures.** Presence AUC is ≈ 1.000 on every fold.
  Our composed audio is easier than real test audio, so **expect the leaderboard below 0.95**.
  The first LB score (not yet reported by the owner) is the only real calibration.
- **D5 — PROBE** (`runs/first-v3/probe/probe_eval.json`, one frozen draw of 3,000):

  | model | score |
  |---|---|
  | all_data_seed0 | **0.853** |
  | seed2 | 0.846 |
  | seed3 | 0.830 |
  | seed1 | 0.827 |
  | uniform soup of 4 | **0.826** |

  - **The soup does not help:** after 54,000 steps on different draw orders the runs are not in
    one basin.
  - PROBE's voice side is mostly Chinese (6.5 h real, 59.4 h CFAD fakes), which training drops.
    It is a ranking instrument, not an estimate.
- **D6 — the GPUs were ~53 % busy.**
  - Sampled with `nvidia-smi` on all 8 GPUs during run 1.
  - The main process is kernel-launch-bound: the per-row BEATs filterbank loop, many small
    kernels. Data wait was ~6 s per 64 s.
  - Candidates for run 3: batch the per-row fbank and patch embedding, CUDA graphs or
    `torch.compile`. Unmeasured.

**The first submission.** `/data/project/private/dacon-submissions/first-v3-seed0.zip` is
1.68 GB, sha256 prefix `f4eba8fbc8f8afa5`: all_data_seed0 alone.

- It was verified under `server-mirror`: offline load from `model/weights/`, WAV / FLAC / MP3,
  0 fallback rows.
- Timing: 200 × 60 s files took 38 s wall on one H200 (3 GB RSS). The L4 worst case is ≈ 20–30
  min against the 60-min limit (5–8× slowdown assumed).
- **UNVERIFIED:** whether the owner has uploaded it, and its LB score. Ask.

### 3.3 Round-2 data (09) — state at 00:05 KST

The owner decided on 2026-09-25: **English and Korean only**. Chinese, Japanese and other
languages are out. `configs/processing_run2.yaml` draws ko 0.55 / en 0.45.

| corpus (`interim/…`) | status | numbers |
|---|---|---|
| `proc/` (S1: EnCodec, DAC 16/44k, XCodec2+Mimi, DeepFilterNet3, DSP pitch/formant/tempo; REAL **and** FAKE) | **done** | 30.8 h real / 30.8 h fake, 6.1–6.2 h per family per side (`proc/REPORT.md`) |
| `emilia-ko/` (S3: real conversational Korean) | **done** | Emilia 40.0 h + Emilia-YODAS 40.0 h + FLEURS 8.4 h, 19,968 speakers (`metadata.csv`, `LICENCE_NOTES.md`) |
| `ko-synth2/` (S2: modern Korean clones) | **running until 04:50 KST** | kept at 23:45: seedvc 8.15 h, maskgct 3.54 h, fishspeech 2.38 h, chatterbox 1.57 h, cosyvoice 0.49 h (QC rejects 5–21 %). **llasa dropped**: 0.14× real time (`_dropped/llasa`) |
| `zh-synth/`, `ctrsvdd/` | **stopped by owner decision**; on disk, not ingested | zh-synth 3.49 h (measured at handoff; an earlier report to the owner said 0.4 h — superseded); ctrsvdd 60.2 h zh singing |

**Live jobs at handoff.** Six `ko_run.sbatch` shards:
- chatterbox 0/2, 1/2
- maskgct 0/2, 1/2
- fishspeech 0/1
- cosyvoice 0/2

Each stops itself at `--deadline 1790365800` = **2026-09-26 04:50 KST**. They are resumable:
per-file metadata is appended under a lock, and each job rewrites its family's `metadata.csv`
after a final QC pass.

**The dispatcher** `scripts/synth2/ko_dispatch.sh` (PID 167805 at handoff) runs **inside the
old session**.

- It keeps ≤ 6 jobs queued from `interim/ko-synth2/_dispatch/queue.txt`, which has one line
  left: `cosyvoice … --shard 1/2`. It stops at 04:30.
- **If it died with the old session, the last shard is never submitted.** Check with
  `ps -u $USER -o pid,cmd | grep ko_dispatch`.
- If it is gone, submit the line by hand:
  `sbatch scripts/synth2/ko_run.sbatch <line>`.

The agents that ran these tracks belonged to the old session. **Nothing will supervise the
jobs unless you do.**

### 3.4 Open problems

| # | problem | evidence | where to act |
|---|---|---|---|
| P1 | same-speaker (matched-pair) weakness | D1 | round-2 data: clones of the same speakers as real Emilia/Zeroth; consider same-speaker real/fake pairs in the draw |
| P2 | unseen Korean cloner weakness | D2 | ko-synth2 (5 new families); hold 1–2 out per fold in run 2 and report per-family EER |
| P3 | validation easier than the test | D4, PROBE | calibrate on the first LB score; harder composed eval (real mixtures, telephone, S1 processing) |
| P4 | GPU utilisation ~53 % | D6 | launch overhead, before run 3 |
| P5 | cell 5 / 6 music-side weakness | D3 | sonics-sep and more fake-music families; check the music head's features |
| P6 | the full test suite is not re-run on HEAD | §1 | before merging `feat/training` |

## 4 · Owner decisions and standing instructions (binding)

- **Compute.**
  - Use all 8 H200s on the debug node, through Slurm.
  - Every job is named `eval-hyeonseop`.
  - The owner's vLLM server on this node was stopped with their approval on 2026-09-24;
    restart it only if asked.
- **Scope of run 1.** "Put all things in the first run." Done.
- **Languages.** English and Korean are primary; Chinese, Japanese and others are dropped for
  now. The data already made is kept on disk, not used.
- **Data.**
  - Korean fakes are synthesised by us.
  - Emilia is used **including** YODAS.
  - TTS texts come from dataset transcripts, not LLM scripts.
  - Fake music: Demucs-separated SONICS on both labels (run 1).
- **Commit practice.** Commit to the branch at every solid advance.
  - Stage explicit paths, never `git add -A`.
  - Keep the touched suites green before each commit.
  - Message style: a one-line subject, then a body with what was measured, then:
    ```
    Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
    Claude-Session: https://claude.ai/code/session_01Lz3hGY4Gxhf5yvgiyJJz7G
    ```
- **Forbidden.**
  - Pushing or opening PRs without the owner's go.
  - `pkill -f` on a pattern in your own command line.
  - Killing jobs or processes you did not start.
  - Writing into an existing manifest dir.
  - Using PROBE for anything but a final check.
- **Submissions.** They go through the owner on DACON's web page, 3 per day. Hand them a verified
  zip path.

## 5 · Remaining work, in order

1. **Report back to the owner.** Confirm the first submission's status and LB score (§3.2).
2. **Diagnose run 1** (the owner's stated next step):
   - per-family and per-cell EERs from `runs/first-v3/fold*/val_predictions.parquet`;
   - `t3_gap`;
   - score histograms per head;
   - calibration of fold scores against the LB once it is known.
   `scripts/eval_checkpoint.py` scores any `joint-pass<k>.pt` (all 18 per fold are kept) for
   learning curves. Run it **inside a Slurm allocation, `-n 1`** (§6 mistake 3).
3. **Supervise round 2 until 04:50** (§3.3). Then check each `ko-synth2/<family>/metadata.csv`
   exists and read QC rejects.
4. **Build strategy-v4:** `bash scripts/build_strategy_v4.sh` (default `ONLY=proc,emilia-ko,ko-synth2`).
   - It runs manifest → folds → cache → both audits at n = 40,000, and stops on the first failure.
   - Check: the audit passes, the drawn language shares are ≈ ko .55 / en .45 on both sides,
     and the drawn *processed* share is equal on both sides.
   - Check where the ko-synth2 families landed in the folds (`folds.caveats.txt`, and
     `folds.parquet` by `artifact_family`).
   - The dry run on the processed data alone passed; the Emilia and ko-synth2 readers are
     **untested on real files**.
5. **Run 2.** Options put to the owner, **undecided**:
   - **A**, from scratch, ~20 h;
   - **B (recommended)**, `--init-weights` from run 1 at ~20,000 steps, ~8 h. Fold k continues
     from run 1's **fold k** `scored.pt`, never from an all-data model, so held-out folds stay clean.
   - Launch:
     ```
     PROCESSING=configs/processing_run2.yaml MANIFEST_DIR=…/strategy-v4 sbatch scripts/train_first_run.sbatch <name> [extra args]
     ```
   - For B, the per-task init path needs a small launcher change: pass `--init-weights` per
     array task.
   - Adjust `--draws` / `epochs` in `configs/train_first_run.yaml` or on the command line.
6. **Package and verify** each candidate:
   - `scripts/package_submission.py --out … --scored …` (one `--scored`: the soup did not help);
   - unzip, then run `script.py` under `server-mirror` on a few files;
   - rank candidates with `scripts/eval_probe.py`.
7. **Before merging `feat/training`:** the full suite in two halves (§1), then ask the owner
   about push and PR.

## 6 · Mistakes made in this phase — do not repeat

| # | what happened | the rule |
|---|---|---|
| 1 | Sub-agents spawned without a model pinned inherited Fable and died instantly at a usage limit, twice — six agents lost | spawn agents with `model: "opus"` explicitly |
| 2 | An agent went silent for 4 h after sending a message; nothing noticed until its output was missing | check each agent's files/logs within ~30 min of launch and again periodically; relaunch with a "state left" brief |
| 3 | A mid-run evaluation launched with `srun --overlap` **without `-n 1`** started 4 copies on a training GPU; the memory they took killed fold 1 at step 3,850 (recovered with `--resume`) | never overlap a training allocation casually; if you must, `-n 1` and `scripts/eval_checkpoint.py`'s `--gpu-fraction` cap |
| 4 | `pgrep -f build_cache.py` matched the shell running it: "already running", nothing built | find processes by PID or `nvidia-smi --query-compute-apps=pid` |
| 5 | Eight concurrent tasks ran at 22 s/step with the GPUs idle: the CUDA JIT cache in `~/.nv` (EFS) serialised them over NFS | `CUDA_CACHE_PATH=/tmp/…` per job (the launcher does it) |
| 6 | Memory grew 0.75 GB/step until cuFFT failed: one cuFFT plan per transform length, outside PyTorch's allocator | `models/audio.py` caps the plan cache at 8 — keep it (it also protects L4 inference) |
| 7 | OOM at batch 32 and 8: both trunks stored every layer's attention maps | layer checkpointing (`loop.grad_checkpointing`) + batch 16 |
| 8 | BEATs was 3.7 of a 3.9 s step (one 3,000-token attention per 60 s row); then per-length encoder calls made it launch-bound | 62-column windows padded to one encoder call (`window_patches`) |
| 9 | A multi-edit patch script aborted halfway and left one file edited and the rest not | every patch script checks each anchor before writing; `grep` that each edit landed |
| 10 | The per-side language renormalisation left Chinese as real-only (CFAD fakes were in PROBE): 20 % of real draws vs 0 % fake | the both-sides rule; **measure** drawn shares with `audit_fold.py`, don't assume |
| 11 | Audit I3 read 0.499 at n = 20,000 against a 0.5 floor. It tracks reuse counts (2p(1−p) = 0.48), not a leak | audit at n ≥ 40,000 |
| 12 | Processed FAKE rows used a processing source key; `check_rules` requires a fake row's `source_name` = its family | the dry run caught it — always dry-run `extend_manifest.py` before building |
| 13 | The four-model soup was assumed to help; it scored worst on PROBE | compare candidates on PROBE before packaging |
| 14 | A stopped agent submitted a job after the stop order, under a non-conforming name (`zh-synth`) | after any stop order, re-check `squeue -u $USER` yourself and cancel stragglers |
| 15 | Slurm FAILED on a fold task was nearly read as a crash | train.py exits 1 on "not quotable"; read `ledger_row.json` |
| 16 | Pooled validation scores look excellent (0.95) while same-speaker and unseen-Korean-cloner numbers are poor | judge models on D1 / D2 first |

## 7 · Quick reference

```bash
V=/data/project/private/dacon-venvs/dacon311/bin/python
cd /home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
# run 1 artifacts
ls /data/project/private/dacon-runs/first-v3/{fold0,all_data_seed0}
# a fold's breakdown
python3 -c "import json;r=json.load(open('/data/project/private/dacon-runs/first-v3/fold1/ledger_row.json'));print(r['per_family_eer'])"
# matched-pair gap
$V -c "import pandas as pd;from training.validate import t3_gap;print(t3_gap(pd.read_parquet('/data/project/private/dacon-runs/first-v3/fold0/val_predictions.parquet'),head='voice'))"
# round-2 progress
for m in /data/project/private/dacon-corpus/interim/ko-synth2/*/metadata.csv; do echo $m; done
```

You can use oh-my-claudecode skills if needed.
