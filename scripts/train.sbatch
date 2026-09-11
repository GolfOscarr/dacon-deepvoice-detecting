#!/bin/bash
# Run scripts/train.py under Slurm. OPTIONAL -- see below.
#
#   sbatch scripts/train.sbatch --corpus /data/corpus/test-v1 --out runs/t1 \
#       --weights /data/weights/beats --folds all --select soup
#
#   # one job per fold, so the folds run concurrently on different nodes
#   sbatch --array=0-1 scripts/train.sbatch --corpus /data/corpus/test-v1 \
#       --out runs/t1 --weights /data/weights/beats --select soup
#
# Everything after the script name is passed through to train.py verbatim. Under
# --array, $SLURM_ARRAY_TASK_ID supplies --folds, so do not also pass --folds.
#
# ⚠️ AN ARRAY DOES NOT GIVE YOU A CROSS-FOLD NUMBER. Each task is a separate
# invocation, so each aggregates its own single fold and reports Score_sd = 0.0
# with sd_is_a_measurement = False. Use `--folds all` in ONE job when you want the
# mean over folds; use the array when you want the folds to run concurrently and
# will combine the per-fold ledger rows yourself. Combining them is not yet
# implemented -- `aggregate_folds` needs FoldResults, not just the saved predictions.
#
# ⚠️ train.py exits 1 when a run is not quotable, so Slurm marks the task FAILED.
# That is right for a real run and noise for a deliberate smoke test; pass
# --allow-unquotable for the latter.
#
# ⚠️ WHY THIS IS OPTIONAL. On a GPU node you have to yourself, running train.py
# directly is simpler and this adds nothing. What it buys is not speed, it is not
# colliding with anyone: a node the scheduler believes is idle can be allocated to
# someone else's job while your processes hold its GPU memory, and neither job is
# told why it slowed down or ran out of memory. That happened during this repo's
# verification pass -- an 8-GPU job landed on a node being used outside the
# scheduler, leaving 9.7 GiB free per device. Use this whenever the node is shared.
#
# 🔴 /tmp IS NODE-LOCAL on this cluster. A job reading its script, corpus, weights
# or venv from /tmp fails on every node but the one that wrote it -- the first
# submission of this wrapper's ancestor died exactly that way. /data/project/private
# is wekafs and shared; /home is shared EFS but slow. The preflight below refuses to
# start rather than fail halfway through a transfer.

#SBATCH --job-name=dacon-train
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.out

set -euo pipefail

: "${DACON_VENV:=/data/project/private/dacon-venvs/dacon311}"

# 🔴 NOT $BASH_SOURCE. Slurm copies the batch script into its spool directory
# before running it, so under sbatch BASH_SOURCE[0] is /var/spool/slurm/d/... and
# the repo is nowhere near it -- the first submission of this wrapper died with
# "missing /var/spool/slurm/d/scripts/train.py". $SLURM_SUBMIT_DIR is the
# directory sbatch was invoked from; DACON_REPO overrides it for a submission made
# from somewhere else, and a direct `bash scripts/train.sbatch` still works.
if [ -n "${DACON_REPO:-}" ]; then
  REPO="$DACON_REPO"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -e "$SLURM_SUBMIT_DIR/scripts/train.py" ]; then
  REPO="$SLURM_SUBMIT_DIR"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_TASK_ID:-none}"
echo "gpus=${CUDA_VISIBLE_DEVICES:-none} cpus=${SLURM_CPUS_PER_TASK:-?} repo=$REPO"

# One GPU per invocation is deliberate (folds are independent; see train.py). More
# than one would sit idle, so say so rather than silently waste the allocation.
n_gpu=$(tr -cd ',' <<<"${CUDA_VISIBLE_DEVICES:-}" | wc -c)
if [ "${CUDA_VISIBLE_DEVICES:-}" != "" ] && [ "$n_gpu" -gt 0 ]; then
  echo "WARNING: $((n_gpu + 1)) GPUs allocated; train.py uses one. Request --gres=gpu:1."
fi

for p in "$DACON_VENV/bin/python" "$REPO/scripts/train.py"; do
  [ -e "$p" ] || { echo "FATAL: missing $p (set DACON_REPO to the checkout, or "\
    "submit from it so \$SLURM_SUBMIT_DIR points there)"; exit 1; }
  case "$p" in /tmp/*) echo "FATAL: $p is under /tmp, which is node-local"; exit 1;; esac
done

# The corpus and weights arrive as pass-through arguments, so check the ones given.
prev=""
for arg in "$@"; do
  case "$prev" in
    --corpus) [ -e "$arg/manifest.parquet" ] || { echo "FATAL: no manifest.parquet under $arg"; exit 1; }
              [ -e "$arg/folds.parquet" ]    || { echo "FATAL: no folds.parquet under $arg"; exit 1; } ;;
    --weights) [ -d "$arg" ] || { echo "FATAL: --weights $arg is not a directory"; exit 1; } ;;
  esac
  case "$arg" in /tmp/*) echo "FATAL: $arg is under /tmp, which is node-local"; exit 1;; esac
  prev="$arg"
done

nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

fold_arg=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  fold_arg=(--folds "$SLURM_ARRAY_TASK_ID")
  echo "array task -> --folds $SLURM_ARRAY_TASK_ID"
fi

cd "$REPO"
# train.py exits non-zero when the run is not quotable, which is the signal an
# array should surface rather than swallow.
exec "$DACON_VENV/bin/python" scripts/train.py "${fold_arg[@]}" "$@"
