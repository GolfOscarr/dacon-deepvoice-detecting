#!/bin/bash
# The first real training run (docs/training/07 §3): eight tasks, one GPU each,
# on the debug node.
#
#   sbatch scripts/train_first_run.sbatch <run name> [extra train.py args...]
#
# task 0-3  -> --folds <task>              the CV estimate and the step count
# task 4-7  -> --all-data --seed <task-4>  the models that ship
#
# Everything shared (manifest, configs, weights) is set below; extra arguments
# pass through to every task. Output: runs/<run name>/{fold<k>,all_data_seed<s>}.

#SBATCH --job-name=dacon-first
#SBATCH --partition=debug
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --output=runs/slurm-%x-%A_%a.out
#SBATCH --error=runs/slurm-%x-%A_%a.out

set -euo pipefail
: "${DACON_VENV:=/data/project/private/dacon-venvs/dacon311}"
: "${MANIFEST_DIR:=/data/project/private/dacon-corpus/manifests/strategy-v3}"
: "${WEIGHTS:=audio=/data/project/private/dacon-weights/beats,speech=/data/project/private/dacon-weights/xlsr-300m}"
REPO="${DACON_REPO:-$SLURM_SUBMIT_DIR}"
RUN="${1:?usage: sbatch scripts/train_first_run.sbatch <run name> [train.py args]}"
shift
TASK="${SLURM_ARRAY_TASK_ID:?run under sbatch --array}"

for p in "$DACON_VENV/bin/python" "$REPO/scripts/train.py" "$MANIFEST_DIR/manifest.parquet" \
         "$MANIFEST_DIR/folds.parquet"; do
  [ -e "$p" ] || { echo "FATAL: missing $p"; exit 1; }
done

if [ "$TASK" -lt 4 ]; then
  role=(--folds "$TASK" --select ema)
else
  role=(--all-data --seed "$((TASK - 4))" --select ema)
fi
echo "node=$(hostname) job=${SLURM_ARRAY_JOB_ID:-?}_${TASK} gpu=${CUDA_VISIBLE_DEVICES:-?} role=${role[*]}"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$DACON_VENV/bin/python" scripts/train.py \
  --manifest-dir "$MANIFEST_DIR" \
  --processing configs/processing_first_run.yaml \
  --model configs/c_first_run.yaml \
  --train configs/train_first_run.yaml \
  --weights "$WEIGHTS" \
  --stages joint \
  --out "runs/$RUN" \
  --device cuda \
  --draws 48000 \
  --eval-n 6000 \
  "${role[@]}" "$@"
