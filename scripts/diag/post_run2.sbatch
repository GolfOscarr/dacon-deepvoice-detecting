#!/bin/bash
# Run 2's post-run scoring (docs/training/11 §4), after train_run2.sbatch finishes:
#   tasks 0-5: T0..T5's scored.pt on run 1's EXACT v3 VAL specs (v3 manifest,
#              processing_first_run.yaml, eval-n 6000, seed 1234) -> <run>/T<i>/v3val/fold<k>/
#   task 6:    PROBE for T6, T7 and run 1's all_data_seed0 (the same frozen draw run 1 was ranked on)
#
#   sbatch scripts/diag/post_run2.sbatch <run dir>
#SBATCH -J eval-hyeonseop
#SBATCH -p debug
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 03:00:00
#SBATCH -o /data/project/private/dacon-runs/_logs/post2-%A_%a.out
set -euo pipefail
RUN=${1:?run dir}
TASK=${SLURM_ARRAY_TASK_ID:?}
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID OMP_NUM_THREADS=1
mkdir -p "$CUDA_CACHE_PATH"
V=/data/project/private/dacon-venvs/dacon311/bin/python
W=audio=/data/project/private/dacon-weights/beats,speech=/data/project/private/dacon-weights/xlsr-300m
V3=/data/project/private/dacon-corpus/manifests/strategy-v3
R1=/data/project/private/dacon-runs/first-v3
cd "${SLURM_SUBMIT_DIR}"
FOLD=(0 1 2 3 1 0)                         # T0..T5's fold (train_run2.sbatch TASKS)
if [ "$TASK" -le 5 ]; then
  k=${FOLD[$TASK]}
  $V scripts/eval_checkpoint.py --ckpt "$RUN/T$TASK/fold$k/scored.pt" --fold "$k" \
    --manifest-dir $V3 --processing configs/processing_first_run.yaml --model configs/c_run2.yaml \
    --weights $W --eval-n 6000 --eval-seed 1234 --gpu-fraction 0.95 --out-dir "$RUN/T$TASK/v3val"
else
  $V scripts/eval_probe.py --manifest-dir $V3 --processing configs/processing_first_run.yaml \
    --scored "$RUN/T6/all_data_seed0/scored.pt" --scored "$RUN/T7/all_data_seed2/scored.pt" \
    --scored "$R1/all_data_seed0/scored.pt" --weights $W --out "$RUN/probe"
fi
