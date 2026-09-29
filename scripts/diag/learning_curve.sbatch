#!/bin/bash
# Score a fold's pass checkpoints on its frozen VAL draw (appends to midrun_eval.jsonl).
#   sbatch scripts/diag/learning_curve.sbatch <run_dir> <fold> "<pass> <pass> ..."
#SBATCH -J eval-hyeonseop
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -o /data/project/private/dacon-runs/_logs/lc-%j.out
set -euo pipefail
RUN=$1; FOLD=$2; PASSES=$3
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID
V=/data/project/private/dacon-venvs/dacon311/bin/python
W=/data/project/private/dacon-weights
for p in $PASSES; do
  echo "== fold$FOLD pass$p $(date +%T)"
  $V scripts/eval_checkpoint.py --ckpt "$RUN/fold$FOLD/joint-pass$p.pt" --fold "$FOLD" \
    --manifest-dir /data/project/private/dacon-corpus/manifests/strategy-v3 \
    --processing configs/processing_first_run.yaml --model configs/c_first_run.yaml \
    --weights audio=$W/beats,speech=$W/xlsr-300m --eval-n 2000 --gpu-fraction 0.95
done
