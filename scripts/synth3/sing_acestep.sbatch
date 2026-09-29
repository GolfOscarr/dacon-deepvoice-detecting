#!/usr/bin/env bash
#SBATCH -p debug
#SBATCH -J eval-hyeonseop
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH -o /data/project/private/dacon-corpus/interim/_logs/sing-acestep-%j.out
# N2c: ACE-Step 1.5 songs, ko/en lyrics. Usage: sbatch scripts/synth3/sing_acestep.sbatch [extra args]
set -euo pipefail
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID
export ACESTEP_CHECKPOINTS_DIR=/data/project/private/dacon-weights/sing/acestep15/checkpoints
export HF_HOME=/data/project/private/dacon-weights/sing/hf-home
export OMP_NUM_THREADS=4
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
/data/project/private/dacon-venvs/sing-acestep15/bin/python $REPO/scripts/synth3/sing_acestep_gen.py \
    --out ${OUT:-/data/project/private/dacon-corpus/interim/aisong-kr-en} "$@"
