#!/usr/bin/env bash
#SBATCH -p debug
#SBATCH -J eval-hyeonseop
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=06:00:00
#SBATCH -o /data/project/private/dacon-corpus/interim/_logs/sing-sep-%j.out
# N2: htdemucs vocal stems for a selection. Usage: SEL=<selection.csv> TAG=<tag> sbatch scripts/synth3/sing_separate.sbatch
set -euo pipefail
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID
export TORCH_HOME=/data/project/private/dacon-weights/demucs
export OMP_NUM_THREADS=2
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
cd "$REPO"
/data/project/private/dacon-venvs/dacon311/bin/python scripts/synth3/sing_separate.py \
    --selection "${SEL:?}" --tag "${TAG:?}" --batch 16 --loaders 6 "$@"
