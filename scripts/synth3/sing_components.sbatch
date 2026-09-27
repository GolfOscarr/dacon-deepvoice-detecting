#!/usr/bin/env bash
#SBATCH -p debug
#SBATCH -J eval-hyeonseop
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=07:00:00
#SBATCH -o /data/project/private/dacon-corpus/interim/_logs/sing-components-%j.out
# Lead's extra: vocal-activity screen over every v4 pool C/D component (same N2 code path).
set -euo pipefail
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID
export TORCH_HOME=/data/project/private/dacon-weights/demucs
export HF_HOME=/data/project/private/dacon-weights/synth2/hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
W=/data/project/private/dacon-corpus/interim/sing-real/_screen
PY=/data/project/private/dacon-venvs/dacon311/bin/python
cd /home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
$PY scripts/synth3/sing_separate.py --selection $W/selection_components.csv --tag comp --batch 32 --loaders 7 --out-base $W/stems
$PY scripts/synth3/sing_screen.py measure --work $W --threads 6 --out-base $W/stems
