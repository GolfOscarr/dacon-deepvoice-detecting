#!/usr/bin/env bash
#SBATCH -p debug
#SBATCH -J eval-hyeonseop
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=03:00:00
#SBATCH -o /data/project/private/dacon-corpus/interim/_logs/sing-aisong-post-%j.out
# N2c after generation: tile the ACE-Step songs, htdemucs vocal stems, measure (same code as N2a/b).
set -euo pipefail
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID
export TORCH_HOME=/data/project/private/dacon-weights/demucs
export HF_HOME=/data/project/private/dacon-weights/synth2/hf HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
W=/data/project/private/dacon-corpus/interim/_sing_work
PY=/data/project/private/dacon-venvs/dacon311/bin/python
cd /home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
$PY scripts/synth3/sing_select.py --work $W --aisong /data/project/private/dacon-corpus/interim/aisong-kr-en
$PY scripts/synth3/sing_separate.py --selection $W/selection_aisong.csv --tag aisong --batch 16 --loaders 6
$PY scripts/synth3/sing_screen.py measure --work $W --threads 6
