#!/bin/bash
#SBATCH -J eval-hyeonseop
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -o /data/project/private/dacon-corpus/interim/proc/_logs/gpu-%j.out
# S1 GPU codecs: DAC and XCodec2 share one GPU (two processes). Resumable: a rerun
# skips files already in interim/proc/metadata.csv.
set -u
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID
export HF_HOME=/data/project/private/dacon-weights/proc/hf HF_HUB_OFFLINE=1
V=/data/project/private/dacon-venvs
L=/data/project/private/dacon-corpus/interim/proc/_logs
cd /home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
nvidia-smi -L
$V/proc-dac/bin/python scripts/proc/run_dac.py --device cuda > $L/dac-$SLURM_JOB_ID.log 2>&1 &
$V/proc-xcodec2/bin/python scripts/proc/run_xcodec2.py --device cuda > $L/xcodec2-$SLURM_JOB_ID.log 2>&1 &
wait
tail -2 $L/dac-$SLURM_JOB_ID.log $L/xcodec2-$SLURM_JOB_ID.log
