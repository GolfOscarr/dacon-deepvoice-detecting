#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH -J eval-hyeonseop
#SBATCH -o /data/project/private/dacon-corpus/interim/ko-synth/_logs/slurm-%x-%j.out
# usage: sbatch run_family.sbatch <family> [script args...]
# One Slurm job = one GPU = one shard of one family. Launch several with
# different --shard i/n to spread a family over GPUs.
set -u
FAMILY=$1; shift
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
export HF_HOME=/data/project/private/dacon-weights/synth/hf
export TORCH_HOME=/data/project/private/dacon-weights/synth/torch
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1
VENV=/data/project/private/dacon-venvs/synth-${FAMILY%%-*}
cd "$REPO"
echo "job $SLURM_JOB_ID family=$FAMILY host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES args=$*"
exec "$VENV/bin/python" "scripts/synth/synth_${FAMILY%%-*}.py" "$@"
