#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=100G
#SBATCH -J eval-hyeonseop
#SBATCH -o /data/project/private/dacon-corpus/interim/ko-synth2/_logs/slurm-%j.out
# One GPU job = one shard of one Korean round-2 family + its QC follower (Whisper large-v3
# on the same GPU), then a final QC pass and metadata.csv rewrite. Resumable.
# usage: sbatch scripts/synth2/ko_run.sbatch <family> [ko_synth_<family>.py args, e.g. --shard 0/2 --limit 5]
# English: SYNTH2_LANG=en sbatch -o /data/project/private/dacon-corpus/interim/en-synth2/_logs/slurm-%j.out \
#          scripts/synth2/ko_run.sbatch <family> ...   (output and logs under interim/en-synth2/)
set -u
FAMILY=$1; shift
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
S2=$REPO/scripts/synth2
V=/data/project/private/dacon-venvs
export SYNTH2_LANG=${SYNTH2_LANG:-ko}
LOGS=/data/project/private/dacon-corpus/interim/$SYNTH2_LANG-synth2/_logs
mkdir -p "$LOGS"
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TORCH_HOME=/data/project/private/dacon-weights/synth2/torch
SHARD=0/1
prev=""
for a in "$@"; do [ "$prev" = "--shard" ] && SHARD=$a; prev=$a; done
case $FAMILY in
  cosyvoice) VENV=$V/synth-cosyvoice; export HF_HOME=/data/project/private/dacon-weights/synth/hf ;;
  *)         VENV=$V/synth2-$FAMILY;  export HF_HOME=/data/project/private/dacon-weights/synth2/hf ;;
esac
cd "$REPO"
TAG="$FAMILY-${SHARD/\//of}-$SLURM_JOB_ID"
echo "job $SLURM_JOB_ID family=$FAMILY shard=$SHARD host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES args=$* start=$(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
# exit 3 = the script hit a poisoned CUDA context and asks for a relaunch (resumable)
( for r in $(seq 0 40); do
    "$VENV/bin/python" "$S2/ko_synth_$FAMILY.py" "$@" >> "$LOGS/synth-$TAG.log" 2>&1; rc=$?
    [ $rc -eq 3 ] || exit $rc
    echo "relaunch $r after exit 3 $(date +%T)" >> "$LOGS/synth-$TAG.log"
  done ) &
SP=$!
HF_HOME=/data/project/private/dacon-weights/synth2/hf $V/synth2-qc/bin/python "$S2/ko_qc.py" qc \
  --family "$FAMILY" --shard "$SHARD" --wait-pid $SP > "$LOGS/qc-$TAG.log" 2>&1
wait $SP; RC=$?
# the follower may have exited on its last pass before the synth's final rows; one more pass
HF_HOME=/data/project/private/dacon-weights/synth2/hf $V/synth2-qc/bin/python "$S2/ko_qc.py" qc \
  --family "$FAMILY" --shard "$SHARD" >> "$LOGS/qc-$TAG.log" 2>&1
echo "synth exit=$RC"; tail -n 3 "$LOGS/synth-$TAG.log"; tail -n 2 "$LOGS/qc-$TAG.log"
echo "job done $(date)"
