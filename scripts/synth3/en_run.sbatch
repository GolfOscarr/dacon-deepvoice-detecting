#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -J eval-hyeonseop
#SBATCH -o /data/project/private/dacon-corpus/interim/en-synth3/_logs/slurm-%j.out
# One GPU job = one shard (or EN3_PROCS shards) of one English round-3 family + its Whisper QC follower on the same GPU
# (round 2's ko_qc.py in English mode), then a final QC pass. Resumable.
# usage: sbatch scripts/synth3/en_run.sbatch <family> [en_synth_<family>.py args, e.g. --shard 0/2 --limit 10]
# N5 (maskgct / seedvc extra hours): EN3_ROOT=/data/project/private/dacon-corpus/interim/en-synth2-extra
#   and EN3_EXCLUDE_SPEAKERS=<file>; exported through sbatch's default --export=ALL.
set -u
FAMILY=$1; shift
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
S3=$REPO/scripts/synth3
V=/data/project/private/dacon-venvs
export SYNTH2_LANG=en
export KO_SYNTH2_ROOT=${EN3_ROOT:-/data/project/private/dacon-corpus/interim/en-synth3}
LOGS=/data/project/private/dacon-corpus/interim/en-synth3/_logs
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TORCH_HOME=/data/project/private/dacon-weights/synth2/torch
export NLTK_DATA=/data/project/private/dacon-weights/synth3/nltk_data
# EN3_PROCS=k (k > 1): run k shards p/k of the family side by side on this one GPU (the AR models
# leave an H200 mostly idle), each with its own QC follower. Do not also pass --shard then.
PROCS=${EN3_PROCS:-1}
[ "$PROCS" -gt 1 ] && export OMP_NUM_THREADS=2
case $FAMILY in
  maskgct|seedvc) VENV=$V/synth2-$FAMILY;  export HF_HOME=/data/project/private/dacon-weights/synth2/hf ;;
  *)              VENV=$V/synth3en-$FAMILY; export HF_HOME=/data/project/private/dacon-weights/synth3/hf ;;
esac
cd "$REPO"
echo "job $SLURM_JOB_ID family=$FAMILY root=$KO_SYNTH2_ROOT procs=$PROCS host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES args=$* start=$(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
QCPY="env HF_HOME=/data/project/private/dacon-weights/synth2/hf $V/synth2-qc/bin/python $REPO/scripts/synth2/ko_qc.py"

run_one() {  # run_one <shard> <synth args...>
  local SHARD=$1; shift
  local TAG="$(basename "$KO_SYNTH2_ROOT")-$FAMILY-${SHARD/\//of}-$SLURM_JOB_ID"
  # exit 3 = poisoned CUDA context, relaunch (resumable)
  ( for r in $(seq 0 40); do
      "$VENV/bin/python" "$S3/en_synth_$FAMILY.py" "$@" >> "$LOGS/synth-$TAG.log" 2>&1; rc=$?
      [ $rc -eq 3 ] || exit $rc
      echo "relaunch $r after exit 3 $(date +%T)" >> "$LOGS/synth-$TAG.log"
    done ) &
  local SP=$!
  $QCPY qc --family "$FAMILY" --shard "$SHARD" --wait-pid $SP > "$LOGS/qc-$TAG.log" 2>&1
  wait $SP; local RC=$?
  $QCPY qc --family "$FAMILY" --shard "$SHARD" >> "$LOGS/qc-$TAG.log" 2>&1
  echo "shard $SHARD synth exit=$RC"; tail -n 2 "$LOGS/synth-$TAG.log"; tail -n 2 "$LOGS/qc-$TAG.log"
}

if [ "$PROCS" -gt 1 ]; then
  LO=${EN3_SHARD_LO:+@$EN3_SHARD_LO}   # EN3_SHARD_LO=n: only job idx >= n (skip earlier failures on a top-up)
  for p in $(seq 0 $((PROCS - 1))); do run_one "$p/$PROCS$LO" "$@" --shard "$p/$PROCS$LO" & done
  wait
else
  SHARD=0/1
  prev=""
  for a in "$@"; do [ "$prev" = "--shard" ] && SHARD=$a; prev=$a; done
  run_one "$SHARD" "$@"
fi
echo "job done $(date)"
