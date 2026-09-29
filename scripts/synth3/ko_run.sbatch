#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -J eval-hyeonseop
#SBATCH -o /data/project/private/dacon-corpus/interim/ko-synth3/_logs/slurm-%j.out
# Round-3 Korean families (docs/training/15 N1-ko) and the N5 extras: one GPU job = one shard of
# one family + its Whisper QC follower on the same GPU (scripts/synth2/ko_qc.py), as ko-synth2.
# usage: sbatch scripts/synth3/ko_run.sbatch <family> [ko_synth_<family>.py args, e.g. --shard 0/2 --limit 10]
#   families with scripts/synth3/ko_synth_<family>.py write interim/ko-synth3/<family>/;
#   maskgct|seedvc (N5) run scripts/synth2's script into interim/ko-synth2-extra/<family>/.
set -u
FAMILY=$1; shift
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
S2=$REPO/scripts/synth2
S3=$REPO/scripts/synth3
V=/data/project/private/dacon-venvs
I=/data/project/private/dacon-corpus/interim
LOGS=$I/ko-synth3/_logs
export SYNTH2_LANG=ko
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TORCH_HOME=/data/project/private/dacon-weights/synth2/torch
SHARD=0/1
prev=""
for a in "$@"; do [ "$prev" = "--shard" ] && SHARD=$a; prev=$a; done
HF3=/data/project/private/dacon-weights/synth3/hf
case $FAMILY in
  maskgct|seedvc) SCRIPT="$S3/ko_extra.py $FAMILY"; VENV=$V/synth2-$FAMILY
                  export HF_HOME=/data/project/private/dacon-weights/synth2/hf
                  export KO_SYNTH2_ROOT=$I/ko-synth2-extra ;;
  rvctrain)       SCRIPT=$S3/ko_rvc_train.py; VENV=$V/synth3-rvc; export HF_HOME=$HF3
                  export KO_SYNTH2_ROOT=$I/ko-synth3 ;;
  cosyvoice3)     SCRIPT=$S3/ko_synth_$FAMILY.py; VENV=$V/synth-cosyvoice; export HF_HOME=$HF3
                  export KO_SYNTH2_ROOT=$I/ko-synth3 ;;
  *)              SCRIPT=$S3/ko_synth_$FAMILY.py; VENV=$V/synth3-$FAMILY; export HF_HOME=$HF3
                  export KO_SYNTH2_ROOT=$I/ko-synth3 ;;
esac
cd "$REPO"
TAG="$FAMILY-${SHARD/\//of}-$SLURM_JOB_ID"
echo "job $SLURM_JOB_ID family=$FAMILY shard=$SHARD out=$KO_SYNTH2_ROOT host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES args=$* start=$(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
# exit 3 = the script asks for a relaunch (poisoned CUDA context; resumable)
( for r in $(seq 0 40); do
    "$VENV/bin/python" $SCRIPT "$@" >> "$LOGS/synth-$TAG.log" 2>&1; rc=$?
    [ $rc -eq 3 ] || exit $rc
    echo "relaunch $r after exit 3 $(date +%T)" >> "$LOGS/synth-$TAG.log"
  done ) &
SP=$!
if [ "$FAMILY" = rvctrain ]; then wait $SP; echo "rvctrain exit=$?"; tail -n 5 "$LOGS/synth-$TAG.log"; exit 0; fi
HF_HOME=/data/project/private/dacon-weights/synth2/hf $V/synth2-qc/bin/python "$S2/ko_qc.py" qc \
  --family "$FAMILY" --shard "$SHARD" --wait-pid $SP > "$LOGS/qc-$TAG.log" 2>&1
wait $SP; RC=$?
HF_HOME=/data/project/private/dacon-weights/synth2/hf $V/synth2-qc/bin/python "$S2/ko_qc.py" qc \
  --family "$FAMILY" --shard "$SHARD" >> "$LOGS/qc-$TAG.log" 2>&1
echo "synth exit=$RC"; tail -n 3 "$LOGS/synth-$TAG.log"; tail -n 2 "$LOGS/qc-$TAG.log"
echo "job done $(date)"
