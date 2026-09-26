#!/bin/bash
# DDP training (docs/training/14 D7): ONE job, N GPUs on the debug node, torchrun --standalone,
# one process per GPU. train.py builds the same draw on every rank at the global batch
# (--batch-size per rank x N x loop.grad_accum) and trains its slice; rank 0 writes everything.
#
#   MANIFEST_DIR=/data/project/private/dacon-corpus/manifests/strategy-v4 \
#   PROC=configs/processing_run2.yaml TRAIN=configs/train_run2.yaml MODEL_CFG=configs/c_run2.yaml \
#   INIT=/data/project/private/dacon-runs/run2-v4/T7/all_data_seed2/scored.pt \
#     sbatch --gres=gpu:2 --cpus-per-task=16 --mem=400G scripts/train_ddp.sbatch <run> <role> \
#       --batch-size 8 [train.py args]
#
#   --batch-size is PER RANK (train_run2.yaml says 16, which on 2 GPUs is a global 32): state it.
#   Submit from the repo root, and make sure runs/ exists there: Slurm opens the --output log
#   before this script can create it.
#
#   <role>: a fold number (e.g. 1)  -> --folds 1 --select ema      (one fold per job under DDP)
#           all:<seed> (e.g. all:0) -> --all-data --seed 0 --select ema
#
# #SBATCH lines do not expand variables: the GPU count is whatever --gres gave, read back from
# SLURM_GPUS_ON_NODE. The node exposes 64 usable CPUs (CPUEfctv; 96 threads): 8 per GPU.
# Main run: sbatch --gres=gpu:8 --cpus-per-task=64 --mem=1600G ...
# Output: $RUNS_ROOT/<run>/  (fold<k>/ or all_data_seed<s>/ below it, written by rank 0).

#SBATCH --job-name=eval-hyeonseop
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --output=runs/slurm-%x-%j.out
#SBATCH --error=runs/slurm-%x-%j.out

set -euo pipefail
: "${DACON_VENV:=/data/project/private/dacon-venvs/dacon311}"
: "${MANIFEST_DIR:?set MANIFEST_DIR (e.g. .../manifests/strategy-v4)}"
: "${PROC:=configs/processing_run2.yaml}"
: "${TRAIN:=configs/train_run2.yaml}"
: "${MODEL_CFG:=configs/c_run2.yaml}"
: "${WEIGHTS:=audio=/data/project/private/dacon-weights/beats,speech=/data/project/private/dacon-weights/xlsr-300m}"
: "${INIT:=-}"
: "${DRAWS:=32000}"
: "${RUNS_ROOT:=/data/project/private/dacon-runs}"
REPO="${DACON_REPO:-$SLURM_SUBMIT_DIR}"
RUN="${1:?usage: sbatch scripts/train_ddp.sbatch <run> <fold|all:seed> [train.py args]}"
ROLE="${2:?usage: sbatch scripts/train_ddp.sbatch <run> <fold|all:seed> [train.py args]}"
shift 2
NGPU="${SLURM_GPUS_ON_NODE:?not under an allocation with GPUs}"

if [[ "$ROLE" == all:* ]]; then role=(--all-data --seed "${ROLE#all:}" --select ema)
else role=(--folds "$ROLE" --select ema); fi
initarg=()
[ "$INIT" != "-" ] && initarg=(--init-weights "$INIT")

for p in "$DACON_VENV/bin/torchrun" "$REPO/scripts/train.py" "$MANIFEST_DIR/manifest.parquet" \
         "$MANIFEST_DIR/folds.parquet" "$REPO/$PROC" "$REPO/$TRAIN" "$REPO/$MODEL_CFG"; do
  [ -e "$p" ] || { echo "FATAL: missing $p"; exit 1; }
done
[ "$INIT" = "-" ] || [ -e "$INIT" ] || { echo "FATAL: missing init $INIT"; exit 1; }

OUT="$RUNS_ROOT/$RUN"
echo "node=$(hostname) job=${SLURM_JOB_ID:-?} gpus=$NGPU ($CUDA_VISIBLE_DEVICES) role=${role[*]}"
echo "processing=$PROC train=$TRAIN model=$MODEL_CFG init=$INIT out=$OUT draws=$DRAWS"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

cd "$REPO"
mkdir -p "$OUT" runs
[ -e "runs/$RUN" ] || ln -sfn "$OUT" "runs/$RUN"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# CUDA JIT cache on NODE-LOCAL disk (docs/training/07 §3: ~/.nv on EFS serialised concurrent jobs)
export CUDA_CACHE_PATH="/tmp/cuda-cache-${SLURM_JOB_ID:-x}" CUDA_CACHE_MAXSIZE=4294967296
mkdir -p "$CUDA_CACHE_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec "$DACON_VENV/bin/torchrun" --standalone --nproc_per_node="$NGPU" scripts/train.py \
  --manifest-dir "$MANIFEST_DIR" \
  --processing "$PROC" \
  --model "$MODEL_CFG" \
  --train "$TRAIN" \
  --weights "$WEIGHTS" \
  --stages joint \
  --out "$OUT" \
  --device cuda \
  --draws "$DRAWS" \
  --eval-n 6000 \
  "${role[@]}" "${initarg[@]}" "$@"
