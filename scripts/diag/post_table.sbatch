#!/bin/bash
# Post-run scoring for any run launched from a TASK_TABLE (scripts/train_run2.sbatch):
# task i's fold models (fold rows only) are scored on run 1's EXACT v3 VAL specs
# (v3 manifest, processing_first_run.yaml, eval-n 6000, seed 1234) -> <run>/T<i>/v3val/fold<k>/,
# then `scripts/diag/run_breakdown.py` on it.  All-data rows are skipped (no VAL).
#
#   sbatch --array=0-<n-1> scripts/diag/post_table.sbatch <run dir> <task table>
#SBATCH -J eval-hyeonseop
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 03:00:00
#SBATCH -o /data/project/private/dacon-runs/_logs/posttab-%A_%a.out
set -euo pipefail
RUN=${1:?run dir}; TABLE=${2:?task table}
TASK=${SLURM_ARRAY_TASK_ID:?}
cd "${SLURM_SUBMIT_DIR}"
mapfile -t ROWS < <(grep -v -E '^\s*(#|$)' "$TABLE")
IFS='|' read -r who man proc train init model <<< "${ROWS[$TASK]}"
who=$(echo "$who" | xargs); model=$(echo "${model:-}" | xargs)
[[ "$who" == all:* ]] && { echo "T$TASK is all-data: no VAL, skipped"; exit 0; }
MODEL=${model:-configs/c_run2.yaml}
CK="$RUN/T$TASK/fold$who/scored.pt"
[ -e "$CK" ] || { echo "FATAL: missing $CK"; exit 1; }
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID OMP_NUM_THREADS=1
mkdir -p "$CUDA_CACHE_PATH"
V=${V:-/data/project/private/dacon-venvs/dacon311/bin/python}   # V=echo for a dry test
W=audio=/data/project/private/dacon-weights/beats,speech=/data/project/private/dacon-weights/xlsr-300m
$V scripts/eval_checkpoint.py --ckpt "$CK" --fold "$who" \
  --manifest-dir /data/project/private/dacon-corpus/manifests/strategy-v3 \
  --processing configs/processing_first_run.yaml --model "$MODEL" --weights $W \
  --eval-n 6000 --eval-seed 1234 --gpu-fraction 0.95 --out-dir "$RUN/T$TASK/v3val"
$V scripts/diag/run_breakdown.py --run "$RUN/T$TASK/v3val" \
  --manifest /data/project/private/dacon-corpus/manifests/strategy-v3/manifest.parquet \
  --out "$RUN/T$TASK/v3val/diag" | head -8
