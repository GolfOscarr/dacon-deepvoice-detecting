#!/bin/bash
# Run 2 (docs/training/11 §4): eight tasks, one GPU each, on the debug node.
#
#   MANIFEST_DIR=/data/project/private/dacon-corpus/manifests/strategy-v4 \
#     sbatch scripts/train_run2.sbatch <run name> [extra train.py args...]
#
# EVERY task writes to its own root <RUNS_ROOT>/<run>/T<i>/: train.py writes <out>/fold<k>/,
# and T0/T5 (fold 0) and T1/T4 (fold 1) would otherwise overwrite each other (review B1).
# Edit the TASKS table below to change the plan; a line is
#   <folds-or-all:seed> | <manifest: v4|v3> | <processing cfg> | <train cfg> | <init scored.pt or ->

#SBATCH --job-name=eval-hyeonseop
#SBATCH --partition=debug
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --output=runs/slurm-%x-%A_%a.out
#SBATCH --error=runs/slurm-%x-%A_%a.out

set -euo pipefail
: "${DACON_VENV:=/data/project/private/dacon-venvs/dacon311}"
: "${MANIFEST_DIR:?set MANIFEST_DIR to strategy-v4}"
: "${V3_DIR:=/data/project/private/dacon-corpus/manifests/strategy-v3}"
: "${MODEL_CFG:=configs/c_run2.yaml}"
: "${WEIGHTS:=audio=/data/project/private/dacon-weights/beats,speech=/data/project/private/dacon-weights/xlsr-300m}"
: "${R1:=/data/project/private/dacon-runs/first-v3}"
: "${DRAWS:=32000}"
REPO="${DACON_REPO:-$SLURM_SUBMIT_DIR}"
RUN="${1:?usage: sbatch scripts/train_run2.sbatch <run name> [train.py args]}"
shift
TASK="${SLURM_ARRAY_TASK_ID:?run under sbatch --array}"

P2=configs/processing_run2.yaml
PC=configs/processing_run2_v3ctl.yaml
FT=configs/train_run2.yaml
SC=configs/train_run2_scratch.yaml
TASKS=(
  "0     | v4 | $P2 | $FT | $R1/fold0/scored.pt"            # T0 F1 on LJ; fishspeech unseen
  "1     | v4 | $P2 | $FT | $R1/fold1/scored.pt"            # T1 F2 XTTS; maskgct unseen
  "2     | v4 | $P2 | $FT | $R1/fold2/scored.pt"            # T2 seedvc unseen
  "3     | v4 | $P2 | $FT | $R1/fold3/scored.pt"            # T3 cosyvoice+chatterbox unseen
  "1     | v4 | $P2 | $SC | -"                             # T4 scratch control (fold 1)
  "0     | v3 | $PC | $FT | $R1/fold0/scored.pt"            # T5 data control (fold 0, v3)
  "all:0 | v4 | $P2 | $FT | $R1/all_data_seed0/scored.pt"   # T6 submission candidate
  "all:2 | v4 | $P2 | $FT | $R1/all_data_seed2/scored.pt"   # T7 second candidate
)
# TASK_TABLE=<file>: one task line per row (same format; '#' comments and blank lines skipped)
if [ -n "${TASK_TABLE:-}" ]; then
  mapfile -t TASKS < <(grep -v -E '^\s*(#|$)' "$TASK_TABLE")
fi
[ "$TASK" -lt "${#TASKS[@]}" ] || { echo "FATAL: task $TASK not in table (${#TASKS[@]} rows)"; exit 1; }
IFS='|' read -r who man proc train init <<< "${TASKS[$TASK]}"
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; echo "${s%"${s##*[![:space:]]}"}"; }
who=$(trim "$who"); man=$(trim "$man"); proc=$(trim "$proc"); train=$(trim "$train")
init=$(trim "${init%%#*}")
case "$man" in v4) MDIR="$MANIFEST_DIR" ;; v3) MDIR="$V3_DIR" ;; *) echo "FATAL: manifest $man"; exit 1 ;; esac
if [[ "$who" == all:* ]]; then role=(--all-data --seed "${who#all:}" --select ema)
else role=(--folds "$who" --select ema); fi
initarg=()
[ "$init" != "-" ] && initarg=(--init-weights "$init")

for p in "$DACON_VENV/bin/python" "$REPO/scripts/train.py" "$MDIR/manifest.parquet" \
         "$MDIR/folds.parquet" "$REPO/$proc" "$REPO/$train" "$REPO/$MODEL_CFG"; do
  [ -e "$p" ] || { echo "FATAL: missing $p"; exit 1; }
done
[ "$init" = "-" ] || [ -e "$init" ] || { echo "FATAL: missing init $init"; exit 1; }

: "${RUNS_ROOT:=/data/project/private/dacon-runs}"
OUT="$RUNS_ROOT/$RUN/T$TASK"
echo "node=$(hostname) job=${SLURM_ARRAY_JOB_ID:-?}_${TASK} gpu=${CUDA_VISIBLE_DEVICES:-?}"
echo "T$TASK role=${role[*]} manifest=$MDIR processing=$proc train=$train init=${init} out=$OUT"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

cd "$REPO"
mkdir -p "$OUT" runs
[ -e "runs/$RUN" ] || ln -sfn "$RUNS_ROOT/$RUN" "runs/$RUN"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# CUDA JIT cache on NODE-LOCAL disk (docs/training/07 §3: ~/.nv on EFS serialised 8 tasks)
export CUDA_CACHE_PATH="/tmp/cuda-cache-${SLURM_ARRAY_JOB_ID:-x}-${TASK}" CUDA_CACHE_MAXSIZE=4294967296
mkdir -p "$CUDA_CACHE_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$DACON_VENV/bin/python" scripts/train.py \
  --manifest-dir "$MDIR" \
  --processing "$proc" \
  --model "$MODEL_CFG" \
  --train "$train" \
  --weights "$WEIGHTS" \
  --stages joint \
  --out "$OUT" \
  --device cuda \
  --draws "$DRAWS" \
  --eval-n 6000 \
  "${role[@]}" "${initarg[@]}" "$@"
