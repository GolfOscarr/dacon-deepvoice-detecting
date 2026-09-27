#!/usr/bin/env bash
#SBATCH -p debug
#SBATCH -J eval-hyeonseop
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --array=0-7
#SBATCH --time=01:30:00
#SBATCH -o /data/project/private/dacon-corpus/interim/_logs/music-acestep-inst-%A_%a.out
# Pool D: ACE-Step 1.5 instrumentals matched to MTG-Jamendo (scripts/synth3/music_acestep_inst_gen.py).
# One shard per array task (1 GPU each). The plan must exist first (`plan`, CPU).
#   sbatch scripts/synth3/music_acestep_inst.sbatch [--stop-at 2026-09-28T13:20]
# Extra args go to `gen`; the last --stop-at wins. Rerunnable: logged clips are skipped.
set -euo pipefail
OUT=${OUT:-/data/project/private/dacon-corpus/interim/acestep-inst}
[ -s "$OUT/plan.jsonl" ] || { echo "FATAL: $OUT/plan.jsonl missing (run the plan step)"; exit 1; }
export CUDA_CACHE_PATH=/tmp/cuda-cache-${SLURM_JOB_ID}
export ACESTEP_CHECKPOINTS_DIR=/data/project/private/dacon-weights/sing/acestep15/checkpoints
export HF_HOME=/data/project/private/dacon-weights/sing/hf-home
export OMP_NUM_THREADS=4
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
/data/project/private/dacon-venvs/sing-acestep15/bin/python $REPO/scripts/synth3/music_acestep_inst_gen.py gen \
    --out "$OUT" --shard "${SLURM_ARRAY_TASK_ID}" --nshards "${NSHARDS:-8}" --stop-at 2026-09-28T13:20 "$@"
