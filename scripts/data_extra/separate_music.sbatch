#!/usr/bin/env bash
#SBATCH -p debug
#SBATCH -J music-sep
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --time=06:00:00
#SBATCH -o /data/project/private/dacon-corpus/interim/_logs/music-sep-%j.out
# D-b: htdemucs no_vocals stems for the selection, 2 GPUs x 1 shard each.
# Usage: sbatch scripts/data_extra/separate_music.sbatch   (rerunnable: skips existing outputs)
set -euo pipefail
REPO=${REPO:-/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting}
WORK=${WORK:-/data/project/private/dacon-corpus/interim/_music_sep_work}
PY=${PY:-/data/project/private/dacon-venvs/dacon311/bin/python}
export TORCH_HOME=/data/project/private/dacon-weights/demucs
export OMP_NUM_THREADS=4
cd "$REPO"
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$s "$PY" scripts/data_extra/separate_music.py \
      --work "$WORK" --shard $s --nshards 2 --batch 8 --loaders 6 \
      > "$WORK/sep_shard$s.log" 2>&1 &
done
wait
echo "both shards finished"
