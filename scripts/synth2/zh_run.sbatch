#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -J zh-synth
#SBATCH -o /data/project/private/dacon-corpus/interim/zh-synth/_logs/slurm-%x-%j.out
# S4 Mandarin fakes, one GPU for everything (track D is allowed one GPU job at a time):
#   1. Whisper large-v3 transcribes the prompt index (skips prompts already done);
#   2. smoke: SMOKE_N files per family into _smoke/, QC'd; a family with > 30 % rejects is dropped;
#   3. production: the surviving families run concurrently on the same GPU (SHARDS processes
#      each) with a QC follower,
#      each until TARGET_H kept hours (synth hours = TARGET_H / smoke keep rate, capped at 10 h)
#      or until DEADLINE (epoch seconds), whichever comes first. Everything is resumable.
# usage: sbatch -J zh-synth scripts/synth2/zh_run.sbatch [smoke|all]   (env: SMOKE_N, TARGET_H, SHARDS, DEADLINE)
set -u
MODE=${1:-all}
SMOKE_N=${SMOKE_N:-10}
TARGET_H=${TARGET_H:-7}
SHARDS=${SHARDS:-2}   # processes per family on the one GPU (each AR loop holds ~1 CPU core)
DEADLINE=${DEADLINE:-$(date -d '2026-09-26 04:40' +%s)}
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
S2=$REPO/scripts/synth2
OUT=/data/project/private/dacon-corpus/interim/zh-synth
V=/data/project/private/dacon-venvs
export CUDA_CACHE_PATH=/tmp/cuda-cache-$SLURM_JOB_ID
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
COSY_ENV="HF_HOME=/data/project/private/dacon-weights/synth/hf"
S2_ENV="HF_HOME=/data/project/private/dacon-weights/synth2/hf"
QC_PY="env $S2_ENV PYTHONPATH=$V/synth2-zhlib $V/synth2-qc/bin/python $S2/zh_qc.py"
COSY_PY="env $COSY_ENV $V/synth-cosyvoice/bin/python $S2/zh_synth_cosyvoice.py"
CB_PY="env $S2_ENV $V/synth2-chatterbox/bin/python $S2/zh_synth_chatterbox.py"
cd "$REPO"
echo "job $SLURM_JOB_ID host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES mode=$MODE deadline=$(date -d @$DEADLINE)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

$QC_PY transcribe || { echo "transcribe failed"; exit 1; }

# ---- smoke
SMK=$OUT/_smoke
for f in cosyvoice2 chatterbox-mtl; do
  if [ "$f" = cosyvoice2 ]; then PY=$COSY_PY; else PY=$CB_PY; fi
  have=$(( $(cat "$SMK/$f/synth.csv" 2>/dev/null | wc -l) - 1 ))
  if [ "$have" -ge "$SMOKE_N" ]; then echo "smoke $f: $have files already, not re-run"; continue; fi
  ZH_SYNTH_ROOT=$SMK $PY --limit "$SMOKE_N" > "$OUT/_logs/smoke-$f-$SLURM_JOB_ID.log" 2>&1
  echo "smoke $f exit=$?"; tail -n 3 "$OUT/_logs/smoke-$f-$SLURM_JOB_ID.log"
done
ZH_SYNTH_ROOT=$SMK $QC_PY qc --families cosyvoice2,chatterbox-mtl
FAMS=()
declare -A MAXH
for f in cosyvoice2 chatterbox-mtl; do
  read -r n k < <(python3 - "$SMK/$f/metadata.csv" <<'EOF'
import csv, sys, os
p = sys.argv[1]
rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
print(len(rows), sum(r["kept"] == "True" for r in rows))
EOF
)
  echo "smoke $f: $k/$n kept"
  if [ "$n" -ge 3 ] && [ $((k * 100)) -ge $((n * 70)) ]; then
    FAMS+=("$f")
    MAXH[$f]=$(python3 -c "print(round(min($TARGET_H * $n / $k, 10.0), 2))")
  else
    echo "DROP $f: smoke reject rate > 30 % or < 3 files"
  fi
done
[ "$MODE" = smoke ] && { echo "smoke only; done"; exit 0; }
[ ${#FAMS[@]} -eq 0 ] && { echo "no family passed smoke"; exit 1; }

# ---- production
LEFT=$(( DEADLINE - $(date +%s) ))
[ $LEFT -le 600 ] && { echo "deadline too close"; exit 1; }
PIDS=()
for f in "${FAMS[@]}"; do
  if [ "$f" = cosyvoice2 ]; then PY=$COSY_PY; else PY=$CB_PY; fi
  echo "production $f --max-hours ${MAXH[$f]} x $SHARDS shards for <= ${LEFT}s"
  for ((i = 0; i < SHARDS; i++)); do
    timeout --signal=TERM "$LEFT" $PY --n-files 6000 --max-hours "${MAXH[$f]}" --shard "$i/$SHARDS" \
      >> "$OUT/_logs/prod-$f-s$i.log" 2>&1 &
    PIDS+=($!)
  done
done
FAMCSV=$(IFS=,; echo "${FAMS[*]}")
PIDCSV=$(IFS=,; echo "${PIDS[*]}")
$QC_PY qc --families "$FAMCSV" --wait-pids "$PIDCSV" >> "$OUT/_logs/qc-$SLURM_JOB_ID.log" 2>&1
wait
$QC_PY finalize --families "$FAMCSV"
echo "job done $(date)"
