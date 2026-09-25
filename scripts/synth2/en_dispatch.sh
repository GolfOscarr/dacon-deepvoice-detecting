#!/bin/bash
# English round-2 dispatcher (en-synth2): from START_AT, keep this USER's GPU jobs (all of
# them, R + PD, any track) <= MAXJ; submit the next line of _dispatch/queue.txt (ko_run.sbatch
# args) with SYNTH2_LANG=en whenever there is room. Stops at STOP_AT or when the queue is empty.
# A new script, not ko_dispatch.sh: that one may still be running, and bash reads a running
# script from disk as it goes.
D=/data/project/private/dacon-corpus/interim/en-synth2/_dispatch
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
MAXJ=${MAXJ:-8}
START_AT=${START_AT:-1790365800}   # 2026-09-26 04:50 KST, when the Korean shards stop
STOP_AT=${STOP_AT:-1790388000}     # 11:00 KST: nothing new starts that late
mkdir -p "$D"
while [ "$(date +%s)" -lt "$START_AT" ]; do sleep 30; done
while [ "$(date +%s)" -lt "$STOP_AT" ] && [ -s "$D/queue.txt" ]; do
  n=$(squeue -u "$USER" -h -o "%b" | grep -c gpu)
  if [ "$n" -lt "$MAXJ" ]; then
    line=$(head -n 1 "$D/queue.txt")
    sed -i 1d "$D/queue.txt"
    out=$(cd "$REPO" && SYNTH2_LANG=en sbatch \
      -o /data/project/private/dacon-corpus/interim/en-synth2/_logs/slurm-%j.out \
      scripts/synth2/ko_run.sbatch $line)
    jid=$(echo "$out" | awk '{print $4}')
    [ -n "$jid" ] && echo "$jid" >> "$D/myjobs.txt"
    echo "$(date +%T) user_gpu_jobs=$n submitted $jid: $line"
    sleep 20
    continue
  fi
  sleep 30
done
echo "dispatcher stopped $(date) queue_left=$(wc -l < "$D/queue.txt")"
