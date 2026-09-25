#!/bin/bash
# Keep <= MAXJ of this track's GPU jobs (ids in _dispatch/myjobs.txt) queued/running; submit the
# next line of _dispatch/queue.txt (ko_run.sbatch args) whenever there is room. Stops at STOP_AT.
D=/data/project/private/dacon-corpus/interim/ko-synth2/_dispatch
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
MAXJ=${MAXJ:-6}
STOP_AT=$(date -d '2026-09-26 04:30' +%s)
while [ "$(date +%s)" -lt "$STOP_AT" ]; do
  ids=$(paste -sd, "$D/myjobs.txt")
  n=$(squeue -h -j "$ids" -o %i 2>/dev/null | wc -l)
  if [ "$n" -lt "$MAXJ" ] && [ -s "$D/queue.txt" ]; then
    line=$(head -n 1 "$D/queue.txt")
    sed -i 1d "$D/queue.txt"
    out=$(cd "$REPO" && sbatch scripts/synth2/ko_run.sbatch $line)
    jid=$(echo "$out" | awk '{print $4}')
    [ -n "$jid" ] && echo "$jid" >> "$D/myjobs.txt"
    echo "$(date +%T) active=$n submitted $jid: $line"
    continue
  fi
  sleep 30
done
echo "dispatcher stopped $(date)"
