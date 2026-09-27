#!/bin/bash
# Keep <= MAXJ of this track's GPU jobs (ids in _dispatch/myjobs.txt) queued/running; submit the
# next line of _dispatch/queue.txt (scripts/synth3/ko_run.sbatch args) whenever there is room.
# Stops submitting at STOP_AT; every queue line carries --deadline (13:00 KST Sunday) so the
# jobs themselves stop starting new files then.
D=/data/project/private/dacon-corpus/interim/ko-synth3/_dispatch
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
MAXJ=${MAXJ:-3}
STOP_AT=$(date -d '2026-09-27 12:40' +%s)
touch "$D/myjobs.txt"
while [ "$(date +%s)" -lt "$STOP_AT" ]; do
  [ -f "$D/MAXJ" ] && MAXJ=$(cat "$D/MAXJ")
  ids=$(paste -sd, "$D/myjobs.txt")
  n=0; [ -n "$ids" ] && n=$(squeue -h -j "$ids" -o %i 2>/dev/null | wc -l)
  if [ "$n" -lt "$MAXJ" ] && [ -s "$D/queue.txt" ]; then
    line=$(head -n 1 "$D/queue.txt")
    sed -i 1d "$D/queue.txt"
    out=$(cd "$REPO" && sbatch scripts/synth3/ko_run.sbatch $line)
    jid=$(echo "$out" | awk '{print $4}')
    [ -n "$jid" ] && echo "$jid" >> "$D/myjobs.txt"
    echo "$(date +%T) active=$n submitted $jid: $line"
    sleep 2
    continue
  fi
  sleep 30
done
echo "dispatcher stopped $(date)"
