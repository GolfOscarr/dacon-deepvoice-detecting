#!/bin/bash
# en-synth3 dispatcher: keep <= MAXJ of THIS track's GPU jobs (ids in _dispatch/myjobs.txt; other agents
# run under the same Slurm user) queued or running; submit the next line of _dispatch/queue.txt whenever
# there is room. A line is `[VAR=value ...] <family> <en_synth args>`; the leading VAR=value words are
# exported to that sbatch only (N5: EN3_ROOT, EN3_EXCLUDE_SPEAKERS). Every job gets --deadline DEADLINE.
# Stops at STOP_AT or when the queue is empty. Edit queue.txt freely while it runs.
D=/data/project/private/dacon-corpus/interim/en-synth3/_dispatch
REPO=/home/hyeonseop.shin/workspace/dacon-deepvoice-detecting
MAXJ=${MAXJ:-2}
DEADLINE=$(date -d '2026-09-27 12:50' +%s)   # synth scripts stop starting new files
STOP_AT=$(date -d '2026-09-27 12:15' +%s)    # nothing new is submitted after this
touch "$D/myjobs.txt"
while [ "$(date +%s)" -lt "$STOP_AT" ]; do
  ids=$(paste -sd, "$D/myjobs.txt")
  n=$( [ -n "$ids" ] && squeue -h -j "$ids" -o %i 2>/dev/null | wc -l || echo 0)
  if [ "$n" -lt "$MAXJ" ] && [ -s "$D/queue.txt" ]; then
    line=$(head -n 1 "$D/queue.txt")
    sed -i 1d "$D/queue.txt"
    envs=(); set -- $line
    while [[ "${1:-}" == *=* ]]; do envs+=("$1"); shift; done
    out=$(cd "$REPO" && env "${envs[@]}" sbatch scripts/synth3/en_run.sbatch "$@" --deadline "$DEADLINE")
    jid=$(echo "$out" | awk '{print $4}')
    [ -n "$jid" ] && echo "$jid" >> "$D/myjobs.txt"
    echo "$(date +%T) active=$n submitted $jid: $line"
    sleep 5
    continue
  fi
  sleep 30
done
echo "dispatcher stopped $(date) queue_left=$(wc -l < "$D/queue.txt")"
