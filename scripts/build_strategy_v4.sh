#!/bin/bash
# strategy-v4 = strategy-v3 + the round-2 corpora (docs/training/09 §3.3), in order:
# manifest -> folds -> cache -> audits. Stops at the first failure. strategy-v3 is the
# base and is never written. Rerunnable: a fresh --out each time (v4 is rebuilt whole).
#
#   bash scripts/build_strategy_v4.sh            # all round-2 corpora present
#   ONLY=proc,emilia-ko bash scripts/build_strategy_v4.sh   # a subset
set -euo pipefail
V=/data/project/private/dacon-venvs/dacon311/bin/python
M=/data/project/private/dacon-corpus/manifests
OUT=${OUT:-$M/strategy-v4}
CFG=configs/processing_run2.yaml
cd "$(dirname "$0")/.."
[ -e "$OUT/manifest.parquet" ] && { echo "FATAL: $OUT exists; move it aside first"; exit 1; }
# owner decision 2026-09-25: English and Korean only -> no zh-synth, no ctrsvdd
ONLY=${ONLY:-proc,emilia-ko,ko-synth2}
only=(--only "$ONLY")
echo "== manifest"; $V scripts/extend_manifest.py --v2 $M/strategy-v3 --out "$OUT" \
  --scheme strategy-v4 --workers 48 "${only[@]}" | tail -40
echo "== folds";    $V scripts/build_folds.py --processing-config $CFG --manifest-dir "$OUT" | tail -12
cat "$OUT/folds.caveats.txt" || true
echo "== cache";    $V scripts/build_cache.py --processing-config $CFG --manifest-dir "$OUT" --workers 48 | tail -12
echo "== audit fold 0"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --fold 0 --n 40000 | grep -E "FAIL|audit ok|drawn"
echo "== audit all-data"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --all-data --n 40000 | grep -E "FAIL|audit ok|drawn"
echo "strategy-v4 built at $OUT"
