#!/bin/bash
# strategy-v5 = strategy-v4 + the round-3 corpora (docs/training/15), in order:
# manifest -> drops -> folds -> cache -> balance -> audits. Stops at the first failure.
# strategy-v4 is the base and is never written. Rerunnable: a fresh --out each time.
#
#   bash scripts/build_strategy_v5.sh
set -euo pipefail
V=/data/project/private/dacon-venvs/dacon311/bin/python
M=/data/project/private/dacon-corpus/manifests
OUT=${OUT:-$M/strategy-v5}
CFG=configs/processing_run5.yaml
cd "$(dirname "$0")/.."
[ -e "$OUT/manifest.parquet" ] && { echo "FATAL: $OUT exists; move it aside first"; exit 1; }
# English and Korean only (owner decision 2026-09-25). ko-synth3/rvc is in when its
# metadata.csv exists (>= 2 h kept); en-synth3 has all 5 families (sparktts/styletts2 keep
# only their files that pass the exact-zero screen; kept=False rows are not ingested).
# emilia-ko is in the base: --append adds only its new (YODAS-2) file_ids.
ONLY=ko-synth3,en-synth3,ko-synth2-extra,en-synth2-extra,sing-fake,sing-real,aisong-kr-en,fma-medium-songs,emilia-ko
echo "== manifest"; $V scripts/extend_manifest.py --v2 $M/strategy-v4 --out "$OUT/_extend" \
  --scheme strategy-v5 --workers 64 --only "$ONLY" --append emilia-ko | tail -30
# zero-run cue (v4 rows too), vocal-carrying music components (pools C and D), and
# singing outside ko/en or fake-only Korean singing: scripts/strategy/v5_filters.py
echo "== drops"; $V scripts/strategy/v5_filters.py --in-dir "$OUT/_extend" --out "$OUT" --base-dir $M/strategy-v4
# PINNED to v4 (docs/training/11 §1.2); the dropped base rows are the only ones allowed missing
echo "== folds"; $V scripts/build_folds_pinned.py --base-dir $M/strategy-v4 --manifest-dir "$OUT" \
  --dropped "$OUT/dropped.parquet" | tail -30
cat "$OUT/folds.caveats.txt" || true
echo "== cache"; $V scripts/build_cache.py --processing-config $CFG --manifest-dir "$OUT" --workers 64 | tail -12
# label symmetry (docs/training/11 §1.1): repeats, processed and language shares must match
echo "== balance all-data"; $V scripts/strategy/draw_balance.py --manifest-dir "$OUT" --processing $CFG --all-data --n 8000
echo "== audit fold 0"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --fold 0 --n 40000 | grep -E "FAIL|PASS|audit ok|drawn"
echo "== audit fold 1"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --fold 1 --n 40000 | grep -E "FAIL|PASS|audit ok|drawn"
echo "== audit all-data"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --all-data --n 40000 | grep -E "FAIL|PASS|audit ok|drawn"
echo "strategy-v5 built at $OUT"
