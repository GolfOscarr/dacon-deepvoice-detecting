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
# owner decision 2026-09-25: English and Korean only -> no zh-synth, no ctrsvdd.
# emilia-en + en-synth2: many-speaker English real and its clones (training/10 F1, F2)
ONLY=${ONLY:-proc,emilia-ko,ko-synth2,emilia-en,en-synth2}
only=(--only "$ONLY")
echo "== manifest"; $V scripts/extend_manifest.py --v2 $M/strategy-v3 --out "$OUT" \
  --scheme strategy-v4 --workers 48 "${only[@]}" | tail -40
# PINNED to v3 (docs/training/11 §1.2): build_folds.py would re-rotate every group and
# break fold-k init from run 1, PROBE, and like-for-like scoring on run 1's VAL specs.
echo "== folds";    $V scripts/build_folds_pinned.py --base-dir $M/strategy-v3 --manifest-dir "$OUT" | tail -12
cat "$OUT/folds.caveats.txt" || true
echo "== cache";    $V scripts/build_cache.py --processing-config $CFG --manifest-dir "$OUT" --workers 48 | tail -12
echo "== audit fold 0"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --fold 0 --n 40000 | grep -E "FAIL|audit ok|drawn"
echo "== audit fold 1"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --fold 1 --n 40000 | grep -E "FAIL|audit ok|drawn"
# label symmetry (docs/training/11 §1.1): repeats and processed shares must match across labels
echo "== balance all-data"; $V scripts/strategy/draw_balance.py --manifest-dir "$OUT" --processing $CFG --all-data --n 8000
echo "== audit all-data"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --all-data --n 40000 | grep -E "FAIL|audit ok|drawn"
echo "strategy-v4 built at $OUT"
