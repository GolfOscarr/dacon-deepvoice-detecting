#!/bin/bash
# strategy-v6b = strategy-v6 + ACE-Step 1.5 instrumentals (pool D, `acestep-inst`), in order:
# config -> manifest -> drops -> folds -> cache -> balance -> audits. Stops at the first
# failure. strategy-v6 is the base and is never written. Rerunnable: a fresh --out each time.
#
#   bash scripts/build_strategy_v6b.sh            # DRY=1: the steps that do not need v6
#
# Inputs: $M/strategy-v6/{manifest,folds}.parquet (scripts/build_strategy_v6.sh) and
# interim/acestep-inst/metadata.csv (scripts/synth3/music_acestep_inst_gen.py finalize).
# The draw config is configs/processing_run6.yaml, unchanged (see CFG below).
set -euo pipefail
# nice + <= 8 workers: this runs on the training node
V="nice -n 19 /data/project/private/dacon-venvs/dacon311/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
M=/data/project/private/dacon-corpus/manifests
BASE=$M/strategy-v6
OUT=${OUT:-$M/strategy-v6b}
# run 6's draw as it is: a domain weight cannot steer music tiles (one bucket per side,
# tiles uniform over the rows that fit: processing.sampler bucket_keys / _tiles), so
# ACE-Step's share of fake-music tiles is its row count (3 x 10 s rows per clip, the
# Jamendo side's cut) and there is no acestep-inst entry to add
CFG=${CFG:-configs/processing_run6.yaml}
ONLY=acestep-inst
cd "$(dirname "$0")/.."

echo "== config"
[ -e "$CFG" ] || { echo "FATAL: $CFG missing"; exit 1; }
ROOT=${CORPUS_ROOT:-/data/project/private/dacon-corpus}
META=$ROOT/interim/acestep-inst/metadata.csv
[ -s "$META" ] || { echo "FATAL: $META missing (music_acestep_inst_gen.py finalize)"; exit 1; }

echo "== new rows (dry run on strategy-v5 when v6 is not there yet)"
if [ ! -e "$BASE/manifest.parquet" ] || [ ! -e "$BASE/folds.parquet" ]; then
  $V scripts/extend_manifest.py --v2 $M/strategy-v5 --out "$OUT/_dry" --scheme strategy-v6b \
    --corpus-root "$ROOT" --workers 8 --only "$ONLY" --dry-run | tail -20
  echo "STOP: $BASE is not built yet (manifest + folds); nothing written"; exit 0
fi
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: stop before writing"; exit 0; }
[ -e "$OUT/manifest.parquet" ] && { echo "FATAL: $OUT exists; move it aside first"; exit 1; }

echo "== manifest"; $V scripts/extend_manifest.py --v2 $BASE --out "$OUT/_extend" \
  --scheme strategy-v6b --corpus-root "$ROOT" --workers 8 --only "$ONLY" | tail -30
# the v5 rules again (a no-op on v6's rows; acestep-inst rows were screened upstream)
echo "== drops"; $V scripts/strategy/v5_filters.py --in-dir "$OUT/_extend" --out "$OUT" --base-dir $BASE
# PINNED to v6: every v6 row keeps (slice, fold); acestep15-inst goes to FAMILY_FOLD's fold
echo "== folds"; $V scripts/build_folds_pinned.py --base-dir $BASE --manifest-dir "$OUT" \
  --dropped "$OUT/dropped.parquet" | tail -30
cat "$OUT/folds.caveats.txt" || true
# resumable: existing cache16k entries are kept, only the new file_ids are decoded
echo "== cache"; $V scripts/build_cache.py --processing-config $CFG --manifest-dir "$OUT" --workers 8 | tail -12
# label symmetry (docs/training/11 §1.1): repeats, processed and language shares must match
echo "== balance all-data"; $V scripts/strategy/draw_balance.py --manifest-dir "$OUT" --processing $CFG --all-data --n 8000
echo "== audit fold 0"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --fold 0 --n 40000 | grep -E "FAIL|PASS|audit ok|drawn"
echo "== audit fold 1"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --fold 1 --n 40000 | grep -E "FAIL|PASS|audit ok|drawn"
echo "== audit all-data"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --all-data --n 40000 | grep -E "FAIL|PASS|audit ok|drawn"
echo "strategy-v6b built at $OUT"
