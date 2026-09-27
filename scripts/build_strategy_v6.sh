#!/bin/bash
# strategy-v6 = strategy-v5 + MTG-Jamendo real music, in order:
# manifest -> drops -> folds -> music cut -> cache -> balance -> audits. Stops at the first
# failure.
# strategy-v5 is the base and is never written. Rerunnable: a fresh --out each time.
#
#   bash scripts/build_strategy_v6.sh
#
# Input: interim/mtg-jamendo/metadata.csv from
# scripts/data_extra/jamendo_prepare.py (select -> extract -> htdemucs calibration subset ->
# tag -> finalize): instrumentals -> pool C components (`mtg-jamendo`). Songs with
# vocals are left out: whole-file cell-5 rows are never drawn under f8 = 1.
# Then every long pool C / D music row (both sides, one rule) is cut into 10 s pieces
# (scripts/strategy/cut_music_rows.py): the music draw tiles its one bucket per side
# uniformly over rows, so a corpus's tile share is its row count; after the cut it
# follows hours. Folds are pinned to v5 BEFORE the cut; pieces copy their parent's row.
set -euo pipefail
# nice + <= 16 workers: this runs on the training node (lead, 2026-09-28)
V="nice -n 19 /data/project/private/dacon-venvs/dacon311/bin/python"
M=/data/project/private/dacon-corpus/manifests
OUT=${OUT:-$M/strategy-v6}
CFG=configs/processing_run6.yaml
cd "$(dirname "$0")/.."
[ -e "$OUT/manifest.parquet" ] && { echo "FATAL: $OUT exists; move it aside first"; exit 1; }
ONLY=mtg-jamendo
echo "== manifest"; $V scripts/extend_manifest.py --v2 $M/strategy-v5 --out "$OUT/_extend" \
  --scheme strategy-v6 --workers 16 --only "$ONLY" | tail -30
# the v5 rules again (a no-op on v5's own rows; Jamendo rows were screened upstream)
echo "== drops"; $V scripts/strategy/v5_filters.py --in-dir "$OUT/_extend" --out "$OUT/_pre" --base-dir $M/strategy-v5
# PINNED to v5: every v5 row keeps (slice, fold); new artists become new atoms
echo "== folds"; $V scripts/build_folds_pinned.py --base-dir $M/strategy-v5 --manifest-dir "$OUT/_pre" \
  --dropped "$OUT/_pre/dropped.parquet" | tail -30
cat "$OUT/_pre/folds.caveats.txt" || true
# pieces from cache16k (the parents' cached audio) into interim/music-cut10; VG1 re-checked
echo "== music cut"; $V scripts/strategy/cut_music_rows.py --in-dir "$OUT/_pre" --out "$OUT" --workers 16 | tail -40
# resumable: existing cache16k entries are kept, only the new file_ids are decoded
echo "== cache"; $V scripts/build_cache.py --processing-config $CFG --manifest-dir "$OUT" --workers 16 | tail -12
# label symmetry (docs/training/11 §1.1): repeats, processed and language shares must match
echo "== balance all-data"; $V scripts/strategy/draw_balance.py --manifest-dir "$OUT" --processing $CFG --all-data --n 8000
echo "== audit fold 0"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --fold 0 --n 40000 | grep -E "FAIL|PASS|audit ok|drawn"
echo "== audit fold 1"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --fold 1 --n 40000 | grep -E "FAIL|PASS|audit ok|drawn"
echo "== audit all-data"; $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG --all-data --n 40000 | grep -E "FAIL|PASS|audit ok|drawn"
echo "strategy-v6 built at $OUT"
