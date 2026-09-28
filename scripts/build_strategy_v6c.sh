#!/bin/bash
# strategy-v6c = strategy-v6b + more of the same music (owner decision D1, 2026-09-28):
#   mtg-jamendo-2   a second MTG-Jamendo instrumental batch (pool C), jamendo_prepare.py
#                   with --out interim/mtg-jamendo-2 --prior <v6 selection> --per-artist 8
#                   --per-album 3 (v6's selected tracks count toward the caps; an artist
#                   already in v6 joins its atom, so its fold is inherited, never moved)
#   acestep-inst-2  a second ACE-Step 1.5 instrumental batch (pool D, family acestep15-inst:
#                   fold 2 with the rest of ACE-Step), new seeds, captions from both batches' tags
# Both are already 10 s pieces (the one cut rule, pieces_of), so there is no cut step here.
# In order: config -> manifest -> drops -> folds -> cache -> music filters -> cache ->
# balance -> music tile shares (v6b and v6c) -> audits. Stops at the first failure.
# strategy-v6b is the base and is never written. Rerunnable: a fresh --out each time.
#
#   bash scripts/build_strategy_v6c.sh        # DRY=1: stop after the dry-run manifest step
set -euo pipefail
V="nice -n 19 /data/project/private/dacon-venvs/dacon311/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=${WORKERS:-16}
M=/data/project/private/dacon-corpus/manifests
BASE=$M/strategy-v6b
OUT=${OUT:-$M/strategy-v6c}
# run 6's draw as it is (see build_strategy_v6b.sh: a music corpus's tile share is its row count)
CFG=${CFG:-configs/processing_run6.yaml}
ONLY=mtg-jamendo-2,acestep-inst-2
cd "$(dirname "$0")/.."

echo "== config"
[ -e "$CFG" ] || { echo "FATAL: $CFG missing"; exit 1; }
ROOT=${CORPUS_ROOT:-/data/project/private/dacon-corpus}
for d in mtg-jamendo-2 acestep-inst-2; do
  [ -s "$ROOT/interim/$d/metadata.csv" ] || { echo "FATAL: $ROOT/interim/$d/metadata.csv missing"; exit 1; }
done
[ -e "$BASE/manifest.parquet" ] && [ -e "$BASE/folds.parquet" ] || { echo "FATAL: $BASE not built"; exit 1; }
if [ "${DRY:-0}" = 1 ]; then
  $V scripts/extend_manifest.py --v2 $BASE --out "$OUT/_dry" --scheme strategy-v6c \
    --corpus-root "$ROOT" --workers $W --only "$ONLY" --dry-run | tail -20
  echo "DRY=1: stop before writing"; exit 0
fi
[ -e "$OUT/manifest.parquet" ] && { echo "FATAL: $OUT exists; move it aside first"; exit 1; }

echo "== manifest"; $V scripts/extend_manifest.py --v2 $BASE --out "$OUT/_extend" \
  --scheme strategy-v6c --corpus-root "$ROOT" --workers $W --only "$ONLY" | tail -30
# the v5 rules again (a no-op on v6b's rows; both batches were screened upstream)
echo "== drops"; $V scripts/strategy/v5_filters.py --in-dir "$OUT/_extend" --out "$OUT/_pre" --base-dir $BASE
# PINNED to v6b: every v6b row keeps (slice, fold); a v6 Jamendo artist's new tracks join its
# atom; new artists are water-filled; acestep15-inst is a base family (fold 2)
echo "== folds"; $V scripts/build_folds_pinned.py --base-dir $BASE --manifest-dir "$OUT/_pre" \
  --dropped "$OUT/_pre/dropped.parquet" | tail -30
cat "$OUT/_pre/folds.caveats.txt" || true
# the music filters read cache16k: decode the new rows first (existing entries are kept)
echo "== cache (new rows)"; $V scripts/build_cache.py --processing-config $CFG --manifest-dir "$OUT/_pre" --workers $W | tail -12
# the exact-zero rule on EVERY music piece (both sides; v6b's ACE-Step rows get the cache-side
# check here too). --acestep-rows 0: the ACE-Step rows are in the manifest now, so the
# sonics-sep target is above the rows v6b kept and no further sonics row is dropped
echo "== music filters"; $V scripts/strategy/v6_music_filters.py --in-dir "$OUT/_pre" --out "$OUT" \
  --acestep-rows 0 --workers $W | tail -60
echo "== cache"; $V scripts/build_cache.py --processing-config $CFG --manifest-dir "$OUT" --workers $W | tail -12
# label symmetry (docs/training/11 §1.1): repeats, processed and language shares must match
echo "== balance all-data"; $V scripts/strategy/draw_balance.py --manifest-dir "$OUT" --processing $CFG --all-data --n 8000 \
  | tee "$OUT/_balance_alldata.log"
# music tile shares per corpus per side, before (v6b) and after (v6c)
echo "== music tile shares"
$V scripts/strategy/music_balance.py --manifest-dir $BASE --processing $CFG --all-data --n 8000 > "$OUT/_music_balance_v6b.json"
$V scripts/strategy/music_balance.py --manifest-dir "$OUT" --processing $CFG --all-data --n 8000 > "$OUT/_music_balance_v6c.json"
cat "$OUT/_music_balance_v6b.json" "$OUT/_music_balance_v6c.json"
# n=80000 (build_strategy_v6b.sh: I3 is draw-budget bound once music rows are 10 s pieces)
for k in 0 1 all; do
  if [ $k = all ]; then sel="--all-data"; else sel="--fold $k"; fi
  echo "== audit $k"
  $V scripts/strategy/audit_fold.py --manifest-dir "$OUT" --processing $CFG $sel --n 80000 > "$OUT/_audit_${k}_n80000.log"
  grep -E "FAIL|PASS|audit ok|drawn" "$OUT/_audit_${k}_n80000.log"
done
echo "strategy-v6c built at $OUT"
