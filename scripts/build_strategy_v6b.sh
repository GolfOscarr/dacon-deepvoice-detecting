#!/bin/bash
# strategy-v6b = strategy-v6 + ACE-Step 1.5 instrumentals (pool D, `acestep-inst`), in order:
# config -> manifest -> drops -> folds -> cache -> balance -> audits. Stops at the first
# failure. strategy-v6 is the base and is never written. Rerunnable: a fresh --out each time.
#
#   bash scripts/build_strategy_v6b.sh            # DRY=1: the steps that do not need v6
#
# Inputs: $M/strategy-v6/{manifest,folds}.parquet (scripts/build_strategy_v6.sh) and
# interim/acestep-inst/metadata.csv (scripts/synth3/music_acestep_inst_gen.py finalize).
# configs/processing_run6b.yaml is derived from processing_run6.yaml here when missing
# (one domain_weights entry, a PROPOSAL for the lead) and must stay run6 + that entry.
set -euo pipefail
# nice + <= 8 workers: this runs on the training node
V="nice -n 19 /data/project/private/dacon-venvs/dacon311/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
M=/data/project/private/dacon-corpus/manifests
BASE=$M/strategy-v6
OUT=${OUT:-$M/strategy-v6b}
CFG6=${CFG6:-configs/processing_run6.yaml}
CFG=${CFG:-configs/processing_run6b.yaml}
ONLY=acestep-inst
cd "$(dirname "$0")/.."

echo "== config"
[ -e "$CFG6" ] || { echo "FATAL: $CFG6 missing"; exit 1; }
grep -q JAMENDO_WEIGHT "$CFG6" && { echo "FATAL: $CFG6 still has the JAMENDO_WEIGHT placeholder"; exit 1; }
if [ ! -e "$CFG" ]; then
  awk '
    NR == 1 {
      print "# [v6b] RUN 6b: processing_run6.yaml on strategy-v6b (= strategy-v6 + ACE-Step 1.5"
      print "# [v6b] instrumentals, scripts/build_strategy_v6b.sh). The only change is one"
      print "# [v6b] draw.domain_weights entry, \"acestep-inst|\" (see there). Everything else is run 6'"'"'s."
      print "# [v6b]"
    }
    { print }
    /^    "mtg-jamendo\/":/ {
      print "    # [v6b] PROPOSAL (lead to confirm): ACE-Step 1.5 instrumentals, pool D domain"
      print "    # [v6b] acestep-inst|acestep15-inst. Pool D is 10 generator domains (5 SONICS versions,"
      print "    # [v6b] 5 FakeMusicCaps models), each capped at 500 rows by the DOSS cap = 10 % of the"
      print "    # [v6b] fake-music draw each. x2.0 gives the newest family 2 of 11 shares (~18 %);"
      print "    # [v6b] x1.0 would be ~9 %, the share of one SONICS version."
      print "    \"acestep-inst|\": 2.0"
      n++
    }
    END { if (n != 1) exit 3 }' "$CFG6" > "$CFG.part" || { rm -f "$CFG.part"; echo "FATAL: no \"mtg-jamendo/\" line in $CFG6"; exit 1; }
  mv "$CFG.part" "$CFG"; echo "derived $CFG from $CFG6"
fi
# run6b must be run6 plus the [v6b] lines, nothing else (a stale copy is an error)
grep -v -e '^ *# \[v6b\]' -e '^    "acestep-inst|":' "$CFG" | diff - "$CFG6" >/dev/null \
  || { echo "FATAL: $CFG is not $CFG6 + the [v6b] lines; re-derive it (rm it and rerun)"; exit 1; }
$V -c "from processing.config import load_processing_config as l; c = l('$CFG'); \
print('domain_weights', [w for w in c.draw.domain_weights if w[0].startswith(('acestep', 'mtg'))])"

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
