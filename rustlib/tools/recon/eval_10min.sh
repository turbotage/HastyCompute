#!/usr/bin/env bash
# The realistic-scan evaluation chain (factory_arms/ph0001, 10-min flow scan + 2.5-min B0 scan):
#   rustlib/tools/recon/eval_10min.sh [<phantom_dir>] [--iters 20] [--mem 60] [--resp-bins 2] [--until smoke|all]
# 1. waits until tools/phantom/realistic_scan.sh is finished (B0 scan simulated);
# 2. recon_b0 on the B0 scan (ω / R2* / fat maps)                                   → <ph>/recon_b0_2p5min
# 3. timing smoke: 2 iterations of the oracle reconstruction (data store, packed tables) → <ph>/eval_10min/smoke, and
#    the same with respiratory bins                                                        → <ph>/eval_10min/smoke_resp
# 4. eval_matrix: oracle, est_field, est_smaps, est_both                                → <ph>/eval_10min
# 5. segment basis: oracle with --seg-basis svdw at one segment fewer than the fixed basis → <ph>/eval_10min_svdw
# Every step is resumable (DONE markers, recon checkpoints); a STOP file in <ph>/eval_10min (or Ctrl-C) ends it.
# GPU pacing: HASTY_GPU_WATTS (default 350 here). Memory: each step under its own cap, no swap.
set -uo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
PH=$(realpath "${1:-data/hasty/phantom/factory_arms/ph0001}"); [[ $# -gt 0 && $1 != --* ]] && shift
ITERS=20; MEM=60; NR=2; UNTIL=all
while [[ $# -gt 0 ]]; do
  case $1 in
    --iters) ITERS=$2; shift 2 ;;
    --mem) MEM=$2; shift 2 ;;
    --resp-bins) NR=$2; shift 2 ;;
    --until) UNTIL=$2; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
export HASTY_GPU_WATTS=${HASTY_GPU_WATTS:-350} HASTY_GPU_WATTS_WINDOW=${HASTY_GPU_WATTS_WINDOW:-3.5}
# k-space stores are caches: on the root disk inside the repo (the data disk filled up once with two 44 GiB stores)
export HASTY_KSTORE_DIR=${HASTY_KSTORE_DIR:-$R/rustlib/target/kstore}
B=rustlib/target/release
SF=$PH/sim_flow_10min/scan
SB=$PH/sim_b0_2.5min/scan
[[ -d $SB ]] || SB=$(ls -d "$PH"/sim_b0_*/scan 2>/dev/null | head -1)
OUT=$PH/eval_10min
mkdir -p "$OUT"
rm -f "$OUT/STOP"
stop() { [[ -e $OUT/STOP ]] && { echo "== STOP file: ending"; exit 0; }; }
cap() { local g=$1; shift; systemd-run --user --scope -q -p MemoryMax=${g}G -p MemorySwapMax=0 "$@"; }
# common reconstruction arguments: full physics with GNL, streamed data in one shared disk store, packed tables
BASE=(--gnl --phys full --kstore disk:zstd1 --table-store zstd)
RESP=(--resp-bins "$NR" --resp bellows)
# warm start: PCGLS per encode on all data (from zero, 10–20 P²nP iterations stay unconverged on this scan)
RECON=("${BASE[@]}" "${RESP[@]}" --iters "$ITERS" --warm 20)

echo "== waiting for the phantom pipeline: $(date +%H:%M:%S)"
while pgrep -f "tools/phantom/realistic_scan.sh" >/dev/null; do stop; sleep 30; done
[[ -n $SB && -f $SB/data_clean.npy && -f $SF/data_clean.npy ]] || { echo "!! scans missing (flow $SF, B0 $SB)"; exit 1; }

# ── 2. B0 / R2* / fat maps from the B0 scan ───────────────────────────────────────────────────────────────────────
B0R=$PH/recon_b0_2p5min
if [[ ! -f $B0R/DONE ]]; then
  stop
  echo "== recon_b0: $(date +%H:%M:%S)"
  mkdir -p "$B0R"
  cap "$MEM" $B/recon_b0 "$SB" "$B0R" 2>&1 | tee -a "$B0R/log.txt" | grep -E "RMSE|ω|R2\*|fat|support|done|error|panicked" | tail -12
  [[ ${PIPESTATUS[0]} -eq 0 ]] && touch "$B0R/DONE" || { echo "!! recon_b0 failed: $B0R/log.txt"; exit 1; }
fi

# ── 3. timing smoke ───────────────────────────────────────────────────────────────────────────────────────────────
SM=$OUT/smoke
if [[ ! -f $SM/DONE ]]; then
  stop
  echo "== smoke (2 iterations, oracle): $(date +%H:%M:%S)"
  mkdir -p "$SM"
  /usr/bin/time -v -o "$SM/time.txt" systemd-run --user --scope -q -p MemoryMax=${MEM}G -p MemorySwapMax=0 \
    $B/recon_p2np "$SF" "${BASE[@]}" --iters 2 --fresh --out "$SM" 2>&1 | tee "$SM/log.txt" | grep -E "segments|SegNufft|store|tables|lambda|it +2|RMSE|panicked|error" | tail -14
  [[ ${PIPESTATUS[0]} -eq 0 ]] && touch "$SM/DONE" || { echo "!! smoke failed: $SM/log.txt"; exit 1; }
  grep -E "Maximum resident|Elapsed" "$SM/time.txt"
fi

SR=$OUT/smoke_resp
if [[ ! -f $SR/DONE ]]; then
  stop
  echo "== smoke with $NR respiratory bins: $(date +%H:%M:%S)"
  mkdir -p "$SR"
  /usr/bin/time -v -o "$SR/time.txt" systemd-run --user --scope -q -p MemoryMax=${MEM}G -p MemorySwapMax=0 \
    $B/recon_p2np "$SF" "${BASE[@]}" "${RESP[@]}" --iters 2 --fresh --out "$SR" 2>&1 | tee "$SR/log.txt" | grep -E "respiratory|segments|SegNufft|lambda|it +2|RMSE|panicked|error|planner|does not fit" | tail -14
  [[ ${PIPESTATUS[0]} -eq 0 ]] && touch "$SR/DONE" || { echo "!! smoke with respiratory bins failed: $SR/log.txt"; exit 1; }
  grep -E "Maximum resident|Elapsed" "$SR/time.txt"
fi

[[ $UNTIL == smoke ]] && { echo "== stopping after the smoke runs (--until smoke): $(date +%H:%M:%S)"; exit 0; }

# ── 4. oracle vs estimates ────────────────────────────────────────────────────────────────────────────────────────
stop
rustlib/tools/recon/eval_matrix.sh "$SF" "$B0R" "$OUT" --mem "$MEM" -- "${RECON[@]}" || exit 1

# ── 5. segment basis: svdw at one segment fewer ───────────────────────────────────────────────────────────────────
stop
L=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['nseg'])" "$OUT/oracle/run.json")
echo "== segment basis: fixed L = $L (eval_10min/oracle) vs svdw L = $((L - 1))"
rustlib/tools/recon/eval_matrix.sh "$SF" "$B0R" "$PH/eval_10min_svdw" --configs oracle --mem "$MEM" -- "${RECON[@]}" --seg-basis svdw --nseg $((L - 1)) || exit 1
echo "== all done: $(date +%H:%M:%S)"
