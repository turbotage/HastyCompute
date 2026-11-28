#!/usr/bin/env bash
# Short LBM check of a phantom (research/flow_phantom.md Tier B5): the case at a coarse lattice (default 1 mm, ~30 s for
# 5 cycles on the 4090), a few cycles, the report (valve fluxes vs the cardiac model, jets, peak speeds) and the leak
# gate of the case builder (its lumen gate exits non-zero when a stub lumen is pinched).
#   rustlib/tools/phantom/lbm_burst.sh <phantom_dir> <out_dir> [dx_mm=1] [cycles=4]
set -euo pipefail
cd "$(dirname "$0")/../../.."
PH=$1; OUT=$2; DX=${3:-1}; CYC=${4:-4}
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
LOCK=${HASTY_GPU_LOCK:-jobs/locks/gpu.lock}
mkdir -p "$OUT"
# CASE_ARGS: extra lbm_case.py options (e.g. "--motion <file> --sl-hinge annulus")
# shellcheck disable=SC2086
$PY $T/lbm_case.py "$PH" "$OUT/case" --dx-mm "$DX" ${CASE_ARGS:-} > "$OUT/case.log" 2>&1 || { echo "lbm_case failed:"; tail -5 "$OUT/case.log"; exit 1; }
tail -2 "$OUT/case.log"
flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 rustlib/target/release/cardiac_lbm "$OUT/case" "$OUT/run" --cycles "$CYC" --warmup $((CYC / 2)) --quiet > "$OUT/run.log" 2>&1 || { echo "cardiac_lbm failed:"; tail -5 "$OUT/run.log"; exit 1; }
tail -1 "$OUT/run.log"
$PY $T/lbm_report.py "$OUT/case" "$OUT/run" > "$OUT/report.log" 2>&1 || { echo "lbm_report failed:"; tail -5 "$OUT/report.log"; exit 1; }
grep -E "^(valve gate|cut cross-check|density gate|tree balance)" "$OUT/report.log" || true
tail -12 "$OUT/report.log"
