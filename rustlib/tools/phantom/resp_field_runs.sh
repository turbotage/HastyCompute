#!/bin/bash
# Datasets with the breathing + cardiac field (research/resp_field_sim.md). Resumable: rerun the same command (finished
# steps are skipped, simphantom resumes from its chunk states). GPU steps take the GPU lock given as $GPU_LOCK (optional).
#   resp_field_runs.sh b0      → <ph>/sim_b0_2.5min_respfield (fields + truth ratemap, then the multi-echo scan, ≈ 2–2.5 h)
#   resp_field_runs.sh adv10   → <ph>/sim_flow_10min/scan/truth_ratemap_bc (truth ratemap of the existing advected-field scan)
set -euo pipefail
cd "$(dirname "$0")/../../.."
PH=data/hasty/phantom/factory_arms/ph0001
PY="rustlib/.venv/bin/python -I"
B=${SIMPHANTOM:-rustlib/target/snap/simphantom_respfield}
cap() { systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 "$@"; }
gpu() { if [[ -n ${GPU_LOCK:-} ]]; then ( flock 9; "$@" ) 9>"$GPU_LOCK"; else "$@"; fi; }
case $1 in
  b0)
    D=$PH/sim_b0_2.5min_respfield; mkdir -p $D
    [[ -e $D/scan/truth_ratemap_bc/meta.json ]] || gpu cap $PY rustlib/tools/phantom/resp_field.py $PH $PH/sim_b0_2.5min/sim_input.hv $D/resp_field.hv --k 6 --q 20 --check 8 --truth-dir $D/scan/truth_ratemap_bc > $D/resp_field.log 2>&1
    # the scan of sim_b0_2.5min (same input, sequence, flip) with the field; no beat library: its 15-beat cache needs
    # ≈ 84 GB (realistic_scan.sh), the blood keeps the frames' LBM mean velocities
    [[ -e $D/scan/data_clean.hv ]] || gpu cap $B $PH/sim_b0_2.5min/sim_input.hv $D/scan --seq b0 --flip 12 --resp-field $D/resp_field.hv >> $D/sim.log 2>&1
    ;;
  adv10)
    T=$PH/sim_flow_10min/scan/truth_ratemap_bc
    # QFROM=<q>: resume an interrupted run at cardiac phase q (keeps the arrays); log appended
    [[ -e $T/meta.json ]] || gpu cap $PY rustlib/tools/phantom/resp_field.py $PH $PH/sim_flow_10min/sim_input.hv /dev/null --k 6 --q 20 --advected-only --truth-dir $T --q-from ${QFROM:-0} >> $PH/logs/truth_ratemap_bc_10min.log 2>&1
    ;;
  *) echo "b0 | adv10"; exit 2 ;;
esac
