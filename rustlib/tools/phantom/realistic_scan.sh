#!/usr/bin/env bash
# Realistic synthetic scans of one phantom, end to end (GPU): improved LBM (vessel trees + coronary zone, beat
# libraries with the protocol's spin history) → blood frames → scanner-like prescription → 10-min 4D flow and 2.5-min
# B0 / R2* / fat k-space → truths. Every step is skipped when its output exists (rerun = resume; the LBM runs and
# simphantom resume inside a step too); heavy steps run in a systemd scope with a memory cap and no swap. A STOP file
# in <phantom_dir> or HASTY_DEADLINE=<epoch s> ends it between steps; GPU power: HASTY_GPU_WATTS / HASTY_GPU_DUTY.
#
#   rustlib/tools/phantom/realistic_scan.sh <phantom_dir> [--beats 15] [--flow-min 10] [--b0-min 2.5] [--atlas]
#        [--res 0.002] [--keep-box heart:40] [--octree] [--tag NAME] [--flow-run data/hasty/seq/flow_thorax]
#
# --res: prescribed resolution (m) = reconstruction grid; --keep-box: tissue and blood only inside the box, FOV from it
# (development: no arms, small FOV; prep_sim --keep-box); --octree: multi-resolution box sources (prep_sim
# --src-octree) instead of 1 mm in the heart box / 2 mm points outside; --tag: suffix of the scan directories
# (sim_flow_<min>min<tag>, sim_b0_<min>min<tag>) so variants do not overwrite each other; --flow-run: the flow
# sequence design (seqdesign run; its resolution must match --res).
#
# --atlas: first rebuild the p0 atlas with the arms (run_p0.sh --labels, 520 × 320 × 360 mm) — then make a new
#          factory phantom from it (make_phantom.py) and pass that phantom here; existing phantoms keep their atlas.
# Outputs under <phantom_dir>: lbm/{case_tree,run_lib,case_coronary,run_cor}, simblood_lib/, sim_flow_<min>min/,
# sim_b0_<min>min/, logs/realistic_*.log.
set -euo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
PH=$(realpath "$1"); shift
NB=15; FLOW_MIN=10; B0_MIN=2.5; ATLAS=0; RES=0.002; KEEPBOX=""; OCTREE=0; TAG=""; FLOWRUN=data/hasty/seq/flow_thorax
while [[ $# -gt 0 ]]; do
  case $1 in
    --beats) NB=$2; shift 2 ;;
    --flow-min) FLOW_MIN=$2; shift 2 ;;
    --b0-min) B0_MIN=$2; shift 2 ;;
    --atlas) ATLAS=1; shift ;;
    --res) RES=$2; shift 2 ;;
    --keep-box) KEEPBOX=$2; shift 2 ;;
    --octree) OCTREE=1; shift ;;
    --tag) TAG=$2; shift 2 ;;
    --flow-run) FLOWRUN=$2; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
PY="rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
B=rustlib/target/release
mkdir -p "$PH/logs"
# a new start clears the STOP files of the last stop (stop_all.sh)
rm -f "$PH/STOP" "$PH"/lbm/run_*/STOP "$PH"/sim_*/scan/STOP
cap() { local g=$1; shift; systemd-run --user --scope -q -p MemoryMax=${g}G -p MemorySwapMax=0 "$@"; }
# stop rules: a STOP file in <phantom_dir> or the epoch HASTY_DEADLINE ends the pipeline before the next step; a step whose
# output is missing after it returned was stopped (its own STOP file / Ctrl-C, resumable): the pipeline ends there.
# Rerunning the same command resumes (finished steps are skipped, the LBM and simphantom continue from their states).
stop_now() {
  [[ -e "$PH/STOP" ]] && { echo "== STOP file: ending before the next step"; exit 0; }
  if [[ -n "${HASTY_DEADLINE:-}" ]] && (( $(date +%s) >= HASTY_DEADLINE )); then echo "== deadline reached: ending before the next step"; exit 0; fi
  return 0
}
step() {  # step <name> <done-file> <cmd...>
  local name=$1 done=$2; shift 2
  if [[ -e $done ]]; then echo "== $name: done ($done)"; return; fi
  stop_now
  echo "== $name: $(date +%H:%M:%S)"
  "$@" >> "$PH/logs/realistic_$name.log" 2>&1 || { echo "!! $name failed, see $PH/logs/realistic_$name.log"; tail -5 "$PH/logs/realistic_$name.log"; exit 1; }
  tail -2 "$PH/logs/realistic_$name.log"
  [[ -e $done ]] || { echo "== $name stopped before finishing (resumable): ending the pipeline"; exit 0; }
}

if [[ $ATLAS == 1 ]]; then
  step atlas data/hasty/phantom/p0/.arms_done bash -c "systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 rustlib/tools/phantom/run_p0.sh --labels && touch data/hasty/phantom/p0/.arms_done"
fi

# protocol: the flow sequence (seqdesign flow_thorax: TR 11.29 ms, 12°) and the anatomy's slab
FLIP=12; TR_FLOW=0.01129
step slab "$PH/slab.json" $PY $T/slab.py "$PH"
CZ=$(python3 -c "import json;print(json.load(open('$PH/slab.json'))['center_z'])")
THK=$(python3 -c "import json;print(json.load(open('$PH/slab.json'))['thk'])")
SPIN="tr=$TR_FLOW,flip=$FLIP,center=0;0;$CZ,thk=$THK"

# improved LBM: vessel trees (heart + great vessels) and the coronary zone, beat libraries with the spin history
step lbm_case "$PH/lbm/case_tree/case.json" cap 24 $PY $T/lbm_case.py "$PH" "$PH/lbm/case_tree"
# an LBM run is finished when cardiac_lbm reports "done" (flow_bins.json is also written at checkpoints): lbm_run
# writes <run>/DONE then; a stopped run (STOP file, Ctrl-C) leaves it out and resumes on the next start
lbm_run() {  # lbm_run <case> <run> <args...>
  local c=$1 r=$2; shift 2
  cap 16 $B/cardiac_lbm "$c" "$r" "$@" | tee -a "$r.stdout" | tail -1 | grep -q "^done after" && touch "$r/DONE"
  return 0
}
step lbm_lib "$PH/lbm/run_lib/DONE" lbm_run "$PH/lbm/case_tree" "$PH/lbm/run_lib" --cycles $((NB + 3)) --warmup 3 --beats --spin "$SPIN" --quiet
step cor_case "$PH/lbm/case_coronary/case.json" cap 16 $PY $T/lbm_coronary.py "$PH" "$PH/lbm/case_coronary"
step cor_lib "$PH/lbm/run_cor/DONE" lbm_run "$PH/lbm/case_coronary" "$PH/lbm/run_cor" --cycles 9 --warmup 3 --beats --spin "$SPIN" --quiet

# blood frames (Eulerian, 1 mm, 100 per cycle) with the LBM chambers, great vessels and coronaries
P0A="$PH/atlas"; [[ -e $P0A/tissue_1mm.hv ]] || P0A=data/hasty/phantom/p0
step blood "$PH/simblood_lib/blood_frames.hv" cap 48 $PY $T/phantom_4d.py "$PH" "$PH/motion" "$PH/simblood_lib" --labels-src "$P0A/tissue_1mm.hv" --vox 1 --frames 100 \
  --blood-export --chamber-flow "lbm:$PH/lbm/run_lib" --lbm-vessels --ref-flow "$PH/lbm/run_cor"

# 4D flow: scanner-like prescription; sources 1 mm in the heart box / 2 mm points outside, or the octree (--octree)
FS=$(python3 -c "print(int(round($FLOW_MIN*60)))"); SF="$PH/sim_flow_${FLOW_MIN}min${TAG}"
PREP=(--slab auto --fov auto --res "$RES")
if [[ $OCTREE == 1 ]]; then PREP+=(--src-octree); else PREP+=(--src-roi heart --src-step-out 2); fi
[[ -n $KEEPBOX ]] && PREP+=(--keep-box "$KEEPBOX")
step flow_prep "$SF/sim_input.hv" cap 48 $PY $T/prep_sim.py "$PH" "$PH/simblood_lib/blood_frames.hv" "$SF" "${PREP[@]}" --scan-time $FS
# every library beat cached (≈ 2.5 GB each at 0.5 M covered sources × 100 frames; base ≈ 38 GB → cap 84 G): ≈ 700 heartbeats drawn from NB beats would recompute ≈ 27 s per miss
step flow_sim "$SF/scan/data_clean.npy" cap 84 $B/simphantom "$SF/sim_input.hv" "$SF/scan" --seq flow --flow-run "$FLOWRUN" --flip $FLIP --beats "$PH/lbm/run_lib" --beats-ref "$PH/lbm/run_cor" --beat-cache $NB
step flow_truth "$SF/scan/truth_fat_fraction.npy" cap 48 $PY $T/sim_truth.py "$PH" "$SF/scan"
step flow_bias "$SF/scan/truth_bias.npz" cap 64 $PY $T/bias_truth.py "$PH" "$SF/scan"

# B0 / R2* / fat multi-echo: same prescription; the flow library's velocities with the steady-state blood signal
BS=$(python3 -c "print(int(round($B0_MIN*60)))"); SB="$PH/sim_b0_${B0_MIN}min${TAG}"
step b0_prep "$SB/sim_input.hv" cap 48 $PY $T/prep_sim.py "$PH" "$PH/simblood_lib/blood_frames.hv" "$SB" "${PREP[@]}" --scan-time $BS
step b0_sim "$SB/scan/data_clean.npy" cap 84 $B/simphantom "$SB/sim_input.hv" "$SB/scan" --seq b0 --flip $FLIP --beats "$PH/lbm/run_lib" --beats-ref "$PH/lbm/run_cor" --beat-spin off --beat-cache $NB
step b0_truth "$SB/scan/truth_fat_fraction.npy" cap 48 $PY $T/sim_truth.py "$PH" "$SB/scan"
echo "== all done: $SF/scan, $SB/scan"
