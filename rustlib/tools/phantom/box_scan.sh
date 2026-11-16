#!/usr/bin/env bash
# Small-FOV benchmark scan at coronary resolution (research/box_benchmark.md, research/sim_convergence.md): the v2
# 4D-flow protocol at 1 mm (flow_thorax_v2_1mm, flip 15°, GIRF resonance chain) on a box around the heart and the
# aortic arch (prep_sim --keep-box great:15), every 4th encode group (the full FOV's per-frame undersampling), the
# full scan's receiver noise. Beat libraries with the v2 spin history (cardiac_lbm --spin), then simphantom, then the
# truths. Steps are skipped when their output exists (rerun = resume; the LBM and simphantom resume inside a step);
# a STOP file in <phantom_dir>/box_STOP or HASTY_DEADLINE=<epoch s> ends it between steps.
#
#   rustlib/tools/phantom/box_scan.sh <phantom_dir> [--sim-coils off] [--tag _box1mm_b] [--stride auto]
#        [--cache 15] [--girf-hist 10] [--blood <blood_frames.hv>] [--collapse 2] [--extra "<simphantom args>"] [--only-lbm]
#
# Discretisation from the convergence study (research/sim_convergence.md), every change < 0.1–0.2 of the per-frame
# noise in its region: 1 ms motion bins, Eulerian blood frames without the frame → bin shift and blended linearly in
# time (--blood-advect off --blood-blend on), blood sources as boxes of their voxel (--blood-box on), 2³ library points per blood source, 5³ per coronary source, 2³ sub-boxes
# for the 4 / 8 mm octree elements. Coil compression does not pay off on the body array (--sim-coils off).
# --blood: the blood frames (default the realistic scan's 1 mm simblood_lib); with 0.5 mm frames (phantom_4d --vox
# 0.5) and --collapse 2, prep_sim makes 1 mm sources that integrate the chamber walls' partial volume at 0.5 mm.
#
# After an LBM run is done, its accumulator (flow_bins.f32) and checkpoint are deleted: simphantom's beat library needs
# only flow_bins.json and beats/ (≈ 25 GB saved per pair).
set -euo pipefail
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
PH=$(realpath "$1"); shift
SIMCOILS=off; TAG=_box1mm_b; STRIDE=auto; CACHE=15; GHIST=10; EXTRA=""; ONLYLBM=0; BLOOD=""; COLLAPSE=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --sim-coils) SIMCOILS=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;;
    --stride) STRIDE=$2; shift 2 ;;
    --cache) CACHE=$2; shift 2 ;;
    --girf-hist) GHIST=$2; shift 2 ;;
    --extra) EXTRA=$2; shift 2 ;;
    --blood) BLOOD=$2; shift 2 ;;
    --collapse) COLLAPSE=$2; shift 2 ;;
    --only-lbm) ONLYLBM=1; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
PY="rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
B=rustlib/target/release
SEQ=data/hasty/seq/flow_thorax_v2_1mm
FLIP=15
TR=$(python3 -c "import json;print(json.load(open('$SEQ/summary.json'))['best']['tr'])" 2>/dev/null || echo 0.01517)
SF="$PH/sim_flow_10min$TAG"
mkdir -p "$PH/logs"
rm -f "$PH/box_STOP"
cap() { local g=$1; shift; systemd-run --user --scope -q -p MemoryMax=${g}G -p MemorySwapMax=0 "$@"; }
stop_now() {
  [[ -e "$PH/box_STOP" ]] && { echo "== STOP file: ending before the next step"; exit 0; }
  if [[ -n "${HASTY_DEADLINE:-}" ]] && (( $(date +%s) >= HASTY_DEADLINE )); then echo "== deadline reached"; exit 0; fi
  return 0
}
step() {  # step <name> <done-file> <cmd...>
  local name=$1 done=$2; shift 2
  if [[ -e $done ]]; then echo "== $name: done ($done)"; return; fi
  stop_now
  echo "== $name: $(date +%H:%M:%S)"
  "$@" >> "$PH/logs/box_$name.log" 2>&1 || { echo "!! $name failed, see $PH/logs/box_$name.log"; tail -5 "$PH/logs/box_$name.log"; exit 1; }
  tail -2 "$PH/logs/box_$name.log"
  [[ -e $done ]] || { echo "== $name stopped before finishing (resumable): ending"; exit 0; }
}

CZ=$(python3 -c "import json;print(json.load(open('$PH/slab.json'))['center_z'])")
THK=$(python3 -c "import json;print(json.load(open('$PH/slab.json'))['thk'])")
SPIN="tr=$TR,flip=$FLIP,center=0;0;$CZ,thk=$THK"
echo "protocol: $SEQ, TR $TR s, flip $FLIP°, spin $SPIN"

lbm_run() {  # lbm_run <case> <run> <args...>: DONE when cardiac_lbm reports "done", then drop the accumulator / checkpoint
  local c=$1 r=$2; shift 2
  cap 16 $B/cardiac_lbm "$c" "$r" "$@" | tee -a "$r.stdout" | tail -1 | grep -q "^done after" && { rm -f "$r/flow_bins.f32" "$r/ckpt.bin"; touch "$r/DONE"; }
  return 0
}
step lib_v2 "$PH/lbm/run_lib_v2/DONE" lbm_run "$PH/lbm/case_tree" "$PH/lbm/run_lib_v2" --cycles 18 --warmup 3 --beats --spin "$SPIN" --quiet
step cor_v2 "$PH/lbm/run_cor_v2/DONE" lbm_run "$PH/lbm/case_coronary" "$PH/lbm/run_cor_v2" --cycles 9 --warmup 3 --beats --spin "$SPIN" --quiet
[[ $ONLYLBM == 1 ]] && { echo "== --only-lbm: libraries done"; exit 0; }

# the box input: heart + aortic arch, 1 mm grid, octree sources (prep_sim; the blood frames of the realistic scan)
BLOOD=${BLOOD:-$PH/simblood_lib/blood_frames.hv}
step prep "$SF/sim_input.hv" cap 48 $PY $T/prep_sim.py "$PH" "$BLOOD" "$SF" --slab auto --fov auto --res 0.001 \
  --src-octree --keep-box great:15 --keep-box-fov 15 --scan-time 600 --blood-collapse $COLLAPSE
SIMDISC="--bin-ms 1 --blood-advect off --blood-blend on --blood-box on --beat-sub 2 --beat-ref-sub 5 --tissue-sub 2 --tissue-sub-min-mm 3"
# every library beat cached (≈ 2.5 GB each; base ≈ 20 GB)
step sim "$SF/scan/meta.json" cap 84 $B/simphantom "$SF/sim_input.hv" "$SF/scan" --seq flow --flow-run "$SEQ" --flip $FLIP \
  --girf resonance --girf-hist $GHIST --beats "$PH/lbm/run_lib_v2" --beats-ref "$PH/lbm/run_cor_v2" --beat-cache $CACHE \
  --readout-stride $STRIDE --sim-coils $SIMCOILS --noise-sigma 39.98 $SIMDISC $EXTRA
step truth "$SF/scan/truth_fat_fraction.npy" cap 48 $PY $T/sim_truth.py "$PH" "$SF/scan"
# velocity truths (u_vol, u_sig, u_pc): memory ∝ LBM cells × frames (≈ 40 GB), not the reconstruction grid
step bias "$SF/scan/truth_bias.npz" cap 64 $PY $T/bias_truth.py "$PH" "$SF/scan"
echo "== box scan done: $SF/scan"
