#!/bin/bash
# Pre-registered (2026-11-20): how many time segments L does the knot engine's segment model need when the constant
# and linear spatial parts of the per-knot ratemap are moved into the per-sample tables before the segment SVD?
# Decision: if L drops noticeably -> build per-knot sample tables with per-knot absorption (written before sorting).
# CPU only (no device). Resumable: each step leaves a .done marker.
set -u
cd /home/turbotage/Documents/GitHub/HastyCompute
B=rustlib/target/release/fieldabsorb
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
O=research/experiments/field_absorb
export RAYON_NUM_THREADS=4
run() { # name, args...
  local n=$1; shift
  [ -f $O/$n.done ] && return
  $B $S "$@" --json $O/$n.json > $O/$n.log 2>&1 && touch $O/$n.done
}
# 8 mm grid (the dev smoke's grid; warm start, support and fit region of research/experiments/ratemap_bc/ab_truth-bc)
run knots_8mm  --warm research/experiments/ratemap_bc/ab_truth-bc --grid-scale 0.25 --vcoils 8 --nofat-stride 4 --opt-stride 8
run frames_8mm --warm research/experiments/ratemap_bc/ab_truth-bc --grid-scale 0.25 --vcoils 8 --knots2d 6:0,16:0 --variants a,c,d,e,g --enc 0,3 --nofat-stride 4 --opt-stride 8 --control 0
# 4 mm grid (vknot/smoke4mm), every 3rd knot
run knots_4mm  --warm data/hasty/phantom/factory_arms/ph0001/vknot/smoke4mm --grid-scale 0.5 --vcoils 31 --knot-stride 3 --enc 0,3 --nofat-stride 4 --opt-stride 4
run frames_4mm --warm data/hasty/phantom/factory_arms/ph0001/vknot/smoke4mm --grid-scale 0.5 --vcoils 31 --knots2d 6:0,16:0 --knot-stride 3 --enc 0 --variants a,c,g --control 0
# 2 mm native grid (vknot/dev_k8/B_vknot), every 6th knot, rule voxels every 12th fit voxel; stride check on 2 knots
run knots_2mm  --warm data/hasty/phantom/factory_arms/ph0001/vknot/dev_k8/B_vknot --vcoils 12 --knot-stride 6 --seg-fit-stride 12 --enc 0,3 --variants a,b,c,d,e,g --nofat-stride 4 --opt-stride 4
run stride_2mm --warm data/hasty/phantom/factory_arms/ph0001/vknot/dev_k8/B_vknot --vcoils 12 --knot-stride 48 --seg-fit-stride 3 --variants a,c --enc 0 --control 0
