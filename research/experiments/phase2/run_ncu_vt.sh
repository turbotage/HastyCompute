#!/bin/bash
# §19: Nsight Compute on one knot launch (2-D 6:1,16:3, order 3, reduced scan at 3 mm: 50 channels × 192 × 192 × 94
# = 1.4 GB slot) of each kernel variant, fixed by env (--no-tune). Targeted sections only (achieved occupancy, warp
# states, scheduler). Reports in the scratchpad; text details here.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/phase2/ncu
BIN=${BIN:-$SP/bin/opbench_p2}
NCU=/usr/local/cuda/bin/ncu
mkdir -p $SP/ncu
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
ARGS="--models seg-full --nseg 10 --vcoils 10 --order 3 --sets 1 --seq flow --knots2d 6:1,16:3 --scan-min 0.44 --res 3e-3 --reps 1 --no-tune --cb-cap 5"
SEC="--section LaunchStats --section Occupancy --section SpeedOfLight --section WarpStateStats --section SchedulerStats"
run() {
  local n=$1 k=$2 e=$3
  [ -e $OUT/$n.done ] && return 0
  [ -e $OUT/../STOP ] && exit 0
  flock $SP/gpu.lock bash -c "echo \"locked \$(date +%T) power limit \$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)\" > $OUT/$n.log; timeout 900 env $e $NCU $SEC -k regex:$k --launch-skip 6 --launch-count 1 -f -o $SP/ncu/$n $BIN $ARGS >> $OUT/$n.log 2>&1; echo \"exit \$?\" >> $OUT/$n.log"
  $NCU -i $SP/ncu/$n.ncu-rep --page details > $OUT/$n.details.txt 2>&1
  touch $OUT/$n.done
}
VT="HASTY_SEG_VSA=0 HASTY_SEG_VT=1 HASTY_SEG_KL=0"
run vt5_base   seg_forward_vs "$VT HASTY_SEG_VS_CFG=4,1,0"
run vt5_nws2   seg_forward_vs "$VT HASTY_SEG_VS_CFG=4,2,0"
run vt5_q24    seg_forward_vs "$VT HASTY_SEG_VS_CFG=4,1,0,24"
run vt5_q16    seg_forward_vs "$VT HASTY_SEG_VS_CFG=4,1,0,16"
touch $OUT/ncu_vt.done
