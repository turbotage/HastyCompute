#!/bin/bash
# §17: Nsight Compute on one knot launch of the engine's ra adjoint and vt forward (1-D cubic 20 knots, order 3, the
# reduced scan at 3 mm so that the replayed slot is small: 50 channels × 192 × 192 × 94 cells = 1.4 GB). Targeted
# sections only (full-set replay of multi-GiB grids fails). Reports stay in the scratchpad; text summaries here.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout/ncu
BIN=${BIN:-$SP/bin/opbench_kl0}
NCU=/usr/local/cuda/bin/ncu
mkdir -p $SP/ncu
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
ARGS="--models seg-full --nseg 10 --vcoils 10 --order 3 --sets 1 --seq flow --temporal cubic --knot --frames 20 --scan-min 0.44 --res 3e-3 --reps 1 --no-tune"
SEC="--section LaunchStats --section Occupancy --section SpeedOfLight --section WarpStateStats --section SchedulerStats --section SourceCounters --section InstructionStats --section MemoryWorkloadAnalysis"
run() {
  local n=$1 k=$2 e=$3
  [ -e $OUT/$n.done ] && return 0
  [ -e $OUT/../STOP ] && exit 0
  flock $SP/gpu.lock timeout 900 env $e HASTY_NVRTC_LINEINFO=1 $NCU $SEC -k regex:$k --launch-skip 6 --launch-count 1 -f -o $SP/ncu/$n $BIN $ARGS > $OUT/$n.log 2>&1
  echo "exit $?" >> $OUT/$n.log
  $NCU -i $SP/ncu/$n.ncu-rep --page details > $OUT/$n.details.txt 2>&1
  $NCU -i $SP/ncu/$n.ncu-rep --page source --csv --print-source sass > $SP/ncu/$n.source.csv 2>/dev/null
  touch $OUT/$n.done
}
run ra_knot seg_adjoint_ra "HASTY_SEG_VSA=1 HASTY_SEG_VSA_CFG=4,7,16 HASTY_SEG_VT=0"
run vt_knot seg_forward_vs "HASTY_SEG_VT=1 HASTY_SEG_VSA=0"
touch $OUT/ncu.done
