#!/bin/bash
# §17.x: why ra on the static 48 coils × 1 segment set (4.3 M samples, order 0) is 2× the prototype: Nsight Compute on
# one ra launch at the autotune's shape (nxg 4, cb 3: two block rows of 24 channels) and at the prototype's (nxg 2,
# cb 3: one row of 48). The slot (48 channels × 288 × 288 × 140) is 4.5 GB: targeted sections only.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout/ncu
BIN=${BIN:-$SP/bin/opbench_kl3}
NCU=/usr/local/cuda/bin/ncu
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
ARGS="--models seg-none --vcoils 48 --cb-cap 48 --frames 7 --sets 1 --seq flow --reps 1 --no-tune"
SEC="--section LaunchStats --section Occupancy --section SpeedOfLight --section WarpStateStats --section SchedulerStats --section SourceCounters --section MemoryWorkloadAnalysis"
for cfg in 4,3,16 2,3,16; do
  n=ra48_${cfg//,/_}
  [ -e $OUT/$n.done ] && continue
  flock $SP/gpu.lock timeout 900 env HASTY_SEG_VSA=1 HASTY_SEG_VSA_CFG=$cfg HASTY_SEG_VS=0 HASTY_NVRTC_LINEINFO=1 $NCU $SEC -k regex:seg_adjoint_ra --launch-skip 2 --launch-count 1 -f -o $SP/ncu/$n $BIN $ARGS > $OUT/$n.log 2>&1
  echo "exit $?" >> $OUT/$n.log
  $NCU -i $SP/ncu/$n.ncu-rep --page details > $OUT/$n.details.txt 2>&1
  touch $OUT/$n.done
done
