#!/bin/bash
# §17 baseline: what a knot launch sees and costs today (before any new layout). opbench on flow_thorax 2 mm, 10
# virtual coils, L 10, order 3 (points_hi / points2d_hi tables), 1-D cubic 20 knots and 2-D 6:1,16:3 knots, at the
# reduced scan (0.44 min, 1.33 M samples per encode) and a 10x denser one (4.4 min, 13 M; the 10-min scan has 30 M).
# One case per invocation, .done per case, STOP to stop.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout
BIN=${BIN:-$SP/bin/opbench_kl0}
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
COMMON="--models seg-full --nseg 10 --vcoils 10 --order 3 --sets 1 --seq flow"
declare -A C
C[k1_lo]="--temporal cubic --knot --frames 20 --scan-min 0.44 --reps 3 --new-compare"
C[k2_lo]="--knots2d 6:1,16:3 --scan-min 0.44 --reps 3 --new-compare"
C[k1_hi]="--temporal cubic --knot --frames 20 --scan-min 4.4 --reps 1"
C[k2_hi]="--knots2d 6:1,16:3 --scan-min 4.4 --reps 1"
for k in ${CASES:-k1_lo k2_lo k1_hi k2_hi}; do
  [ -e $OUT/STOP ] && exit 0
  [ -e $OUT/base_$k.done ] && continue
  echo "=== $k: $BIN ${C[$k]} $COMMON ($(date))" > $OUT/base_$k.log
  flock $SP/gpu.lock timeout 900 $BIN ${C[$k]} $COMMON >> $OUT/base_$k.log 2>&1
  echo "exit $?" >> $OUT/base_$k.log
  touch $OUT/base_$k.done
done
touch $OUT/base.done
