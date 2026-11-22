#!/bin/bash
# §17 opbench with the knot-path layout: --kl-compare rows (previous path / skip of empty tiles / whole-tile kl kernel),
# kl4_*: the low-density cases again, at the lowered power limit (power.log), so that every table row comes from one limit
# the autotune times on knot 0, and the coil batch curve (1-D dense at coil batch 2 and 10).
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout
BIN=${BIN:-$SP/bin/opbench_kl5}
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
COMMON="--models seg-full --nseg 10 --vcoils 10 --sets 1 --seq flow"
declare -A C
C[k1_lo]="--order 3 --temporal cubic --knot --frames 20 --scan-min 0.44 --reps 3 --kl-compare"
C[k2_lo]="--order 3 --knots2d 6:1,16:3 --scan-min 0.44 --reps 3 --kl-compare"
C[k1_lo_o2]="--order 2 --temporal cubic --knot --frames 20 --scan-min 0.44 --reps 3 --kl-compare"
C[k2_lo_o2]="--order 2 --knots2d 6:1,16:3 --scan-min 0.44 --reps 3 --kl-compare"
C[k1_hi]="--order 3 --temporal cubic --knot --frames 20 --scan-min 4.4 --reps 1 --kl-compare"
C[k2_hi]="--order 3 --knots2d 6:1,16:3 --scan-min 4.4 --reps 1 --kl-compare"
C[k1_hi_cb2]="--order 3 --temporal cubic --knot --frames 20 --scan-min 4.4 --reps 1 --cb-cap 2 --kl-compare"
C[k1_hi_cb10]="--order 3 --temporal cubic --knot --frames 20 --scan-min 4.4 --reps 1 --cb-cap 10 --max-grid-gb 10 --kl-compare"
for k in ${CASES:-k2_hi k2_lo k1_lo k1_hi k2_lo_o2 k1_lo_o2}; do
  [ -e $OUT/STOP ] && exit 0
  [ -e $OUT/kl4_$k.done ] && continue
  echo "=== $k: $BIN $COMMON ${C[$k]} ($(date))" > $OUT/kl4_$k.log
  nvidia-smi --query-gpu=power.limit --format=csv,noheader >> $OUT/kl4_$k.log
  flock $SP/gpu.lock timeout 900 $BIN $COMMON ${C[$k]} >> $OUT/kl4_$k.log 2>&1
  echo "exit $?" >> $OUT/kl4_$k.log
  touch $OUT/kl4_$k.done
done
