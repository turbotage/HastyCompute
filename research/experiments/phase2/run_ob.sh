#!/bin/bash
# §19 operator timing: opbench on flow_thorax 2 mm (288×288×140 grid), 10 virtual coils, L 10, order 3, 2-D knots
# 6:1,16:3 (the velocity reconstruction), 1.33 M (lo) and 13.3 M (hi) samples per encode, coil batch 5.
# "old" = the candidate set before phase 2 (no staging variants, no hints, no spill screen, no vs sub-chunks);
# "new" = defaults. Per-candidate kernel times in the logs (HASTY_SEG_*_LOG). One case per invocation, .done per case.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/phase2
BIN=${BIN:-$SP/bin/opbench_p2}
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
COMMON="--models seg-full --nseg 10 --vcoils 10 --sets 1 --seq flow --order 3 --knots2d 6:1,16:3"
OLD="HASTY_SEG_VSA_STAGE=0 HASTY_SEG_LBMIN=0 HASTY_SEG_SPILLSKIP=0 HASTY_SEG_VS_SCH=0"
LOG="HASTY_SEG_VSA_LOG=1 HASTY_SEG_KL_LOG=1 HASTY_SEG_VS_LOG=1"
declare -A C E
C[lo_old]="--scan-min 0.44 --reps 3"; E[lo_old]="$OLD"
C[lo_new]="--scan-min 0.44 --reps 3"; E[lo_new]=""
C[hi_old]="--scan-min 4.4 --reps 1"; E[hi_old]="$OLD"
C[hi_new]="--scan-min 4.4 --reps 1"; E[hi_new]=""
for k in ${CASES:-lo_old lo_new hi_old hi_new}; do
  [ -e $OUT/STOP ] && exit 0
  [ -e $OUT/ob_$k.done ] && continue
  echo "=== $k: env ${E[$k]} $BIN $COMMON ${C[$k]} ($(date))" > $OUT/ob_$k.log
  flock $SP/gpu.lock bash -c "echo \"locked $(date +%T) power limit \$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)\" >> $OUT/ob_$k.log; timeout 900 env ${E[$k]} $LOG $BIN $COMMON ${C[$k]} >> $OUT/ob_$k.log 2>&1; echo \"exit \$? \$(date +%T)\" >> $OUT/ob_$k.log"
  touch $OUT/ob_$k.done
done
touch $OUT/ob.done
