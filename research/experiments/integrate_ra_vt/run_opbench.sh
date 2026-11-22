#!/bin/bash
# opbench of the ra / vt integration (research/sample_kernels.md §16): forward / adjoint / gradient with the new
# variants off and on (--new-compare). One case per invocation, .done per case, STOP to stop; a case that hits the
# 900 s timeout (first-time NVRTC of the autotune candidates) is rerun once on the warm kernel cache.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/integrate_ra_vt
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
COMMON="--sets 1 --reps 3 --seq flow --new-compare"
declare -A C
C[knot_o2]="--models seg-full --nseg 10 --vcoils 10 --order 2 --temporal cubic --knot --frames 20 --scan-min 0.44"
C[knot_o3]="--models seg-full --nseg 10 --vcoils 10 --order 3 --temporal cubic --knot --frames 20 --scan-min 0.44"
C[static_o2]="--models seg-full --nseg 10 --vcoils 10 --order 2 --frames 7"
C[c48]="--models seg-none --vcoils 48 --cb-cap 48 --frames 7"
for k in ${CASES:-knot_o2 knot_o3 static_o2 c48}; do
  [ -e $OUT/STOP ] && exit 0
  [ -e $OUT/ob_$k.done ] && continue
  for try in 1 2; do
    flock $SP/gpu.lock timeout 900 $SP/bin/opbench ${C[$k]} $COMMON > $OUT/ob_$k.log 2>&1
    rc=$?
    echo "exit $rc (try $try)" >> $OUT/ob_$k.log
    [ $rc -ne 124 ] && break
  done
  touch $OUT/ob_$k.done
done
touch $OUT/opbench.done
