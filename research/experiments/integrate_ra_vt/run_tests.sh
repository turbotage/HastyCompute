#!/bin/bash
# GPU tests of the ra / vt integration (research/sample_kernels.md §16), one test per invocation, resumable (.done)
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/integrate_ra_vt
cd $SP
for t in ${TESTS:-ra_static_orders ra_knot_orders vt_static_order2 vt_knot_orders ra_vt_fused_and_slabs ra_group_and_source ra_vt_autotune_choice}; do
  [ -e $OUT/STOP ] && exit 0
  [ -e $OUT/test_$t.done ] && continue
  flock $SP/gpu.lock timeout 900 $SP/bin/seg_ra_vt $t --exact --nocapture --test-threads=1 > $OUT/test_$t.log 2>&1
  echo "exit $?" >> $OUT/test_$t.log
  touch $OUT/test_$t.done
done
touch $OUT/tests.done
