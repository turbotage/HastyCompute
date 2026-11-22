#!/bin/bash
# final kl tests (t5, seg_kl5: build-to-build bound in kl_split_equals_unsplit_ra), the control, seg_ra_vt rerun
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
TBIN=$SP/bin/seg_kl5 TAG=t5 TESTS="kl_knot_orders kl_knot_split_lists kl_fused_and_slabs kl_support_union kl_unsplit_equals_ra kl_split_equals_unsplit_ra kl_autotune_choice" $D/run_tests1.sh
if [ ! -e $D/t5_control.done ]; then
  KL_CONTROL=1 flock $SP/gpu.lock timeout 900 $SP/bin/seg_kl5 kl_support_union --exact --nocapture --test-threads=1 > $D/t5_control.log 2>&1; echo "exit $?" >> $D/t5_control.log; touch $D/t5_control.done
fi
if [ ! -e $D/reg2_seg_ra_vt.done ]; then
  flock $SP/gpu.lock timeout 900 $SP/bin/reg2/seg_ra_vt ra_static_orders --exact --nocapture --test-threads=1 > $D/reg2_seg_ra_vt.log 2>&1; echo "exit $?" >> $D/reg2_seg_ra_vt.log; touch $D/reg2_seg_ra_vt.done
fi
