#!/bin/bash
# after chain3: the kl tests with the final bounds (t3, binary seg_kl3)
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
while systemctl --user is-active --quiet kl-chain3; do sleep 20; done
TBIN=$SP/bin/seg_kl3 TAG=t3 TESTS="kl_knot_orders kl_knot_split_lists kl_fused_and_slabs kl_support_union kl_unsplit_equals_ra kl_autotune_choice" $D/run_tests1.sh
if [ ! -e $D/t3_control.done ]; then
  KL_CONTROL=1 flock $SP/gpu.lock timeout 900 $SP/bin/seg_kl3 kl_support_union --exact --nocapture --test-threads=1 > $D/t3_control.log 2>&1; echo "exit $?" >> $D/t3_control.log; touch $D/t3_control.done
fi
