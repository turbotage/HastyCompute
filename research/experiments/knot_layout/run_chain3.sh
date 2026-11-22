#!/bin/bash
# waits for chain2 to stop (STOP after its current case), then: kl kernel tests (t2), the control, opbench (kl3_*)
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
while systemctl --user is-active --quiet kl-chain2; do sleep 20; done
rm -f $D/STOP
TBIN=$SP/bin/seg_kl2 TAG=t2 TESTS="kl_knot_orders kl_fused_and_slabs kl_support_union kl_autotune_choice kl_knot_split_lists" $D/run_tests1.sh
if [ ! -e $D/t2_control.done ]; then
  KL_CONTROL=1 flock $SP/gpu.lock timeout 900 $SP/bin/seg_kl2 kl_support_union --exact --nocapture --test-threads=1 > $D/t2_control.log 2>&1; echo "exit $?" >> $D/t2_control.log; touch $D/t2_control.done
fi
/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout/run_kl3.sh
/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout/run_ncu48.sh
