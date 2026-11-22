#!/bin/bash
O=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/phase2
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
while systemctl --user is-active --quiet p2-tests1; do sleep 20; done
$O/run_tests.sh $SP/bin/seg_phase2_v2 t2 p2_ra_static p2_ra_knot p2_kl_knot p2_vs_subchunks p2_spill_screen p2_autotune
$O/run_ob.sh
$O/run_ncu.sh
$O/run_san.sh
touch $O/chain1.done
