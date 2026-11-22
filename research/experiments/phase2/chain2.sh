#!/bin/bash
O=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/phase2
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
$O/run_tests_split.sh $SP/bin/seg_phase2_v4 t4
$O/run_ob.sh
$O/run_ncu.sh
B=$SP/bin/seg_phase2_v4 $O/run_san.sh
$O/run_reg.sh
touch $O/chain2.done
