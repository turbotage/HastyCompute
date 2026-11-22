#!/bin/bash
O=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/phase2
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
B=$SP/bin/seg_phase2_v5 $O/run_san2.sh
$O/run_reg.sh
touch $O/chain4.done
