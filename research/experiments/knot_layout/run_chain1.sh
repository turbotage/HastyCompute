#!/bin/bash
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
export BIN=$SP/bin/opbench_kl1
CASES="k2_lo" $D/run_base.sh
$D/run_ncu.sh
CASES="k1_hi k2_hi" $D/run_base.sh
