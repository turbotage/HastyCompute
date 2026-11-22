#!/bin/bash
# after gs-bp: slabs off (HASTY_SEG_FFT_SLAB=0) on four b/ cases, same binary
while systemctl --user is-active --quiet gs-bp; do sleep 30; done
HASTY_SEG_FFT_SLAB=0 BIN=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/bin/opbench_gs2 TAG=s CASES_FILE=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes/cases_s.txt /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes/run_cases.sh
