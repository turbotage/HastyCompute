#!/bin/bash
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
for i in 1 2 3; do echo "== run $i $(date +%T) power $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"; flock /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock timeout 600 /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/bin/gs_reg/seg_warp_gpu warp_adjoint_gpu --exact --nocapture --test-threads=1 2>&1 | grep -E "mismatch|test result"; done > /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes/tests/warp3.log
touch /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes/tests/warp3.done
