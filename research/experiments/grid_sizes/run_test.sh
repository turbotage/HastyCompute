#!/bin/bash
# one test binary (TBIN) with optional filter (FILT) under the GPU lock; log tests/$TAG.log, .done
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
L=$D/tests/$TAG.log
echo "=== $TAG: $TBIN $FILT ($(date '+%F %T')); power.limit $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)" > $L
flock $SP/gpu.lock timeout 900 ${PRE} $TBIN $FILT --nocapture --test-threads=1 >> $L 2>&1
echo "exit $? ($(date '+%F %T'))" >> $L
touch $D/tests/$TAG.done
