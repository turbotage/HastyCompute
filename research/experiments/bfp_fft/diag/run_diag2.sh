#!/bin/bash
# diagnosis 2: segment basis (fixed-node interpolators vs SVD) and the operator's one-ulp sensitivity; static order 0
S=/home/turbotage/Documents/GitHub/HastyCompute/data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
export RAYON_NUM_THREADS=8
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
for bs in fixed svd; do
  L=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/bfp_fft/diag/st_o0_basis_$bs.log
  [ -e $L.done ] && continue
  echo "=== basis $bs" > $L
  flock /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock bash -c "echo locked \$(date +%T); /usr/bin/time -v timeout 900 /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/bfi/bfp_phantom_d2 $S --order 0 --basis $bs --stores bfp16,bfp22; echo exit \$?" >> $L 2>&1
  touch $L.done
done
