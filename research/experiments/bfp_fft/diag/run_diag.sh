#!/bin/bash
# diagnosis of the b-independent error floor on ph0001 (static, order 0): in place vs dense, k-support pruning on / off
S=/home/turbotage/Documents/GitHub/HastyCompute/data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
export RAYON_NUM_THREADS=8
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
for v in "inplace 1" "dense 1" "inplace 0" "dense 0"; do
  set -- $v
  L=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/bfp_fft/diag/st_o0_$1_kp$2.log
  [ -e $L.done ] && continue
  echo "=== layout $1 kprune $2" > $L
  flock /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock bash -c "echo locked \$(date +%T); HASTY_SEG_FFT_BFP_LAYOUT=$1 /usr/bin/time -v timeout 900 /tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/bfi/bfp_phantom_d $S --order 0 --kprune $2 --stores bfp22,bfp24,bfp16; echo exit \$?" >> $L 2>&1
  touch $L.done
done
