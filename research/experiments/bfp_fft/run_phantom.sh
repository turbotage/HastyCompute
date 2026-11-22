#!/bin/bash
# accuracy on ph0001 (research/bfp_storage.md §8): bfp_phantom, static (one frame / encode set) and 2-D knots 3:1,8:3
# (340 readouts of encode 3 over the scan), orders 0 and 3, 10 virtual coils × 10 segments, bfp22 / bfp20 / bfp24
# against f32 on one engine. Host memory ≈ 6–10 GB (32 true maps, physics grid, tables).
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/bfp_fft
PB=$SP/bfi/bfp_phantom
S=/home/turbotage/Documents/GitHub/HastyCompute/data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
export RAYON_NUM_THREADS=8
run() {
  local name=$1; shift
  [ -e $D/STOP ] && { echo "STOP before $name"; exit 0; }
  [ -e $D/$name.done ] && return
  echo "=== $name: $* ($(date '+%F %T'))" > $D/$name.log
  flock $SP/gpu.lock bash -c "echo \"locked \$(date +%T); power limit \$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)\" >> $D/$name.log; /usr/bin/time -v timeout 900 $* >> $D/$name.log 2>&1; echo \"exit \$? \$(date +%T)\" >> $D/$name.log"
  touch $D/$name.done
}
run ph_static_o0 $PB $S --order 0 --stores bfp22,bfp20,bfp24,bfp16
run ph_static_o3 $PB $S --order 3 --stores bfp22,bfp20,bfp24,bfp16
run ph_k2_o0 $PB $S --order 0 --knots2d 3:1,8:3 --readouts 340 --stores bfp22,bfp20,bfp24,bfp16
run ph_k2_o3 $PB $S --order 3 --knots2d 3:1,8:3 --readouts 340 --stores bfp22,bfp20,bfp24,bfp16
touch $D/phantom.done
