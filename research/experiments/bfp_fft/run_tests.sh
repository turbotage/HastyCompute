#!/bin/bash
# BFP step 1 correctness: hasty-fft bfp_store (dense and in-place layouts) + chanfft, hasty-nufft seg_fft_store (in
# place, and dense) + seg_fft_traffic (switch off, bfp22). Frozen binaries; one log per step, .done per step, STOP file
# stops before the next step; every step under the GPU lock with timeout 900; peak RSS from /usr/bin/time -v.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
B=$SP/bfi/bin
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/bfp_fft
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
run() {
  local name=$1; shift
  [ -e $D/STOP ] && { echo "STOP before $name"; exit 0; }
  [ -e $D/$name.done ] && return
  echo "=== $name: $* ($(date '+%F %T'))" > $D/$name.log
  flock $SP/gpu.lock bash -c "echo \"locked \$(date +%T); power limit \$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)\" >> $D/$name.log; /usr/bin/time -v timeout 900 $* >> $D/$name.log 2>&1; echo \"exit \$? \$(date +%T)\" >> $D/$name.log"
  touch $D/$name.done
}
run t_bfp_store $B/bfp_store --nocapture --test-threads 1
run t_chanfft $B/chanfft --nocapture --test-threads 1 pruned_matches_full ksupport_matches_full slab_passes_equal_whole_grid
run t_seg_fft_store $B/seg_fft_store --nocapture --test-threads 1
run t_seg_fft_store_dense env HASTY_SEG_FFT_BFP_LAYOUT=dense $B/seg_fft_store --nocapture --test-threads 1 bfp22_engine_matches_f32
run t_seg_fft_traffic_f32 $B/seg_fft_traffic --nocapture --test-threads 1
run t_seg_fft_traffic_bfp22 env HASTY_SEG_FFT_STORE=bfp22 $B/seg_fft_traffic --nocapture --test-threads 1
touch $D/tests.done
