#!/bin/bash
# seg_* GPU regression on frozen binaries (scratchpad/bin/reg), one log per binary under tests/reg_*, .done per binary
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
for b in hasty_nufft segnufft seg_vs seg_knot_kv seg_knot_hi seg_knot seg_knot2d seg_tile_block seg_stream seg_fft_traffic seg_pruned seg_multi seg_frames seg_kernel seg_b0_direct seg_wf seg_warp_gpu seg_joint seg_fat_union_gpu seg_kl seg_ra_vt seg_joint_chunks seg_split accuracy; do
  [ -e $D/STOP ] && exit 0
  [ -e $D/tests/reg_$b.done ] && continue
  L=$D/tests/reg_$b.log
  echo "=== $b ($(date '+%F %T')); power.limit $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)" > $L
  flock $SP/gpu.lock timeout 900 $SP/bin/gs_reg/$b --test-threads=1 >> $L 2>&1
  echo "exit $? ($(date '+%F %T'))" >> $L
  touch $D/tests/reg_$b.done
done
touch $D/tests/reg_ALL.done
