#!/bin/bash
# existing seg_* GPU regression (+ seg_ra_vt) with the knot-path layout in the build (default HASTY_SEG_KL=auto: off
# until an autotune keeps it). Frozen binaries, one per invocation, .done per binary, STOP to stop.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout
while systemctl --user is-active --quiet kl-chain3 || systemctl --user is-active --quiet kl-chain4; do sleep 20; done
cd $SP
for t in ${TESTS:-seg_ra_vt seg_vs seg_knot_kv seg_knot_hi seg_knot seg_knot2d seg_tile_block seg_stream segnufft seg_fft_traffic seg_pruned seg_multi seg_frames seg_kernel seg_b0_direct seg_wf seg_warp_gpu seg_joint seg_fat_union_gpu seg_joint_chunks}; do
  [ -e $OUT/STOP ] && exit 0
  [ -e $OUT/reg_$t.done ] && continue
  flock $SP/gpu.lock timeout 900 $SP/bin/reg2/$t --test-threads=1 > $OUT/reg_$t.log 2>&1
  echo "exit $?" >> $OUT/reg_$t.log
  touch $OUT/reg_$t.done
done
touch $OUT/regress.done
