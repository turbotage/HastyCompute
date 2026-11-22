#!/bin/bash
# existing seg_* GPU regression with the new defaults (ra / vt in the autotune), then compute-sanitizer memcheck on the
# small ra / vt case. One binary per invocation, .done per binary, STOP to stop.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/integrate_ra_vt
cd $SP
for t in ${TESTS:-seg_vs seg_knot_kv seg_knot_hi seg_knot seg_knot2d seg_tile_block seg_stream segnufft seg_fft_traffic seg_pruned seg_multi seg_frames seg_kernel seg_b0_direct seg_wf seg_warp_gpu seg_joint seg_fat_union_gpu seg_joint_chunks}; do
  [ -e $OUT/STOP ] && exit 0
  [ -e $OUT/reg_$t.done ] && continue
  flock $SP/gpu.lock timeout 900 $SP/bin/reg/$t --test-threads=1 > $OUT/reg_$t.log 2>&1
  echo "exit $?" >> $OUT/reg_$t.log
  touch $OUT/reg_$t.done
done
if [ ! -e $OUT/san.done ]; then
  RA_SAN=1 HASTY_NVRTC_LINEINFO=1 flock $SP/gpu.lock timeout 900 /usr/local/cuda/bin/compute-sanitizer --tool memcheck $SP/bin/seg_ra_vt ra_vt_sanitizer_case --exact --nocapture --test-threads=1 > $OUT/san_ra_vt.log 2>&1
  echo "exit $?" >> $OUT/san_ra_vt.log
  touch $OUT/san.done
fi
touch $OUT/regress.done
