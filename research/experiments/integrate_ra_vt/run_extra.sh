#!/bin/bash
# rerun of vt_static_order2 (shape-count assertion fixed) and memcheck with split tile lists (HASTY_SEG_MAXITEM=64)
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/integrate_ra_vt
cd $SP
if [ ! -e $OUT/test_vt_static_order2b.done ]; then
  flock $SP/gpu.lock timeout 900 $SP/bin/seg_ra_vt2 vt_static_order2 --exact --nocapture --test-threads=1 > $OUT/test_vt_static_order2b.log 2>&1; echo "exit $?" >> $OUT/test_vt_static_order2b.log; touch $OUT/test_vt_static_order2b.done
fi
if [ ! -e $OUT/san_split.done ]; then
  RA_SAN=1 HASTY_SEG_MAXITEM=64 HASTY_NVRTC_LINEINFO=1 flock $SP/gpu.lock timeout 900 /usr/local/cuda/bin/compute-sanitizer --tool memcheck $SP/bin/seg_ra_vt2 ra_vt_sanitizer_case --exact --nocapture --test-threads=1 > $OUT/san_ra_vt_split.log 2>&1; echo "exit $?" >> $OUT/san_ra_vt_split.log; touch $OUT/san_split.done
fi
touch $OUT/extra.done
