#!/bin/bash
# §17 tests of the knot-path skip (tests/seg_kl.rs, frozen binary), then compute-sanitizer memcheck. One test per
# invocation, .done per test, STOP to stop.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout
BIN=${TBIN:-$SP/bin/seg_kl1}
TAG=${TAG:-t1}
cd $SP
for t in ${TESTS:-kl_knot_orders kl_knot_split_lists kl_fused_and_slabs kl_support_union kl_autotune_choice}; do
  [ -e $OUT/STOP ] && exit 0
  [ -e $OUT/${TAG}_$t.done ] && continue
  flock $SP/gpu.lock timeout 900 $BIN $t --exact --nocapture --test-threads=1 > $OUT/${TAG}_$t.log 2>&1
  echo "exit $?" >> $OUT/${TAG}_$t.log
  touch $OUT/${TAG}_$t.done
done
for t in kl_sanitizer_case kl_support_union; do
  [ -e $OUT/${TAG}_san_$t.done ] && continue
  KL_SAN=1 HASTY_NVRTC_LINEINFO=1 flock $SP/gpu.lock timeout 900 /usr/local/cuda/bin/compute-sanitizer --tool memcheck $BIN $t --exact --nocapture --test-threads=1 > $OUT/${TAG}_san_$t.log 2>&1
  echo "exit $?" >> $OUT/${TAG}_san_$t.log
  touch $OUT/${TAG}_san_$t.done
done
