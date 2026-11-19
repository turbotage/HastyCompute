#!/bin/bash
# vs forward in the engine (research/sample_kernels.md §14): operator-level opbench, vs off / on (--vs-compare), static
# and knot-major paths, at the §10 / §12 shapes. Resumable: one .done file per case; touch STOP to end after the case.
#   static sets = frame × encode of the 10-min flow_thorax scan: --frames 7 → ≈ 4.3 M, --frames 23 → ≈ 1.3 M samples
#   knot sets = one encode, all frames (cubic, 20 knots): --scan-min 1.33 → ≈ 4 M, --scan-min 0.44 → ≈ 1.3 M samples
# --max-grid-gb 10: 10 coils × 10 segments in one coil batch (as the prototype); 48 × 1 with --cb-cap 48 (one batch).
set -u
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/vs_engine
L=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
BIN=target/release/opbench
COMMON="--seq flow --sets 1 --iters 1 --reps 5 --max-grid-gb 10 --vs-compare"
run() {
  local name=$1; shift
  [ -f $OUT/STOP ] && exit 0
  [ -f $OUT/$name.done ] && return
  echo "$(date +%T) start $name: $*" >> $OUT/queue.log
  if timeout 900 flock $L $BIN $COMMON "$@" --json $OUT/$name.json > $OUT/$name.log 2>&1; then
    touch $OUT/$name.done
    echo "$(date +%T) done $name" >> $OUT/queue.log
  else
    echo "$(date +%T) FAILED $name (exit $?)" >> $OUT/queue.log
  fi
}
# compute-sanitizer memcheck of the new kernels at small size (tests/seg_vs.rs `vs_env_case`: every valid launch shape
# of the case; static / knot 1-D / knot 2-D, orders 0 and 1)
san() {
  local name=$1 dbg=$2
  [ -f $OUT/STOP ] && exit 0
  [ -f $OUT/$name.done ] && return
  local T=$(ls -t target/release/deps/seg_vs-* | grep -v '\.d$' | head -1)
  echo "$(date +%T) start $name: VS_DBG=$dbg" >> $OUT/queue.log
  if HASTY_NVRTC_LINEINFO=1 VS_DBG=$dbg timeout 900 flock $L /usr/local/cuda/bin/compute-sanitizer --tool memcheck --leak-check no $T vs_env_case --nocapture > $OUT/$name.log 2>&1; then
    touch $OUT/$name.done; echo "$(date +%T) done $name" >> $OUT/queue.log
  else
    echo "$(date +%T) FAILED $name (exit $?)" >> $OUT/queue.log
  fi
}
san san_static_o0 0,6,4,0,10,3000,6,4,4,16,64
san san_static_o1 0,6,4,1,10,3000,6,2,4,16,32
san san_knot1d_o0 1,4,3,0,10,3000,4,4,4,16,64
san san_knot2d_o1 2,4,3,1,8,3000,4,4,2,8,48
run st_4m_10x10_o0   --models seg-b0   --nseg 10 --vcoils 10 --order 0 --frames 7
run st_4m_48x1_o0    --models seg-none --vcoils 48 --cb-cap 48 --frames 7
run st_1m3_10x10_o0  --models seg-b0   --nseg 10 --vcoils 10 --order 0 --frames 23
run kn_4m_10x10_o0   --models seg-b0   --nseg 10 --vcoils 10 --order 0 --temporal cubic --knot --frames 20 --scan-min 1.33
run kn_4m_48x1_o0    --models seg-none --vcoils 48 --cb-cap 48 --temporal cubic --knot --frames 20 --scan-min 1.33
run kn_1m3_10x10_o0  --models seg-b0   --nseg 10 --vcoils 10 --order 0 --temporal cubic --knot --frames 20 --scan-min 0.44
run st_4m_10x10_o1   --models seg-full --nseg 10 --vcoils 10 --order 1 --frames 7
run kn_4m_10x10_o1   --models seg-full --nseg 10 --vcoils 10 --order 1 --temporal cubic --knot --frames 20 --scan-min 1.33
touch $OUT/all.done
