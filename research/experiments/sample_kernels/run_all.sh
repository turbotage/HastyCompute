#!/bin/bash
# Detached, resumable driver of the sample-kernel prototype (research/sample_kernels.md). Each step is one locked
# GPU invocation (flock on the session gpu.lock); a step is skipped when out/<name>.done exists.
#   systemd-run --user --unit=proto-gather --collect -p MemoryMax=24G -p MemorySwapMax=0 $PWD/research/experiments/sample_kernels/run_all.sh
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/sample_kernels
O=$D/out
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
BIN=/home/turbotage/Documents/GitHub/HastyCompute/rustlib/target/release/proto_gather
mkdir -p $O
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
step() {   # step <name> <args...>
  local n=$1; shift
  [ -f $O/$n.done ] && return
  [ -f $D/STOP ] && exit 0
  echo "=== $n: $*" >> $O/$n.log
  if timeout 1500 flock $LOCK $BIN "$@" --json $O/results.jsonl >> $O/$n.log 2>&1; then touch $O/$n.done; else echo "FAILED rc=$?" >> $O/$n.log; fi
}
# 1. correctness smoke (small M, fixed shapes; sb 1 and 4)
step smoke_fwd_o0 --mode fwd --m 200000 --ncb 4 --nseg 2 --order 0 --designs halo --fixed 4,4,2,128,64,0,1,1 --check 300 --reps 2
step smoke_fwd_o0_sb --mode fwd --m 200000 --ncb 4 --nseg 2 --order 0 --designs halo,gm --fixed 4,4,2,128,64,0,4,2 --check 300 --reps 2
step smoke_fwd_o2_sb --mode fwd --m 200000 --ncb 4 --nseg 2 --order 2 --designs halo,gm --fixed 4,4,2,128,64,0,2,2 --check 300 --reps 2
step smoke_adj --mode adj --m 200000 --ncb 4 --nseg 2 --order 2 --fixed 4,4,2,128,64,0,1,1 --reps 2
# 2. today's engine (kernel times from its autotune), 3. prototypes (autotuned), per case
for c in "4000000 16 1 0" "4000000 10 10 0" "4000000 48 1 0" "4000000 16 12 0" "4000000 10 10 2" "1300000 16 1 0" "1300000 10 10 0" "1300000 48 1 0" "1300000 192 1 0" "1300000 10 10 2"; do
  set -- $c
  tag=m$1_c$2_l$3_o$4
  step base_$tag --mode base --m $1 --ncb $2 --nseg $3 --order $4
  step fwd_$tag --mode fwd --m $1 --ncb $2 --nseg $3 --order $4 --designs halo,gm --check 800 --reps 5
done
# 4. shared-memory spreading adjoint (expected to lose: global atomics per halo point)
step adj_m4000000_c16_l1_o0 --mode adj --m 4000000 --ncb 16 --nseg 1 --order 0 --reps 3
step adj_m4000000_c10_l10_o0 --mode adj --m 4000000 --ncb 10 --nseg 10 --order 0 --reps 3
echo ALL DONE >> $O/driver.log
