#!/bin/bash
# §13 final: two-stage vs autotune (shape search + vs_refine tunables) for the 5 order-0 cases and the order-2/3
# 10 x 10 cases; compute-sanitizer memcheck of the new paths. Same pattern as run_vt.sh.
#   systemd-run --user --unit=proto-vs2c --collect -p MemoryMax=24G -p MemorySwapMax=0 $PWD/research/experiments/sample_kernels/run_vt3.sh
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/sample_kernels
O=$D/out
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
LOCK=$SP/gpu.lock
R=/home/turbotage/Documents/GitHub/HastyCompute/rustlib
BIN=${BIN:-$SP/bin/proto_gather_s13d}
SAN=/usr/local/cuda/bin/compute-sanitizer
cd $R
step() {
  local n=$1 e=$2; shift 2
  [ -f $O/$n.done ] && return 0
  [ -f $D/STOP ] && exit 0
  echo "=== $n: env [$e] $* ($(date))" >> $O/$n.log
  if flock $LOCK env $e timeout 900 $BIN "$@" --json $O/results.jsonl --tag $n >> $O/$n.log 2>&1; then touch $O/$n.done; else echo "FAILED rc=$?" >> $O/$n.log; return 1; fi
}
san() {
  local n=$1 e=$2; shift 2
  [ -f $O/$n.done ] && return 0
  [ -f $D/STOP ] && exit 0
  echo "=== $n: env [$e] memcheck $* ($(date))" >> $O/$n.log
  if flock $LOCK env $e timeout 900 $SAN --tool memcheck --print-limit 20 $BIN "$@" >> $O/$n.log 2>&1; then touch $O/$n.done; else echo "FAILED rc=$?" >> $O/$n.log; return 1; fi
}
san san_vt_o3 "HASTY_NVRTC_LINEINFO=1" --mode fwd --m 200000 --ncb 10 --nseg 10 --order 3 --designs vs --each --check 50 --reps 1 --fixed "4,4,8,48,7,1,vt=1,vtp=1,shr=1,pt=1;2,4,8,16,1,2,vt=1"
san san_vs_o0t "HASTY_NVRTC_LINEINFO=1" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 0 --designs vs --each --check 50 --reps 1 --fixed "2,8,16,32,1,2,shr=1,pf=1,s2=1,pt=1;4,4,16,32,1,1,shr=1,pt=1,s2=1"
for c in "4000000 48 1 0" "4000000 10 10 0" "4000000 16 1 0" "1300000 10 10 0" "4000000 16 12 0" "4000000 10 10 2" "1300000 10 10 2" "4000000 10 10 3" "1300000 10 10 3"; do
  set -- $c
  step fin_m$1_c$2_l$3_o$4 "" --mode fwd --m $1 --ncb $2 --nseg $3 --order $4 --designs vs --check 800 --reps 5
done
echo "VT3 DONE $(date)" >> $O/vt_driver.log
