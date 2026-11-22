#!/bin/bash
# §15: ra adjoint autotune per case, production references (knot kernels on one knot, static base), memcheck.
#   systemd-run --user --unit=adj-ra --collect -p MemoryMax=24G -p MemorySwapMax=0 -E PATH=$HOME/.cargo/bin:$PATH $PWD/research/experiments/adjoint/run_ra.sh
# Steps write out/<step>.log and rows to results.jsonl; finished steps (.done) are skipped; touch STOP to stop between steps.
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/adjoint
O=$D/out
mkdir -p $O
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
LOCK=$SP/gpu.lock
BIN=${BIN:-$SP/bin/proto_gather_ra1}
SAN=/usr/local/cuda/bin/compute-sanitizer
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
step() {
  local n=$1 e=$2; shift 2
  [ -f $O/$n.done ] && return 0
  [ -f $D/STOP ] && exit 0
  echo "=== $n: env [$e] $* ($(date))" >> $O/$n.log
  if flock $LOCK env $e timeout 900 $BIN "$@" --json $D/results.jsonl --tag $n >> $O/$n.log 2>&1; then touch $O/$n.done; else echo "FAILED rc=$?" >> $O/$n.log; fi
}
san() {
  local n=$1 e=$2; shift 2
  [ -f $O/$n.done ] && return 0
  [ -f $D/STOP ] && exit 0
  echo "=== $n: env [$e] memcheck $* ($(date))" >> $O/$n.log
  if flock $LOCK env $e timeout 900 $SAN --tool memcheck --print-limit 20 $BIN "$@" >> $O/$n.log 2>&1; then touch $O/$n.done; else echo "FAILED rc=$?" >> $O/$n.log; fi
}
san san_ra_o0 "HASTY_NVRTC_LINEINFO=1" --mode adj --designs ra --m 100000 --ncb 4 --nseg 3 --order 0 --check 50 --reps 1 --each --fixed "4,4,4,2,4,16,2048;2,4,8,2,1,16,64;4,4,4,1,4,32,2048"
san san_ra_o3 "HASTY_NVRTC_LINEINFO=1" --mode adj --designs ra --m 100000 --ncb 4 --nseg 3 --order 3 --check 50 --reps 1 --each --fixed "4,4,4,2,4,16,2048;2,4,8,2,1,16,64;4,4,4,1,4,32,2048"
for c in "4000000 10 10 0" "4000000 10 10 3" "4000000 48 1 0" "4000000 16 1 0" "1300000 10 10 0" "4000000 16 12 0" "4000000 10 10 2" "1300000 10 10 2" "1300000 10 10 3"; do
  set -- $c
  step ra_m$1_c$2_l$3_o$4 "" --mode adj --designs ra --m $1 --ncb $2 --nseg $3 --order $4 --check 1000 --reps 5
  step knot_m$1_c$2_l$3_o$4 "" --mode base --knot --m $1 --ncb $2 --nseg $3 --order $4
done
step base_m4000000_c48_l1_o0 "" --mode base --m 4000000 --ncb 48 --nseg 1 --order 0
step base_m4000000_c10_l10_o0 "" --mode base --m 4000000 --ncb 10 --nseg 10 --order 0
echo "RA QUEUE DONE $(date)" >> $D/queue.log
touch $D/all.done
