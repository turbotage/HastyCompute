#!/bin/bash
# §13: vt A/B at the tuned shapes: VTP (software-pipelined table build), VSHR, TZ 16, TX 2 vs 4; smoke gate for VTP.
# Same pattern as run_vt.sh (flock gpu.lock, timeout 900 under the lock, .done files, STOP file).
#   systemd-run --user --unit=proto-vs2b --collect -p MemoryMax=24G -p MemorySwapMax=0 $PWD/research/experiments/sample_kernels/run_vt2.sh
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/sample_kernels
O=$D/out
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
LOCK=$SP/gpu.lock
R=/home/turbotage/Documents/GitHub/HastyCompute/rustlib
BIN=${BIN:-$SP/bin/proto_gather_s13c}
cd $R
step() {
  local n=$1 e=$2; shift 2
  [ -f $O/$n.done ] && return 0
  [ -f $D/STOP ] && exit 0
  echo "=== $n: env [$e] $* ($(date))" >> $O/$n.log
  if flock $LOCK env $e timeout 900 $BIN "$@" --json $O/results.jsonl --tag $n >> $O/$n.log 2>&1; then touch $O/$n.done; else echo "FAILED rc=$?" >> $O/$n.log; return 1; fi
}
ok=1
step smoke_vtp_a "" --mode fwd --m 200000 --ncb 10 --nseg 10 --order 2 --designs vs --each --check 300 --reps 2 --fixed "4,4,8,48,7,1,vt=1,vtp=1;2,8,8,16,1,2,vt=1,vtp=1,shr=1" || ok=0
step smoke_vtp_b "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 3 --designs vs --each --check 300 --reps 2 --fixed "2,4,8,16,1,2,vt=1,vtp=1;4,4,16,32,1,1,vt=1,vtp=1,shr=1" || ok=0
step smoke_vtp_c "" --mode fwd --m 200000 --ncb 10 --nseg 10 --order 3 --designs vs --each --check 300 --reps 2 --fixed "2,4,8,48,7,1,vt=1,vtp=1" || ok=0
grep -h "BEST" $O/smoke_vtp_*.log | grep -oE "rel err [0-9.e+-]+" | awk '{ n++; if ($3 + 0 > 1e-5) bad = 1 } END { exit (bad || n < 5) }' || ok=0
[ $ok = 1 ] || { echo "VTP SMOKE FAILED $(date)" >> $O/vt_driver.log; exit 1; }
echo "VTP SMOKE OK $(date)" >> $O/vt_driver.log
for c in "4000000 2" "1300000 2" "4000000 3" "1300000 3"; do
  set -- $c
  step vtab_m$1_c10_l10_o$2 "" --mode fwd --m $1 --ncb 10 --nseg 10 --order $2 --designs vs --each --check 400 --reps 5 \
    --fixed "4,4,8,48,7,1,vt=1;4,4,8,48,7,1,vt=1,vtp=1;4,4,8,48,7,1,vt=1,shr=1;4,4,8,48,7,1,vt=1,vtp=1,shr=1;2,4,8,48,7,1,vt=1;2,4,8,48,7,1,vt=1,vtp=1;2,4,8,48,7,1,vt=1,vtp=1,shr=1;4,4,16,48,7,1,vt=1;4,4,16,48,7,1,vt=1,vtp=1;2,8,16,48,7,1,vt=1,vtp=1,shr=1"
done
echo "VT2 DONE $(date)" >> $O/vt_driver.log
