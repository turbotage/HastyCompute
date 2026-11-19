#!/bin/bash
# §13 (research/sample_kernels.md): vs with the per-(sample, plane) weight table (vt, orders 2 / 3) against the
# production knot forward, and the order-0 tunables (VSHR, carve-out, VPF, VS2, VPT, TX 1 / VMINB) one at a time at
# the §12.6 shapes. Same pattern as run_vs.sh: flock on the session gpu.lock, timeout 900 s under the lock,
# out/<name>.done, STOP file. Binary: a frozen copy (another agent rebuilds hasty-nufft in parallel).
#   systemd-run --user --unit=proto-vs2 --collect -p MemoryMax=24G -p MemorySwapMax=0 -E PATH=$HOME/.cargo/bin:$PATH $PWD/research/experiments/sample_kernels/run_vt.sh
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/sample_kernels
O=$D/out
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
LOCK=$SP/gpu.lock
R=/home/turbotage/Documents/GitHub/HastyCompute/rustlib
BIN=${BIN:-$SP/bin/proto_gather_s13}
mkdir -p $O
cd $R
step() {   # step <name> <env> <args...>
  local n=$1 e=$2; shift 2
  [ -f $O/$n.done ] && return 0
  [ -f $D/STOP ] && exit 0
  echo "=== $n: env [$e] $* ($(date))" >> $O/$n.log
  if flock $LOCK env $e timeout 900 $BIN "$@" --json $O/results.jsonl --tag $n >> $O/$n.log 2>&1; then touch $O/$n.done; else echo "FAILED rc=$?" >> $O/$n.log; return 1; fi
}
# smoke gate (200 k samples): vt o2 / o3 (own and atomic epilogue, NWS 2, TX 4), o0 tunables
ok=1
step smoke_vt_a "" --mode fwd --m 200000 --ncb 10 --nseg 10 --order 2 --designs vs --each --check 300 --reps 2 --fixed "2,4,8,32,7,1,vt=1;4,4,8,16,1,2,vt=1" || ok=0
step smoke_vt_b "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 2 --designs vs --each --check 300 --reps 2 --fixed "2,4,8,16,1,2,vt=1;2,8,8,32,1,1,vt=1" || ok=0
step smoke_vt_c "" --mode fwd --m 200000 --ncb 10 --nseg 10 --order 3 --designs vs --each --check 300 --reps 2 --fixed "2,4,8,32,7,1,vt=1;4,4,8,16,1,2,vt=1" || ok=0
step smoke_vt_d "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 3 --designs vs --each --check 300 --reps 2 --fixed "2,4,8,16,1,2,vt=1;2,8,8,32,1,1,vt=1" || ok=0
step smoke_o0t_a "" --mode fwd --m 200000 --ncb 10 --nseg 10 --order 0 --designs vs --each --check 300 --reps 2 --fixed "4,4,16,64,7,1,shr=1;4,4,16,64,7,1,pf=1;4,4,16,64,7,1,pt=1;2,8,16,32,7,1,s2=1,shr=1,pf=1;1,16,16,32,7,1,shr=1,minb=2;4,4,16,64,7,1,carve=50" || ok=0
step smoke_o0t_b "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 0 --designs vs --each --check 300 --reps 2 --fixed "2,4,16,32,1,2,shr=1,pf=1;2,8,8,32,1,4,shr=1,s2=1,pt=1" || ok=0
grep -h "BEST" $O/smoke_vt_*.log $O/smoke_o0t_*.log | grep -oE "rel err [0-9.e+-]+" | awk '{ n++; if ($3 + 0 > 1e-5) bad = 1 } END { exit (bad || n < 14) }' || ok=0
[ $ok = 1 ] || { echo "SMOKE FAILED $(date)" >> $O/vt_driver.log; exit 1; }
echo "SMOKE OK $(date)" >> $O/vt_driver.log
# ---- Task A: vt autotune (order 2: vt 0 / 1 candidates; order 3: vt only) and same-session production references
step vt_m4000000_c10_l10_o2 "" --mode fwd --m 4000000 --ncb 10 --nseg 10 --order 2 --designs vs --check 400 --reps 5
step vt_m1300000_c10_l10_o2 "" --mode fwd --m 1300000 --ncb 10 --nseg 10 --order 2 --designs vs --check 400 --reps 5
step vt_m4000000_c10_l10_o3 "" --mode fwd --m 4000000 --ncb 10 --nseg 10 --order 3 --designs vs --check 400 --reps 5
step vt_m1300000_c10_l10_o3 "" --mode fwd --m 1300000 --ncb 10 --nseg 10 --order 3 --designs vs --check 400 --reps 5
step knotref2_m4000000_c10_l10_o2 "" --mode base --knot --m 4000000 --ncb 10 --nseg 10 --order 2
step knotref2_m4000000_c10_l10_o3 "" --mode base --knot --m 4000000 --ncb 10 --nseg 10 --order 3
step knotref2_m1300000_c10_l10_o2 "" --mode base --knot --m 1300000 --ncb 10 --nseg 10 --order 2
step knotref2_m1300000_c10_l10_o3 "" --mode base --knot --m 1300000 --ncb 10 --nseg 10 --order 3
# ---- Task B: order-0 tunables one at a time at the §12.6 shapes (one invocation per case: same session, same data)
ab() {   # ab <name> <m> <ncb> <nseg> <shape> <tx1 shape> <minb>
  local n=$1 m=$2 c=$3 l=$4 s=$5 t1=$6 mb=$7
  step $n "" --mode fwd --m $m --ncb $c --nseg $l --order 0 --designs vs --each --check 400 --reps 5 \
    --fixed "$s;$s,shr=1;$s,carve=50;$s,carve=0;$s,pf=1;$s,s2=1;$s,pt=1;$t1;$t1,shr=1;$t1,shr=1,minb=$mb"
}
ab ab_m4000000_c48_l1 4000000 48 1 "2,4,16,32,3,1" "1,16,16,32,3,1" 5
ab ab_m4000000_c10_l10 4000000 10 10 "4,4,16,64,7,1" "1,16,16,32,7,1" 2
ab ab_m4000000_c16_l1 4000000 16 1 "4,4,16,32,1,1" "1,16,16,32,1,1" 16
ab ab_m1300000_c10_l10 1300000 10 10 "4,4,16,64,7,1" "1,16,16,32,7,1" 2
ab ab_m4000000_c16_l12 4000000 16 12 "4,4,16,64,4,1" "1,16,16,32,4,1" 4
echo "VT DONE $(date)" >> $O/vt_driver.log
