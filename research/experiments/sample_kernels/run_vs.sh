#!/bin/bash
# Design 6 `vs` (value-stationary rows, research/sample_kernels.md §12): smoke gate, autotune per case, same-session ps
# (fixed §11.6 shapes) and production knot-kernel references, compute-sanitizer memcheck of the knot kernels and of
# ps / vs. Same pattern as run_ps_fix.sh: flock on the session gpu.lock, timeout 900 s under the lock,
# out/<name>.done, STOP file.
#   systemd-run --user --unit=proto-vs --collect -p MemoryMax=24G -p MemorySwapMax=0 -E PATH=$HOME/.cargo/bin:$PATH $PWD/research/experiments/sample_kernels/run_vs.sh
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/sample_kernels
O=$D/out
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
R=/home/turbotage/Documents/GitHub/HastyCompute/rustlib
BIN=$R/target/release/proto_gather
SAN=/usr/local/cuda/bin/compute-sanitizer
mkdir -p $O
cd $R
step() {   # step <name> <env> <args...>
  local n=$1 e=$2; shift 2
  [ -f $O/$n.done ] && return 0
  [ -f $D/STOP ] && exit 0
  echo "=== $n: env [$e] $* ($(date))" >> $O/$n.log
  if flock $LOCK env $e timeout 900 $BIN "$@" --json $O/results.jsonl --tag $n >> $O/$n.log 2>&1; then touch $O/$n.done; else echo "FAILED rc=$?" >> $O/$n.log; return 1; fi
}
san() {   # san <name> <env> <args...>: compute-sanitizer memcheck (no json)
  local n=$1 e=$2; shift 2
  [ -f $O/$n.done ] && return 0
  [ -f $D/STOP ] && exit 0
  echo "=== $n: env [$e] memcheck $* ($(date))" >> $O/$n.log
  if flock $LOCK env $e timeout 900 $SAN --tool memcheck --print-limit 20 $BIN "$@" >> $O/$n.log 2>&1; then touch $O/$n.done; else echo "FAILED rc=$?" >> $O/$n.log; return 1; fi
}
# smoke gate (200 k samples): own / atomic epilogue, NWS > 1, VXPAD, orders 0 / 2
ok=1
step smoke_vs_a "" --mode fwd --m 200000 --ncb 16 --nseg 1 --order 0 --designs vs --fixed 2,8,8,64,1,1 --check 300 --reps 2 || ok=0
step smoke_vs_b "" --mode fwd --m 200000 --ncb 10 --nseg 10 --order 0 --designs vs --fixed 4,4,8,32,1,1 --check 300 --reps 2 || ok=0
step smoke_vs_c "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 0 --designs vs --fixed 2,4,8,32,1,2,1 --check 300 --reps 2 || ok=0
step smoke_vs_d "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 2 --designs vs --fixed 2,4,8,32,1,1,0 --check 300 --reps 2 || ok=0
step smoke_vs_e "" --mode fwd --m 200000 --ncb 10 --nseg 10 --order 2 --designs vs --fixed 2,4,8,16,1,2,1 --check 300 --reps 2 || ok=0
grep -h "BEST" $O/smoke_vs_*.log | grep -oE "rel err [0-9.e+-]+" | awk '{ if ($3 + 0 > 1e-5) bad = 1 } END { exit bad }' || ok=0
[ $ok = 1 ] || { echo "SMOKE FAILED $(date)" >> $O/vs_driver.log; exit 1; }
# vs autotune per case
for c in "4000000 48 1 0" "4000000 10 10 0" "4000000 16 1 0" "1300000 10 10 0" "4000000 16 12 0" "4000000 10 10 2" "1300000 10 10 2"; do
  set -- $c
  step vs_m$1_c$2_l$3_o$4 "" --mode fwd --m $1 --ncb $2 --nseg $3 --order $4 --designs vs --check 800 --reps 5
done
# same-session references: ps at the §11.6 tuned shapes, production knot kernels (variants 0 / 1 autotuned)
step psref_m4000000_c48_l1_o0 "" --mode fwd --m 4000000 --ncb 48 --nseg 1 --order 0 --designs ps --fixed 4,4,8,4,2,3,256,0 --check 400 --reps 5
step psref_m4000000_c10_l10_o0 "" --mode fwd --m 4000000 --ncb 10 --nseg 10 --order 0 --designs ps --fixed 4,4,8,2,2,2,128,1 --check 400 --reps 5
step psref_m4000000_c16_l1_o0 "" --mode fwd --m 4000000 --ncb 16 --nseg 1 --order 0 --designs ps --fixed 4,4,8,4,2,3,256,0 --check 400 --reps 5
step psref_m1300000_c10_l10_o0 "" --mode fwd --m 1300000 --ncb 10 --nseg 10 --order 0 --designs ps --fixed 4,8,8,2,2,1,256,1 --check 400 --reps 5
step psref_m4000000_c16_l12_o0 "" --mode fwd --m 4000000 --ncb 16 --nseg 12 --order 0 --designs ps --fixed 4,4,8,8,2,3,256,1 --check 400 --reps 5
step knotref_m4000000_c10_l10_o0 "" --mode base --knot --m 4000000 --ncb 10 --nseg 10 --order 0
step knotref_m4000000_c10_l10_o2 "" --mode base --knot --m 4000000 --ncb 10 --nseg 10 --order 2
# compute-sanitizer memcheck (small M): production knot kernels (variant 1 via HASTY_SEG_KV=1), ps, vs
san san_knot_o0 "HASTY_SEG_KV=1 HASTY_NVRTC_LINEINFO=1" --mode base --knot --m 200000 --ncb 10 --nseg 10 --order 0
san san_knot_o2 "HASTY_SEG_KV=1 HASTY_NVRTC_LINEINFO=1" --mode base --knot --m 200000 --ncb 10 --nseg 10 --order 2
san san_ps_o0 "HASTY_NVRTC_LINEINFO=1" --mode fwd --m 200000 --ncb 10 --nseg 10 --order 0 --designs ps --fixed 4,4,8,2,2,2,128,1 --check 50 --reps 1
san san_vs_o0 "HASTY_NVRTC_LINEINFO=1" --mode fwd --m 200000 --ncb 10 --nseg 10 --order 0 --designs vs --fixed 2,8,8,64,1,2 --check 50 --reps 1
san san_vs_o2 "HASTY_NVRTC_LINEINFO=1" --mode fwd --m 200000 --ncb 10 --nseg 10 --order 2 --designs vs --fixed 2,4,8,16,1,2,1 --check 50 --reps 1
echo "VS DONE $(date)" >> $O/vs_driver.log
