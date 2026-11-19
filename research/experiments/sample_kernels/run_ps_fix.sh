#!/bin/bash
# Fix probes for the ps limiter found by run_ps_diag.sh (research/sample_kernels.md §11): flat-z home tiles (every warp
# active in every streamed plane), PS_PFS (plane copy interleaved with the row loop: fewer registers), PS_MINB
# (register cap), PS_ACFG (segment coefficients out of the shared sample record), FSB 3. Fixed shapes
# tx,ty,tz,ncg,fc,fsb,nt,acfg,pfs,minb; then (stage 2, `AUTOTUNE=1`) the autotune over the extended candidate set.
# Same pattern as run_ps.sh: flock on the session gpu.lock, timeout 900 s under the lock, out/<name>.done, STOP file.
#   systemd-run --user --unit=proto-fix --collect -p MemoryMax=24G -p MemorySwapMax=0 -E PATH=$HOME/.cargo/bin:$PATH $PWD/research/experiments/sample_kernels/run_ps_fix.sh
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/sample_kernels
O=$D/out
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
R=/home/turbotage/Documents/GitHub/HastyCompute/rustlib
BIN=$R/target/release/proto_gather
mkdir -p $O
cd $R
step() {   # step <name> <env> <args...>
  local n=$1 e=$2; shift 2
  [ -f $O/$n.done ] && return 0
  [ -f $D/STOP ] && exit 0
  echo "=== $n: env [$e] $* ($(date))" >> $O/$n.log
  if flock $LOCK env $e timeout 900 $BIN "$@" --json $O/results.jsonl --tag $n >> $O/$n.log 2>&1; then touch $O/$n.done; else echo "FAILED rc=$?" >> $O/$n.log; return 1; fi
}
probe() {   # probe <M> <ncb> <nseg> <fixed> [check]
  step fix_$(echo $4 | tr , _)_m$1_c$2_l$3_o0 "" --mode fwd --m $1 --ncb $2 --nseg $3 --order 0 --designs ps --fixed $4 --check ${5:-400} --reps 5
}
# correctness of the new paths (PFS, MINB, ACFG; flat tile; atomic and own outputs; order 2)
step fixsmoke_a "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 0 --designs ps --fixed 4,8,2,4,2,2,128,1,1,0 --check 300 --reps 2
step fixsmoke_b "" --mode fwd --m 200000 --ncb 3 --nseg 4 --order 0 --designs ps --fixed 4,16,1,1,2,3,128,1,1,3 --check 300 --reps 2
step fixsmoke_c "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 2 --designs ps --fixed 2,4,4,4,2,1,128,1,1,0 --check 300 --reps 2
step fixsmoke_d "" --mode fwd --m 200000 --ncb 8 --nseg 1 --order 0 --designs ps --fixed 4,4,8,4,2,2,256,0,0,3 --check 300 --reps 2
for c in 4,4,8,4,2,2,256,0,0,0 4,4,8,4,2,2,256,0,1,0 4,4,8,4,2,2,256,0,1,3 4,4,8,4,2,2,128,0,1,0 4,4,8,4,2,2,256,0,0,3 \
         4,8,2,4,2,2,256,0,0,0 4,8,2,4,2,2,256,0,1,0 4,8,2,4,2,2,256,0,1,3 4,16,2,4,2,2,256,0,1,0 4,16,2,4,2,2,128,0,1,0 \
         4,16,1,4,2,2,256,0,1,0 4,8,2,4,2,3,256,0,1,0 4,8,1,4,2,2,256,0,1,0; do
  probe 4000000 48 1 $c
done
for c in 4,4,8,2,2,2,128,0,0,0 4,4,8,2,2,2,128,1,0,0 4,4,8,2,2,2,128,1,1,0 4,8,2,2,2,2,128,1,1,0 4,8,2,2,2,2,256,1,1,0 \
         4,16,2,2,2,2,128,1,1,0 4,16,1,2,2,2,128,1,1,0; do
  probe 4000000 10 10 $c
done
for c in 4,4,8,2,2,1,128,0,0,0 4,4,8,2,2,1,128,1,1,0 4,8,2,2,2,1,128,1,1,0 4,16,2,2,2,1,128,1,1,0 4,8,2,2,2,2,128,1,1,0; do
  probe 1300000 10 10 $c
done
# shared float4 broadcast-load cost vs distinct addresses per warp, FFMA per load and resident warps
step ldsbench "" --mode ldsbench --m 1000
if [ -n "$AUTOTUNE" ]; then
  for c in "4000000 48 1" "4000000 10 10" "1300000 10 10" "4000000 16 1" "4000000 16 12" "1300000 48 1"; do
    set -- $c
    step fixtune_m$1_c$2_l$3_o0 "" --mode fwd --m $1 --ncb $2 --nseg $3 --order 0 --designs ps --check 800 --reps 5
  done
fi
echo "FIX DONE $(date) AUTOTUNE=$AUTOTUNE" >> $O/ps_driver.log
