#!/bin/bash
# Limiter of the plane-streamed forward `ps` (proto_gather.cu pg_fwd_ps; research/sample_kernels.md §11): timing-only
# ablations (PG_DEFS=PS_NOLOAD|PS_NOSYNC|PS_NOMAC|PS_NOSTORE|PS_FMAONLY, wrong results) at the best fixed configs of
# §10, resources (regs / local / blocks per SM) and the host-side work accounting (WORK line), FSB probes, and the
# 4 M 16 x 12 autotune (OOM fixed: per-tile-shape shared sample arrays). Detached, resumable: each step one locked GPU
# invocation (flock on the session gpu.lock; the 900 s timeout starts when the lock is held); a step is skipped when
# out/<name>.done exists; `touch STOP` stops between steps.
#   systemd-run --user --unit=proto-diag --collect -p MemoryMax=24G -p MemorySwapMax=0 -E PATH=$HOME/.cargo/bin:$PATH $PWD/research/experiments/sample_kernels/run_ps_diag.sh
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
# smoke: the new host code (work accounting, resources) and one ablation compile, small problem
step diag_smoke "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 0 --designs ps --fixed 4,4,4,4,2,1,128 --check 300 --reps 2
step diag_smoke_abl PG_DEFS=PS_FMAONLY,PS_NOLOAD,PS_NOSYNC --mode fwd --m 200000 --ncb 4 --nseg 2 --order 0 --designs ps --fixed 4,4,4,4,2,1,128 --check 50 --reps 2
# case: "M ncb nseg order fixed"
for c in "4000000 48 1 0 4,4,8,4,2,2,256" "4000000 10 10 0 4,4,8,2,2,2,128" "1300000 10 10 0 4,4,8,2,2,1,128"; do
  set -- $c
  tag=m$1_c$2_l$3_o$4
  A="--mode fwd --m $1 --ncb $2 --nseg $3 --order $4 --designs ps --fixed $5 --reps 5"
  step diag_base_$tag "" $A --check 800
  for d in PS_NOLOAD PS_NOSYNC PS_NOLOAD,PS_NOSYNC PS_NOMAC PS_NOSTORE PS_FMAONLY PS_FMAONLY,PS_NOLOAD,PS_NOSYNC; do
    n=$(echo $d | tr 'A-Z,' 'a-z_' | sed 's/ps_//g')
    step diag_${n}_$tag PG_DEFS=$d $A --check 50
  done
done
# FSB probes at 4 M 48 x 1 (more samples per slot = more FFMA per shared load; existing define, new values)
for f in "4,4,8,4,2,3,256" "4,4,8,4,2,4,256" "4,4,8,4,2,4,128" "4,4,8,2,2,4,128" "4,4,8,8,2,4,256"; do
  step diag_fsb_$(echo $f | tr , _)_m4000000_c48_l1_o0 "" --mode fwd --m 4000000 --ncb 48 --nseg 1 --order 0 --designs ps --fixed $f --check 400 --reps 5
done
# 4 M 16 coils x 12 segments: autotune (crashed with OOM before the shared sample arrays)
step ps_m4000000_c16_l12_o0 "" --mode fwd --m 4000000 --ncb 16 --nseg 12 --order 0 --designs ps --check 800 --reps 5
echo "DIAG DONE $(date)" >> $O/ps_driver.log
