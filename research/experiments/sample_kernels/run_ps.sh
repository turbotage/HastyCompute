#!/bin/bash
# Plane-streamed home-tile forward (`ps`, proto_gather.cu pg_fwd_ps) + knot variant-1 adjoint coil blocking (acb) +
# ablation of today's kernels (research/sample_kernels.md §6-8). Detached, resumable: each step one locked GPU
# invocation (flock on the session gpu.lock; the 900 s timeout starts when the lock is held); a step is skipped when
# out/<name>.done exists; `touch STOP` stops between steps.
#   systemd-run --user --unit=proto-ps --collect -p MemoryMax=24G -p MemorySwapMax=0 $PWD/research/experiments/sample_kernels/run_ps.sh
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
# smoke: correctness of ps at fixed shapes; gate = every BEST line with rel err < 1e-5
smoke_ok() {
  for n in "$@"; do
    [ -f $O/$n.done ] || return 1
    awk '/BEST/ { for (i = 1; i < NF; i++) if ($i == "err") { v = $(i + 1) + 0; if (v > 1e-5 || v != v) bad = 1; seen = 1 } } END { exit (bad || !seen) }' $O/$n.log || return 1
  done
}
step smoke_ps_o0 "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 0 --designs ps --fixed 4,4,4,4,2,1,128 --check 300 --reps 2
step smoke_ps_o0b "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 0 --designs ps --fixed 2,8,8,2,4,2,256 --check 300 --reps 2
step smoke_ps_o0c "" --mode fwd --m 200000 --ncb 3 --nseg 4 --order 0 --designs ps --fixed 4,4,8,1,2,2,128 --check 300 --reps 2
step smoke_ps_o2 "" --mode fwd --m 200000 --ncb 4 --nseg 2 --order 2 --designs ps --fixed 2,4,4,4,2,1,128 --check 300 --reps 2
# knot kernel variants incl. the new adjoint acb blocking against variant 0 (GPU test, small)
if [ ! -f $O/test_kv.done ] && [ ! -f $D/STOP ]; then
  T=$(ls -t $R/target/release/deps/seg_knot_kv-* | grep -v '\.d$' | head -1)
  if flock $LOCK timeout 900 $T --nocapture --test-threads 1 > $O/test_kv.log 2>&1; then touch $O/test_kv.done; else echo "FAILED rc=$?" >> $O/test_kv.log; fi
fi
if smoke_ok smoke_ps_o0 smoke_ps_o0b smoke_ps_o0c smoke_ps_o2; then
  for c in "4000000 10 10 0" "4000000 48 1 0" "4000000 16 1 0" "1300000 10 10 0" "1300000 48 1 0" "4000000 16 12 0" "1300000 192 1 0" "4000000 10 10 2" "1300000 10 10 2"; do
    set -- $c
    step ps_m$1_c$2_l$3_o$4 "" --mode fwd --m $1 --ncb $2 --nseg $3 --order $4 --designs ps --check 800 --reps 5
  done
else
  echo "ps smoke failed: no ps tuning" >> $O/ps_driver.log
fi
# gm at 4 M (the earlier 4 M fwd steps failed on the timing-event bug)
step gm_m4000000_c48_l1_o0 "" --mode fwd --m 4000000 --ncb 48 --nseg 1 --order 0 --designs gm --check 800 --reps 5
step gm_m4000000_c10_l10_o0 "" --mode fwd --m 4000000 --ncb 10 --nseg 10 --order 0 --designs gm --check 800 --reps 5
# today's kernels: ablation (autotuned under each switch) and the knot-major production kernels on one knot
for c in "4000000 10 10 0" "4000000 48 1 0" "1300000 10 10 0" "4000000 10 10 2"; do
  set -- $c
  tag=m$1_c$2_l$3_o$4
  step abl_noatomic_$tag HASTY_SEG_DEFS=PROF_NOATOMIC --mode base --m $1 --ncb $2 --nseg $3 --order $4
  step abl_nomac_$tag HASTY_SEG_DEFS=PROF_NOMAC --mode base --m $1 --ncb $2 --nseg $3 --order $4
  step abl_nostage_$tag HASTY_SEG_DEFS=PROF_NOSTAGE --mode base --m $1 --ncb $2 --nseg $3 --order $4
  step knot_kv0_$tag HASTY_SEG_KV=0 --mode base --knot --m $1 --ncb $2 --nseg $3 --order $4
  step knot_kv1_$tag HASTY_SEG_KV=1 --mode base --knot --m $1 --ncb $2 --nseg $3 --order $4
done
echo "ALL DONE $(date)" >> $O/ps_driver.log
