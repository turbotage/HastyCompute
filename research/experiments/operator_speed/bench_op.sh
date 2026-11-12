#!/bin/bash
# Operator timing batch (research/operator_speed.md): opspeed op on 4 sets of the real 2 mm ph0001 geometry, the
# recon's configuration (31 virtual coils, L 13, joint water + fat per-frame maps, order 2, host-packed zstd1 tables
# streamed per set, k-space staged from the read-only store), one variant per line. Needs the GPU to itself
# (holds data/hasty/GPU_BENCH_LOCK while running). Usage: bench_op.sh [variant-name ...] (default: all)
set -u
REPO=/home/turbotage/Documents/GitHub/HastyCompute
BIN=$REPO/rustlib/.agents/seg/target/release/opspeed
OUT=$REPO/data/hasty/studies/operator_speed_2026-11-12
KS=$REPO/rustlib/target/kstore/a3d37d3daa50f29c.kst
LOCK=$REPO/data/hasty/GPU_BENCH_LOCK
export HASTY_REPO=$REPO RAYON_NUM_THREADS=6
SETS="0:0,5:3,10:6,15:2"
COMMON="--sets $SETS --nc 31 --nseg 13 --tables packed --kstore $KS --reps 3 --pipe-reps 2 --json $OUT/bench_op.json"
declare -A V
V[w10_plain]="HASTY_SEG_BALANCE=0 --width 10"
V[w10]="--width 10"
V[w8]="--width 8"
V[w7]="--width 7"
V[w6]="--width 6"
V[w7_tt2]="HASTY_SEG_TT=2 --width 7"
V[w7_tt3]="HASTY_SEG_TT=3 --width 7"
V[w6_tt2]="HASTY_SEG_TT=2 --width 6"
V[w7_cb8]="--width 7 --grid-gb 7"
V[w10_cb8]="--width 10 --grid-gb 7"
V[s15_w6]="--sigma 1.5 --width 6 --grid-gb 11"
ORDER="w10_plain w10 w8 w7 w6 w7_tt2 w7_tt3 w6_tt2 w7_cb8 w10_cb8 s15_w6"
[ $# -gt 0 ] && ORDER="$*"
# wait for the GPU: another job may hold the lock (never remove a lock that is not ours)
while [ -e "$LOCK" ]; do echo "$(date +%T) waiting for $LOCK: $(head -c 200 "$LOCK")"; sleep 60; done
echo "$(date +%T) opspeed: operator batch ($ORDER)" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
mkdir -p $OUT/logs
for v in $ORDER; do
  spec=${V[$v]}
  envs=$(echo "$spec" | tr ' ' '\n' | grep '=' | tr '\n' ' ')
  args=$(echo "$spec" | tr ' ' '\n' | grep -v '=' | tr '\n' ' ')
  echo "== $v: $envs $args ($(date +%T))"
  env $envs HASTY_BENCH_VARIANT=$v systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 \
    env $envs HASTY_REPO=$REPO RAYON_NUM_THREADS=6 $BIN op $COMMON $args > $OUT/logs/op_$v.json 2> $OUT/logs/op_$v.log
  grep -E "engine:|autotuned|set [0-9]|pipeline|mean per set" $OUT/logs/op_$v.log
done
