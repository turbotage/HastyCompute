#!/bin/bash
# The remaining measurements of research/operator_speed.md that need the GPU to themselves (each block waits for
# data/hasty/GPU_BENCH_LOCK to be absent, then holds it): kernel split under nsys, the GPU LLR on the full 2 mm stack
# (synthetic and the pyr_2mm L2 result), and the GPU test suites touched by the patch. Run after a build of
# rustlib/.agents/seg (cargo build --release -j 4 -p hasty-bench --bin opspeed --bin recon_p2np --bin opval_v2).
# Usage: bench_rest.sh [nsys] [llr] [tests]   (default: all three)
set -u
REPO=/home/turbotage/Documents/GitHub/HastyCompute
SEG=$REPO/rustlib/.agents/seg
BIN=$SEG/target/release/opspeed
OUT=$REPO/data/hasty/studies/operator_speed_2026-11-12
KS=$REPO/rustlib/target/kstore/a3d37d3daa50f29c.kst
PH=$REPO/data/hasty/phantom/factory_arms/ph0001
LOCK=$REPO/data/hasty/GPU_BENCH_LOCK
PY=$REPO/rustlib/.venv/bin/python
export HASTY_REPO=$REPO RAYON_NUM_THREADS=6
WHAT=${*:-nsys llr tests}
lock() { while [ -e "$LOCK" ]; do echo "$(date +%T) waiting for $LOCK: $(head -c 200 "$LOCK")"; sleep 60; done; echo "$(date +%T) opspeed: $1" > "$LOCK"; }
unlock() { rm -f "$LOCK"; }
trap unlock EXIT
mkdir -p $OUT/nsys $OUT/logs
SETS="0:0,5:3,10:6,15:2"
if [[ " $WHAT " == *" nsys "* ]]; then
  lock "nsys kernel split"
  for v in "w10_plain:HASTY_SEG_BALANCE=0:--width 10" "w10:X=1:--width 10" "w7:X=1:--width 7"; do
    IFS=: read name envs args <<< "$v"
    env $envs HASTY_PROFILE_RANGE=1 nsys profile -q -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop -f true -o $OUT/nsys/$name \
      $BIN op --sets $SETS --nc 31 --nseg 13 --tables packed --reps 3 $args > $OUT/logs/nsys_$name.json 2> $OUT/logs/nsys_$name.log
    $PY $REPO/research/experiments/operator_speed/nsys_split.py $OUT/nsys/$name.nsys-rep | tee $OUT/logs/nsys_$name.split.txt
    rm -f $OUT/nsys/$name.sqlite
  done
  unlock
fi
if [[ " $WHAT " == *" llr "* ]]; then
  lock "GPU LLR full stack"
  for b in 32 64 128 256; do
    systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 env HASTY_REPO=$REPO RAYON_NUM_THREADS=6 $BIN llr --batch $b --reps 2 --json $OUT/llr_full.json 2>&1 | grep -v "^{" | tee -a $OUT/logs/llr_full.log
  done
  # real data: the pyr_2mm L2 result (9.3 GB read into RAM), support from the run
  systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 env HASTY_REPO=$REPO RAYON_NUM_THREADS=6 $BIN llr --batch 128 --reps 2 \
    --from $PH/pyr_2mm/L2_s1/recon_seg_zmshr1s_k1_air0.05m4i0e2s.npy --crop 0:114 --support $PH/pyr_2mm/L2_s1/mask_support.npy --json $OUT/llr_full.json 2>&1 | grep -v "^{" | tee -a $OUT/logs/llr_full.log
  unlock
fi
if [[ " $WHAT " == *" tests "* ]]; then
  lock "GPU tests of the opspeed patch"
  cd $SEG
  for t in "hasty-nufft seg_balance_gpu" "hasty-nufft seg_fat_union_gpu" "hasty-nufft segnufft" "hasty-nufft seg_stream" "hasty-nufft seg_multi" \
           "hasty-recon llr_gpu" "hasty-recon kstage_ahead" "hasty-recon host_state" "hasty-recon grouped_sets" "hasty-recon fat_direct" "hasty-recon stream_tables" \
           "hasty-recon coil_sampling" "hasty-bench seg_physics_direct"; do
    set -- $t
    echo "== cargo test -p $1 --test $2 ($(date +%T))"
    systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 cargo test --release -j 4 -p $1 --test $2 2>&1 | grep -E "^test |test result|panicked" | tee -a $OUT/logs/tests.log
  done
  systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 cargo test --release -j 4 -p hasty-recon --lib llr 2>&1 | grep -E "^test |test result" | tee -a $OUT/logs/tests.log
  unlock
fi
