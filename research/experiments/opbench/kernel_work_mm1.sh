#!/bin/bash
# 1 mm benches: knot cubic-10 unit and static 167k; extra env as args
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
echo "== knot cubic10 $*"
env HASTY_BENCH_PLAIN=0 HASTY_BENCH_CFG="11,2" HASTY_BENCH_FRAMES="10,3" HASTY_BENCH_KNOT=1 "$@" systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 target/release/segnufft_bench 198 189 228 334000 22 2>&1 | grep -E "SegNufft|autotune"
echo "== static $*"
env HASTY_BENCH_PLAIN=0 HASTY_BENCH_CFG="11,2" "$@" systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 target/release/segnufft_bench 198 189 228 167000 22 2>&1 | grep -E "SegNufft|autotune"
