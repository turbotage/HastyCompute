#!/bin/bash
# regimes.sh <label> [env...]: 1 mm static, 1 mm knot cubic10, 2 mm dense 2-D (W 8, order 0) with extra env
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
B=target/release/segnufft_bench
lab=$1; shift
gpu() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -v "^133565" | tr '\n' ' '; }
r() { local name=$1; shift; local pre=$(gpu); local out; out=$(env HASTY_BENCH_PLAIN=0 "$@" systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 $B "${ARGS[@]}" 2>&1 | grep -E "SegNufft L=" | tail -1); echo "$lab | $name | $out | other-gpu-before: [$pre]"; }
for spec in ${REGIMES:-static knot dense}; do
  case $spec in
    static) ARGS=(198 189 228 167000 22); r static "$@" HASTY_BENCH_CFG=11,2 ;;
    knot)   ARGS=(198 189 228 334000 22); r knot "$@" HASTY_BENCH_CFG=11,2 HASTY_BENCH_FRAMES=10,3 HASTY_BENCH_KNOT=1 ;;
    dense)  ARGS=(254 144 114 250000 8); r dense2mm "$@" HASTY_BENCH_CFG=10,0 HASTY_BENCH_FRAMES=10,3 HASTY_BENCH_2D=6,1 HASTY_BENCH_W=8 ;;
  esac
done
