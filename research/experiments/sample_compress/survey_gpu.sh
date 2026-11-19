#!/bin/bash
# GPU timing + accuracy of hasty_recon::rcompress (rcompress_bench): SegNufft forward + adjoint per encode pass on all
# samples vs the kept samples + the banded R step. Target-like operator: 2-D knots 6 linear x 16 cubic (96 knot
# images), L 10, order 3, W 10, sigma 1.25; 16 coils (timing) / 8 (accuracy vs the W 12, sigma 2 reference).
# The bench takes the GPU lock itself (--gpu-lock) only around its GPU part. Resumable: cases with a finished json are
# skipped. Logs: logs/gpu_<case>.log. Outputs: out/impl/gpu_<case>.json. research/sample_compression.md § Implementation.
#   research/experiments/sample_compress/survey_gpu.sh <gpu.lock> [case-filter regex]
set -u
cd "$(dirname "$0")/../../.."
L=${1:?gpu lock}
F=${2:-.}
# snapshot binaries (other agents rebuild the workspace): rustlib/target/rcompress_snap/
B=rustlib/target/rcompress_snap/rcompress_bench
D2=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
D1=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min_box1mm_b/scan
O=research/experiments/sample_compress/out/impl
K="--knots2d 6,16 --nseg 10 --order 3 --reps 1 --gate 0"
run() { local name=$1; shift; [[ $name =~ $F ]] || return 0; [ -f $O/gpu_$name.json ] && grep -q '"speedup"' $O/gpu_$name.json && { echo "[$name] done before"; return 0; }; systemd-run --user --scope -q -p MemoryMax=36G -p MemorySwapMax=0 env "${ENVS[@]}" $B "$@" --gpu-lock "$L" --json $O/gpu_$name.json > research/experiments/sample_compress/logs/gpu_$name.log 2>&1; grep -E "^design|^GPU|^accuracy|^memory|skipped|panicked|error" research/experiments/sample_compress/logs/gpu_$name.log | grep -E "^design|^GPU|^accuracy|^memory|skipped|panicked|error" | sed "s/^/[$name] /"; }
# accuracy against the reference (stride 8 / 4: fewer samples), 8 coils
ENVS=(RC_BENCH_REF=12,2.0)
run acc_box2_box     $D2 --dims 80,80,80 --vox 2 --roi box --stride 8 --coils 8 $K
run acc_box2_ball    $D2 --dims 80,80,80 --vox 2 --roi ball --stride 8 --coils 8 $K
run acc_box1_box     $D1 --dims 160,160,160 --vox 1 --roi box --stride 4 --coils 8 $K
run acc_box1_ball80  $D1 --dims 160,160,160 --vox 1 --roi ball:80 --stride 4 --coils 8 $K
# timing: whole encodes, 16 coils, no reference
ENVS=(RC_BENCH_REF=none)
run box2_box     $D2 --dims 80,80,80 --vox 2 --roi box --coils 16 $K
run box2_ball    $D2 --dims 80,80,80 --vox 2 --roi ball --coils 16 $K
run box2_ball80  $D2 --dims 80,80,80 --vox 2 --roi ball:80 --coils 16 $K
run box1_box     $D1 --dims 160,160,160 --vox 1 --roi box --coils 16 $K
run box1_ball    $D1 --dims 160,160,160 --vox 1 --roi ball --coils 16 $K
run box1_ball80  $D1 --dims 160,160,160 --vox 1 --roi ball:80 --coils 16 $K
run box1_fov_box $D1 --dims 198,189,228 --vox 1 --roi box --coils 16 $K
run box1p_ball80 $D2 --dims 160,160,160 --vox 1 --kscale 2 --roi ball:80 --stride 2 --coils 16 --grid-gb 3 $K
run box1p_box    $D2 --dims 160,160,160 --vox 1 --kscale 2 --roi box --stride 2 --coils 16 --grid-gb 3 $K
# real-data compressed-vs-uncompressed CGLS test (tests/rcompress.rs; takes the lock for its GPU part only)
if [[ test =~ $F ]] && ! grep -q "test result: ok" research/experiments/sample_compress/logs/test_rcompress.log 2>/dev/null; then
  systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 env RC_GPU_LOCK="$L" rustlib/target/rcompress_snap/rcompress_test --nocapture > research/experiments/sample_compress/logs/test_rcompress.log 2>&1
  grep -E "W 1|test result|panicked" research/experiments/sample_compress/logs/test_rcompress.log
fi
echo done
