#!/usr/bin/env bash
# Offline compile audit of the production NUFFT / FFT kernels for sm_80, sm_89, sm_90, sm_100, sm_120 (no GPU).
# research/arch_targets.md §5. CPU only: NVRTC in 4 threads, memory capped, no swap.
#   run.sh                      full audit            -> out/
#   run.sh lb4 --subst "__launch_bounds__(NTA)=>__launch_bounds__(NTA, 4)" --families ra,kl
#                               what-if variant       -> out_lb4/
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
tag="${1:-}"
[ $# -gt 0 ] && shift
out="$here/out${tag:+_$tag}"
cd "$repo/rustlib"
cargo build -j 4 -p hasty-bench --bin kernaudit
exec systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 -p CPUQuota=400% \
  nice -n 10 ./target/debug/kernaudit --out "$out" --jobs 4 "$@" 2> >(tee "$out.log" >&2)
