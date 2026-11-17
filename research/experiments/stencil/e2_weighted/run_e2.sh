#!/usr/bin/env bash
# E2 of research/stencil_kernel_optimality.md §7 (research/stencil_weighted_update.md): CPU only, no GPU.
# Standalone crate against the rustlib crates (no rustlib file is changed); tests/e2.rs = a copy of
# rustlib/crates/hasty-bench/tests/seg_physics_direct.rs (flow_thorax path made absolute) + e2_weighted_chirp_polynomial_cpu.
# Usage: run_e2.sh [B0 ...]   env: E2_QONLY=1 (only the f16 / band rows), E2_PROXY=1 (proxy-vs-measured check), E2_EPS (ridge)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
R=$here/../../../../rustlib
[ -f "$here/Cargo.lock" ] || cp "$R/Cargo.lock" "$here/"
cd "$here"
CARGO_TARGET_DIR=$R/target nice -n 10 cargo build --release --tests -j 4
bin=$(ls -t "$R"/target/release/deps/e2-* | grep -v '\.d$' | head -1)
for b in "${@:-2.89 1.5 0.55}"; do
  for v in $b; do
    E2_B0=$v RAYON_NUM_THREADS=8 taskset -c 0-7 nice -n 10 "$bin" --ignored --exact e2_weighted_chirp_polynomial_cpu --nocapture 2>&1 | tee "$here/out/e2_$v.txt"
  done
done
