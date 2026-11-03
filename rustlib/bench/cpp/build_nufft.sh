#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
CUDA="${CUDA_HOME:-/usr/local/cuda-13.0}"
SRC="${FINUFFT_SRC:-/home/turbotage/Documents/GitHub/HastyCompute/out/build/Config-ClangDebug-Linux/lib-build/_deps/finufft-src}"
"$CUDA/bin/nvcc" -O2 -std=c++17 -allow-unsupported-compiler -I"$SRC/include" "$here/nufft_baseline.cu" -o "$here/nufft_baseline" \
  "$here/build-cufinufft/libcufinufft.a" "$here/build-cufinufft/src/common/libfinufft_common.a" -lcufft -lcudart -Wno-deprecated-gpu-targets
