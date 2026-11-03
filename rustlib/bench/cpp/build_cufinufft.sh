#!/usr/bin/env bash
# Release build of cuFINUFFT (v2.5.1 sources already fetched by the C++ project) for fair baselines.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
SRC="${FINUFFT_SRC:-/home/turbotage/Documents/GitHub/HastyCompute/out/build/Config-ClangDebug-Linux/lib-build/_deps/finufft-src}"
cmake -S "$SRC" -B "$here/build-cufinufft" -G Ninja -DCMAKE_BUILD_TYPE=Release -DFINUFFT_USE_CUDA=ON -DFINUFFT_USE_CPU=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DFINUFFT_BUILD_TESTS=OFF -DFINUFFT_BUILD_EXAMPLES=OFF -DFINUFFT_BUILD_PYTHON=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler
cmake --build "$here/build-cufinufft" --target cufinufft -j8
