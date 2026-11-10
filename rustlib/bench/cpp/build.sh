#!/usr/bin/env bash
# Builds the VkFFT/cuFFT baseline against the VkFFT copy vendored in bench/cpp/vendor/vkFFT.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$(cd "$here/../../.." && pwd)"
CUDA="${CUDA_HOME:-/usr/local/cuda-13.0}"
g++ -O2 -std=c++17 -I"$here/vendor/vkFFT" -I"$CUDA/include" "$here/vkfft_bench.cpp" -o "$here/vkfft_bench" \
  -L"$CUDA/lib64" -L"$CUDA/lib64/stubs" -lcuda -lnvrtc -lcudart -lcufft -Wl,-rpath,"$CUDA/lib64"
