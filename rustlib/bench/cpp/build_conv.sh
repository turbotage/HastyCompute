#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$(cd "$here/../../.." && pwd)"
CUDA="${CUDA_HOME:-/usr/local/cuda-13.0}"
"$CUDA/bin/nvcc" -O2 -std=c++17 -I"$root/cpp/lib/external/vkFFT" "$here/conv_baseline.cu" -o "$here/conv_baseline" -lcuda -lnvrtc -lcufft -Wno-deprecated-gpu-targets
