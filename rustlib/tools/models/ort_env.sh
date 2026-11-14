#!/bin/bash
# Environment for the ONNX Runtime CUDA backend (source it):  source rustlib/tools/models/ort_env.sh
#   HASTY_ORT          libonnxruntime (onnxruntime-gpu wheel, CUDA 13 build) unpacked in rustlib/target/ort; the provider libs
#                      libonnxruntime_providers_{shared,cuda}.so must sit NEXT TO it (symlinks are fine)
#   LD_LIBRARY_PATH    CUDA 13 runtime + cuDNN 9 from the repo venv's torch wheels (the system toolkit libs do not load:
#                      "Cannot load symbol cublasLtCreate")
# The CUDA provider is requested with error_on_failure; without these the backend falls back to the CPU (report: cuda=false).
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
N="$R/.venv/lib/python3.12/site-packages/nvidia"
export HASTY_ORT="$R/target/ort/libonnxruntime.so"
export LD_LIBRARY_PATH="$N/cu13/lib:$N/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
