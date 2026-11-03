#!/usr/bin/env bash
# Profiles a HastyCompute test binary with Nsight Systems (nsys) and emits
# browser-viewable HTML summary reports — no Nsight Systems GUI/install needed
# to view the result, just scp the .html files (or the whole report dir) to
# wherever you're looking at them and open in any browser.
#
# Capture is bounded by cuda::ProfilerCaptureGuard (cudaProfilerStart/Stop) in
# the target binary, so the trace covers only the code region that wrapped
# itself in that guard, not the whole process lifetime.
#
# Usage: nsys_profile.sh <binary-path> [binary-args...]
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: nsys_profile.sh <binary-path> [binary-args...]" >&2
    exit 1
fi

BINARY="$1"; shift
NAME="$(basename "$BINARY")"

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${WORKSPACE_ROOT}/.nsys-reports"
STAMP="$(date +%Y%m%d-%H%M%S)"
PREFIX="${OUTDIR}/${NAME}-${STAMP}"

mkdir -p "${OUTDIR}"

nsys profile \
    --capture-range=cudaProfilerApi \
    -t cuda,nvtx,osrt \
    -o "${PREFIX}" \
    --force-overwrite=true \
    -- "${BINARY}" "$@"

# Tabular HTML reports (<table>-based, self-contained, no JS dependency on
# external/network resources) — not a visual timeline, but answers the usual
# questions directly: top kernels/memcpys by time, NVTX range durations,
# full chronological CUDA API/kernel/memcpy trace.
nsys stats \
    --report nvtx_pushpop_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_gpu_trace,nvtx_gpu_proj_trace \
    -f hdoc \
    -o "${PREFIX}" \
    --force-overwrite=true \
    "${PREFIX}.nsys-rep"

echo ""
echo "nsys trace:    ${PREFIX}.nsys-rep   (open with nsys-ui for the full visual timeline)"
echo "HTML reports:  ${PREFIX}_*.html     (open any of these directly in a browser, no install needed)"
