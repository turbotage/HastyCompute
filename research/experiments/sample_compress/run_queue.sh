#!/bin/bash
# The queued GPU work of the rcompress implementation (research/sample_compression.md § 9.5), serialised on the GPU lock:
# 1. the real-data CGLS test, 2. the bench survey (accuracy vs reference, then timing), 3. the driver checks.
# Resumable (finished parts are skipped). Logs: research/experiments/sample_compress/logs/.
#   systemd-run --user --unit=rc-gpu-queue --collect -p MemoryMax=38G -p MemorySwapMax=0 \
#     --working-directory=$PWD research/experiments/sample_compress/run_queue.sh <gpu.lock>
set -u
cd "$(dirname "$0")/../../.."
L=${1:?gpu lock}
G=research/experiments/sample_compress
$G/survey_gpu.sh "$L" test >> $G/logs/queue.log 2>&1
$G/survey_gpu.sh "$L" '^(acc_|box|full)' >> $G/logs/queue.log 2>&1
$G/driver_check.sh "$L" $G/out/drivers >> $G/logs/queue.log 2>&1
echo "queue done $(date)" >> $G/logs/queue.log
