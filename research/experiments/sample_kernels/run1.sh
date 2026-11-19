#!/bin/bash
# one locked, memory-capped proto_gather invocation: run1.sh <log> <args...>
LOG=$1; shift
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
BIN=/home/turbotage/Documents/GitHub/HastyCompute/rustlib/target/release/proto_gather
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
flock $LOCK systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 $BIN "$@" 2>&1 | tee -a $LOG
