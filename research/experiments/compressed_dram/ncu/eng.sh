#!/bin/bash
# ncu: every kernel of one profiled gradient (opbench --profile), DRAM bytes + time per kernel (one pass, no replay)
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
B=${B:-$SP/cz/base/opbench}
tag=$1; shift
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
flock $SP/gpu.lock systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 /usr/local/cuda/bin/ncu --profile-from-start off --cache-control none --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum --csv $B "$@" --reps 1 --profile > $SP/ncu_cz/$tag.csv 2> $SP/ncu_cz/$tag.err
