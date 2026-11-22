#!/bin/bash
# ncu: time + DRAM bytes of every kernel of the warm-up forward + inverse (one pass, no replay, no cache flush)
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
B=${B:-/home/turbotage/Documents/GitHub/HastyCompute/rustlib/target/release/examples/bfp_fft_bench}
tag=$1; shift
flock $SP/gpu.lock /usr/local/cuda/bin/ncu --cache-control none --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum --csv "$@" $B ${GRID:-320 320 320 256 256 256 24} --once ${ST:+--store $ST} ${SLAB:+--slab $SLAB} > $SP/ncu_cz/$tag.csv 2> $SP/ncu_cz/$tag.err
