#!/bin/bash
# start the campaign as the detached systemd user unit cz-campaign (24 GB cap, no swap)
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/compressed_dram
systemd-run --user --unit cz-campaign -E PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH -p MemoryMax=24G -p MemorySwapMax=0 $D/run.sh
