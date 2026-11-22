#!/bin/bash
# start the campaign as the detached systemd user unit tic-campaign (24 GB cap, no swap); run freeze.sh first
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/tensor_interp
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
[ -e $SP/tic/frozen_at.txt ] || $D/freeze.sh
systemd-run --user --unit tic-campaign -E PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH -p MemoryMax=24G -p MemorySwapMax=0 $D/run.sh
