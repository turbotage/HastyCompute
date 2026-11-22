#!/bin/bash
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes
while true; do echo "$(date '+%F %T') $(nvidia-smi --query-gpu=power.limit,power.draw,clocks.sm,memory.used --format=csv,noheader)" >> $D/power.log; sleep 30; done
