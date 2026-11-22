#!/bin/bash
# GPU power limit / draw every 30 s (the user changes the limit during the session; logs are matched by time)
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout
while true; do echo "$(date '+%F %T') $(nvidia-smi --query-gpu=power.limit,power.draw,clocks.sm --format=csv,noheader)" >> $D/power.log; sleep 30; done
