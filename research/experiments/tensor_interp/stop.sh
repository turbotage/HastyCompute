#!/bin/bash
# quiet time: STOP file (no new case starts) AND stop the unit now (the running case has no .done and reruns on resume)
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/tensor_interp
touch $D/STOP
systemctl --user stop tic-campaign
echo "stopped $(date '+%F %T')" >> $D/campaign.log
systemctl --user is-active tic-campaign || true
