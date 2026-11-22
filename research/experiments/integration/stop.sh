#!/bin/bash
# quiet time: STOP file (no new case starts) AND stop the unit now (the GPU goes quiet within seconds; the case that was
# running has no .done and reruns from its start on resume.sh)
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/integration
touch $D/STOP
systemctl --user stop int-campaign
echo "stopped $(date '+%F %T')" >> $D/campaign.log
systemctl --user is-active int-campaign || true
