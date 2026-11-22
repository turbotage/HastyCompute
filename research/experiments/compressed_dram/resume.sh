#!/bin/bash
# resume after stop.sh: remove STOP and start the unit again (cases with a .done are skipped)
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/compressed_dram
rm -f $D/STOP
systemctl --user reset-failed cz-campaign 2>/dev/null
$D/start.sh
