#!/bin/bash
# after chain4: opbench of the knot-path layout (kl3_*)
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout
while systemctl --user is-active --quiet kl-chain3 || systemctl --user is-active --quiet kl-chain4; do sleep 20; done
echo "chain5 start $(date)" > $D/chain5.log
bash -x $D/run_kl3.sh >> $D/chain5.log 2>&1
echo "chain5 end $(date)" >> $D/chain5.log
