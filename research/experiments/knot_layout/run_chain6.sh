#!/bin/bash
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/knot_layout
while systemctl --user is-active --quiet kl-chain5; do sleep 20; done
$D/run_kl4.sh
