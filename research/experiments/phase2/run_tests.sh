#!/bin/bash
# phase-2 correctness tests (frozen binary $1), one flock'd run per test; logs in research/experiments/phase2/
B=$1; TAG=$2; shift 2
O=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/phase2
L=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
for t in "$@"; do
  echo "== $t $(date +%T) power limit $(nvidia-smi --query-gpu=power.limit --format=csv,noheader)" >> $O/${TAG}_$t.log
  flock $L timeout 900 $B $t --nocapture --test-threads=1 >> $O/${TAG}_$t.log 2>&1
  echo "== exit $? $(date +%T)" >> $O/${TAG}_$t.log
done
touch $O/${TAG}.done
