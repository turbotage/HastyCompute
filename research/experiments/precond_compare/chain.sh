#!/bin/bash
# runs the remaining arms one after another, each its own unit (pc_<arm>), each only if it is expected to end before
# DEADLINE (est. minutes per arm given); usage: chain.sh  (skips arms with DONE)
D=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/precond_compare
R=/home/turbotage/Documents/GitHub/HastyCompute/data/hasty/phantom/factory_arms/ph0001/precond_compare
DEADLINE=$(date -d "2026-11-23 06:30" +%s)
A="--holdout 25 --holdout-res --warm 0 --warm-full 60 --pool-metric none --init warm --iters 60 --every 5"
run() { # name est_min args...
  n=$1; est=$2; shift 2
  [ -f $R/$n/DONE ] && { echo "$n done"; return; }
  while systemctl --user is-active -q pc_$n; do sleep 30; done
  end=$(( $(date +%s) + est * 60 ))
  if [ $end -gt $DEADLINE ]; then echo "$n not started: expected end $(date -d @$end) after the deadline"; return; fi
  echo "$n start $(date), expected end $(date -d @$end)"
  systemd-run --user --unit=pc_$n --collect --wait -q bash $D/run.sh $n 5400 "$@"
  echo "$n end $(date)"
}
while systemctl --user is-active -q pc_zmshr1m; do sleep 30; done
run zmshr1 55 $A --precond zmshr1
run p2 60 $A --precond zmshr1m --pp-metric p2
run pe 60 $A --precond zmshr1m --pp-metric pe1e-1_0.5
