#!/bin/bash
# phase-2 correctness tests (frozen binary $1, tag $2): each test split by order and list mode (P2_ORDERS, P2_MI) so
# every invocation stays under the 900 s timeout; one flock'd run per part; logs research/experiments/phase2/<tag>_*.log
B=$1; TAG=$2
O=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/phase2
L=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
part() {  # test orders mi
  local n=${TAG}_$1_o${2//,/}_m$3
  [ -e $O/$n.done ] && return
  [ -e $O/STOP ] && exit 0
  echo "== $1 orders $2 mi $3 $(date +%T)" > $O/$n.log
  flock $L bash -c "echo \"locked \$(date +%T) power limit \$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)\" >> $O/$n.log; P2_ORDERS=$2 P2_MI=$3 timeout 900 $B $1 --exact --nocapture --test-threads=1 >> $O/$n.log 2>&1; echo \"== exit \$? \$(date +%T)\" >> $O/$n.log"
  touch $O/$n.done
}
for mi in 16384 64; do
  for o in 0 1 2; do part p2_ra_static $o $mi; done
  for o in 0 1 2 3; do part p2_ra_knot $o $mi; done
  for o in 0 1 2 3; do part p2_kl_knot $o $mi; done
  for o in 0 1 2 3; do part p2_vs_subchunks $o $mi; done
done
part p2_spill_screen 0 16384
part p2_autotune 0 16384
touch $O/$TAG.done
