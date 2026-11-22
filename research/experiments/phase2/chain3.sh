#!/bin/bash
# after chain2's test half: the vs sub-chunk test with the default lists (v5: nwc 5 / x padding added)
O=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/phase2
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
L=$SP/gpu.lock
until [ -e $O/t4.done ]; do sleep 20; done
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
for mi in 1024 64; do for o in 0 1 2 3; do
  n=t5_p2_vs_subchunks_o${o}_m$mi
  [ -e $O/$n.done ] && continue
  echo "== p2_vs_subchunks orders $o mi $mi $(date +%T)" > $O/$n.log
  flock $L bash -c "P2_ORDERS=$o P2_MI=$mi timeout 900 $SP/bin/seg_phase2_v5 p2_vs_subchunks --exact --nocapture --test-threads=1 >> $O/$n.log 2>&1; echo \"== exit \$? \$(date +%T)\" >> $O/$n.log"
  touch $O/$n.done
done; done
touch $O/t5.done
