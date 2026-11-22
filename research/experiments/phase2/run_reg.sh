#!/bin/bash
# seg_* GPU regression with frozen binaries (defaults: phase-2 candidates in the autotune), one flock'd run per binary
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/phase2/reg
mkdir -p $OUT
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
for b in $SP/bin/reg/*; do
  n=$(basename $b)
  [ -e $OUT/$n.done ] && continue
  [ -e $OUT/../STOP ] && exit 0
  flock $SP/gpu.lock timeout 900 $b --test-threads=1 > $OUT/$n.log 2>&1
  echo "exit $?" >> $OUT/$n.log
  touch $OUT/$n.done
done
touch $OUT/reg.done
