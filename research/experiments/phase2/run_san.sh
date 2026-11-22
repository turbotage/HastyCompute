#!/bin/bash
# compute-sanitizer memcheck / racecheck / synccheck on the phase-2 variants (seg_phase2 p2_sanitizer_case), per case
# kind 0..3, unsplit (MAXITEM 16384) and split (64) lists; line info on
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/phase2/san
B=${B:-$SP/bin/seg_phase2_v2}
mkdir -p $OUT
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
for tool in memcheck racecheck synccheck; do
  for split in 0 1; do
    for k in 0 1 2 3; do
      n=${tool}_k${k}_s${split}
      [ -e $OUT/$n.done ] && continue
      [ -e $OUT/../STOP ] && exit 0
      SPL=""; [ $split = 1 ] && SPL="P2_SAN_SPLIT=1"
      flock $SP/gpu.lock timeout 900 env P2_SAN=1 P2_SAN_KIND=$k $SPL HASTY_NVRTC_LINEINFO=1 /usr/local/cuda/bin/compute-sanitizer --tool $tool --print-limit 20 $B p2_sanitizer_case --exact --nocapture --test-threads=1 > $OUT/$n.log 2>&1
      echo "exit $?" >> $OUT/$n.log
      touch $OUT/$n.done
    done
  done
done
touch $OUT/san.done
