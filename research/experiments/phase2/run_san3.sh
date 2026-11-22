#!/bin/bash
# compute-sanitizer racecheck / synccheck on the phase-2 variants (seg_phase2 p2_sanitizer_case), restricted to the
# three phase-2 kernel entries, one case kind (0..3) × kernel family (ra / kl / vs) × list mode per process: racecheck's
# host memory grows with every loaded module (the 1-D case with all families exceeded 24 GB). memcheck: run_san.sh.
SP=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad
OUT=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/phase2/san
B=${B:-$SP/bin/seg_phase2_v7}
FILT="--kernel-name kns=seg_adjoint_ra --kernel-name kns=seg_adjoint_kl --kernel-name kns=seg_forward_vs"
cd /home/turbotage/Documents/GitHub/HastyCompute/rustlib
for tool in racecheck synccheck; do
  for split in 0 1; do
    for k in 0 1 2 3; do
      for fam in ra kl vs; do
        [ $fam = kl ] && [ $k -lt 2 ] && continue   # kl: knot engines only
        [ $fam = vs ] && [ $k = 2 ] && continue     # the 1-D order-1 case has no vs geometry
        n=${tool}_k${k}_s${split}_$fam
        [ -e $OUT/$n.done ] && continue
        [ -e $OUT/../STOP ] && exit 0
        # case 0 / 1 with all families together already ran clean under racecheck (unsplit): racecheck_k{0,1}_s0
        [ $tool = racecheck ] && [ $split = 0 ] && [ $k -lt 2 ] && [ -e $OUT/racecheck_k${k}_s0.done ] && continue
        SPL=""; [ $split = 1 ] && SPL="P2_SAN_SPLIT=1"
        flock $SP/gpu.lock timeout 900 env P2_SAN=1 P2_SAN_KIND=$k P2_SAN_FAM=$fam $SPL HASTY_NVRTC_LINEINFO=1 /usr/local/cuda/bin/compute-sanitizer --tool $tool --print-limit 20 $FILT $B p2_sanitizer_case --exact --nocapture --test-threads=1 > $OUT/$n.log 2>&1
        echo "exit $?" >> $OUT/$n.log
        touch $OUT/$n.done
      done
    done
  done
done
touch $OUT/san3.done
