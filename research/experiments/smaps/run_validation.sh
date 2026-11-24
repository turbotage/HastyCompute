#!/bin/bash
# Coil-map validation arms (research/smaps_estimation.md §5). NOT started by the agent: GPU campaign first.
# Binary: the separate build (do not rebuild target/release while the campaign uses it):
#   cd rustlib && nice cargo build --release -j 4 --target-dir target/alt -p hasty-bench --bin recon_p2np
# Usage: run_validation.sh <arm> [timeout s]      arms: smoke truth esp fit16 resp respfit16   (or: all)
# Dev config = research/dev_config_precond.md (4 resp x 20 cardiac, 128x72x60, vcoils auto 0.999, full physics + GNL
# + deg4, nseg auto, LLR k1 + db4 mix 0.3 + DFW sure + static lock group), out-half ADC, calibration out half (default).
# Each arm computes its OWN pooled warm start (the warm start depends on the maps: no copied warm state).
# One factor per arm relative to `esp`: fit16 = + --smaps-fit bspline:16; resp = + --smaps-resp; respfit16 = both;
# truth = --smaps truth (ceiling). Same precond / iterations everywhere (PRE, ITERS overridable).
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/smaps_val
PC=data/hasty/phantom/factory_arms/ph0001/tpyr_score/pc
B=rustlib/target/alt/release/recon_p2np
PRE=${PRE:---precond zmshr1}
ITERS=${ITERS:-40}
COM="--adc-part out --calib-adc out --keep 1 --resp-bins 4 --grid-scale 0.5 --kcrop --vcoils auto --coil-energy 0.999 \
 --phys full --gnl --conc-fit deg4 --nseg auto --prior llr --kappa 1 --wavelet-mix 0.3 --dfw sure \
 --static-lock 20:10:5 --lock-mode group --lam-method lanczos --report-detail --score-pc $PC \
 --holdout 25 --holdout-res --warm 0 --warm-full 60 --pool-metric none --init warm --every 5 $PRE"
arm() {
  case $1 in
    smoke) echo "--smaps espirit --smaps-resp --smaps-fit bspline:16 --warm-full 2 --iters 2 --every 1" ;;
    truth) echo "--smaps truth --iters $ITERS" ;;
    esp) echo "--smaps espirit --iters $ITERS" ;;
    fit16) echo "--smaps espirit --smaps-fit bspline:16 --iters $ITERS" ;;
    resp) echo "--smaps espirit --smaps-resp --iters $ITERS" ;;
    respfit16) echo "--smaps espirit --smaps-resp --smaps-fit bspline:16 --iters $ITERS" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}
run1() {
  n=$1; t=${2:-5400}
  [ -f $R/$n/DONE ] && { echo "skip $n"; return; }
  [ -x $B ] || { echo "no binary $B: build it first (see the header)"; exit 1; }
  mkdir -p $R/$n
  echo "$(date) $n: $B $S $COM $(arm $n)" >> $R/$n/log.txt
  flock $LOCK timeout $t systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 /usr/bin/time -v $B $S $COM $(arm $n) --out $R/$n >> $R/$n/log.txt 2>&1
  rc=$?; echo "exit $rc" >> $R/$n/log.txt; [ $rc = 0 ] && ! grep -q "^stopped" $R/$n/log.txt && touch $R/$n/DONE
  echo "$n exit $rc"
}
if [ "$1" = all ]; then
  # smoke first (flags parse, per-bin maps, fit, PerSetMaps + held-out swap run); stop if it fails
  run1 smoke 3600 || exit 1; grep -q "^exit 0" $R/smoke/log.txt || { echo "smoke failed: see $R/smoke/log.txt"; exit 1; }
  for a in esp fit16 resp respfit16 truth; do run1 $a 7200; done
else
  run1 "$1" "$2"
fi
