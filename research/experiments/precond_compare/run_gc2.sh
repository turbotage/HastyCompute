#!/bin/bash
# Gated-cell campaign (2026-11-24, pre-registered in results.md "Gated-cell campaign"). Measured data of the 10-min
# ph0001 scan, out half, all readouts, respiratory gating to the quietest 50 % of breathing (1 bin, true state 0.184),
# 20 cardiac frames × 7 encodes = 140 sets of 162–202 readouts, 128 × 72 × 60 grid, 31 virtual coils, ESPIRiT maps,
# full physics + GNL + deg-4 concomitant fit, production segment settings (auto:noise, svdn basis, energy weights),
# P²nP with zmshr1 (no metric), 60 iterations, air auto, held-out 1/25 of the readouts. Scored against the truth at
# breathing state 0.181 (--score-pc-bins; the old mean-state truth is printed as an oracle).
# run_gc2.sh = run_gc.sh with the segment options from GC_SEG (default: the production ones).
# Usage: run_gc2.sh <name> <timeout s> <extra args (the one changed factor)...>
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/precond_compare
T=data/hasty/phantom/factory_arms/ph0001/tpyr_score
B=rustlib/target/admm/snap_gc/recon_p2np
n=$1; t=$2; shift 2
[ -f $R/$n/DONE ] && { echo "skip $n"; exit 0; }
mkdir -p $R/$n
COM="--adc-part out --keep 1 --grid-scale 0.5 --kcrop --vcoils auto --coil-energy 0.999 --smaps espirit \
 --phys full --gnl --conc-fit deg4 ${GC_SEG:---nseg auto:noise --seg-basis svdn --seg-weight energy} \
 --resp-bins 1 --resp-accept 0.5 --prior llr --kappa 1 --air auto \
 --lam-method lanczos --report-detail --score-pc $T/pc --score-pc-bins $T/pc_r0.181 \
 --holdout 25 --holdout-res --warm 0 --warm-full 60 --pool-metric none --init warm --every 5 \
 --precond zmshr1 --iters 60 --step-guard 2"
flock $LOCK timeout $t systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 /usr/bin/time -v $B $S $COM "$@" --out $R/$n >> $R/$n/log.txt 2>&1
rc=$?; echo "exit $rc" >> $R/$n/log.txt; [ $rc = 0 ] && ! grep -q "^stopped" $R/$n/log.txt && touch $R/$n/DONE
echo "$n exit $rc"
