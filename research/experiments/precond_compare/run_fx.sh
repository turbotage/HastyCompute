#!/bin/bash
# Fixed-objective solver test (2026-11-24, research/solver_choice.md §5). gc_llr's data / operator / scoring config
# (gated 1 bin 50 %, 20 cardiac frames × 7 encodes, static ratemap, svdn auto:noise, ESPIRiT, held-out 1/25), with a
# FIXED objective: one fixed LLR block partition (--llr-shifts 1; a proximal average over S > 1 shifts is not
# homogeneous in the prox step), penalty λ fixed at setup in objective units from the start stack
# (--llr-thr freeze), binary air mask from the start stack, never refreshed (--air start). The solver arm comes in
# as extra args (precond / prior-mode / prox-metric). Binary snap_fx.
# Usage: run_fx.sh <name> <timeout s> <solver args...>
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/precond_compare
T=data/hasty/phantom/factory_arms/ph0001/tpyr_score
B=rustlib/target/admm/snap_fx/recon_p2np
n=$1; t=$2; shift 2
[ -f $R/$n/DONE ] && { echo "skip $n"; exit 0; }
mkdir -p $R/$n
COM="--adc-part out --keep 1 --grid-scale 0.5 --kcrop --vcoils auto --coil-energy 0.999 --smaps espirit \
 --phys full --gnl --conc-fit deg4 --nseg auto:noise --seg-basis svdn --seg-weight energy \
 --resp-bins 1 --resp-accept 0.5 --prior llr --kappa 1 --air start --llr-shifts 1 --llr-thr freeze --state device \
 --lam-method lanczos --report-detail --score-pc $T/pc --score-pc-bins $T/pc_r0.181 \
 --holdout 25 --holdout-res --warm 0 --warm-full 60 --pool-metric none --init warm --every 5 --iters ${FX_ITERS:-100} --step-guard 2"
flock $LOCK timeout $t systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 /usr/bin/time -v $B $S $COM "$@" --out $R/$n >> $R/$n/log.txt 2>&1
rc=$?; echo "exit $rc" >> $R/$n/log.txt; [ $rc = 0 ] && ! grep -q "^stopped" $R/$n/log.txt && touch $R/$n/DONE
echo "$n exit $rc"
