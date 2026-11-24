#!/bin/bash
# Run 1 (2026-11-23): the dev config of run.sh (measured noisy data, all readouts, --holdout 25, 4 resp bins x 20
# cardiac frames x 7 encodes, 128x72x60, vcoils auto 0.999, full physics + GNL + deg4, nseg auto, LLR k1 + db4 wavelet
# mix 0.3 + DFW sure + static lock group) with the coil maps as a parameter. Unlike run.sh, every arm computes its
# own pooled start in the same process (--warm-full 60 plain CG; the start depends on the maps).
# Usage: run_guard.sh <name> <timeout s> <smaps: espirit|truth> <extra args...>
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/precond_compare
PC=data/hasty/phantom/factory_arms/ph0001/tpyr_score/pc
B=${RG_BIN:-rustlib/target/release/recon_p2np}
n=$1; t=$2; sm=$3; shift 3
[ -f $R/$n/DONE ] && { echo "skip $n"; exit 0; }
mkdir -p $R/$n
COM="--adc-part out --keep 1 --resp-bins 4 --grid-scale 0.5 --kcrop --vcoils auto --coil-energy 0.999 --smaps $sm \
 --phys full --gnl --conc-fit deg4 --nseg auto --prior llr --kappa 1 --wavelet-mix 0.3 --dfw sure \
 --static-lock 20:10:5 --lock-mode group --lam-method lanczos --report-detail --score-pc $PC \
 --holdout 25 --holdout-res --warm 0 --warm-full 60 --pool-metric none --init warm --every 5"
flock $LOCK timeout $t systemd-run --user --scope -q -p MemoryMax=${RG_MEM:-24G} -p MemorySwapMax=0 /usr/bin/time -v $B $S $COM "$@" --out $R/$n >> $R/$n/log.txt 2>&1
rc=$?; echo "exit $rc" >> $R/$n/log.txt; [ $rc = 0 ] && ! grep -q "^stopped" $R/$n/log.txt && touch $R/$n/DONE
echo "$n exit $rc"
