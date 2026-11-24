#!/bin/bash
# Preconditioner comparison in the dev config of research/dev_config_precond.md (measured noisy data, all readouts,
# 4 resp bins x 20 cardiac frames, 128x72x60, vcoils auto 0.999, ESPIRiT, full physics + GNL + deg4, nseg auto,
# LLR k1 + db4 wavelet mix 0.3 + DFW sure + static lock group). Usage: run.sh <name> <timeout s> <extra args...>
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/precond_compare
PC=data/hasty/phantom/factory_arms/ph0001/tpyr_score/pc
B=rustlib/target/release/recon_p2np
n=$1; t=$2; shift 2
[ -f $R/$n/DONE ] && { echo "skip $n"; exit 0; }
mkdir -p $R/$n
# arms: the pooled start of the warm run (its --warm-full checkpoints; --warm-full N then loads them)
[ "$n" != warm ] && [ -f $R/warm/DONE ] && for f in $R/warm/state_warm_full_done_*.hv; do [ -f $R/$n/$(basename $f) ] || cp $f $R/$n/; done
COM="--adc-part out --keep 1 --resp-bins 4 --grid-scale 0.5 --kcrop --vcoils auto --coil-energy 0.999 --smaps espirit \
 --phys full --gnl --conc-fit deg4 --nseg auto --prior llr --kappa 1 --wavelet-mix 0.3 --dfw sure \
 --static-lock 20:10:5 --lock-mode group --lam-method lanczos --report-detail --score-pc $PC"
flock $LOCK timeout $t systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 /usr/bin/time -v $B $S $COM "$@" --out $R/$n >> $R/$n/log.txt 2>&1
rc=$?; echo "exit $rc" >> $R/$n/log.txt; [ $rc = 0 ] && ! grep -q "^stopped" $R/$n/log.txt && touch $R/$n/DONE
echo "$n exit $rc"
