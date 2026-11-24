#!/bin/bash
# Controls for "why is this hard" (2026-11-23, pre-registered in results.md):
#   crime: inverse crime in the production dev config: data = E x★ + noise (measured σ), x★ = PD_water + PD_fat + the
#          truth's blood signal per cardiac frame (--synthetic --synth-x pc; no respiratory motion in x★). Same prior,
#          solver, coil maps (ESPIRiT, the operator's own) and sampling as g_esp_air. --holdout-res is not allowed with
#          --synthetic, so it is scored by pc only.
#   base:  practice recipe on the measured data: LLR only (no wavelet / DFW / static lock), 20 cardiac frames from the
#          quietest 50 % of breathing (--resp-bins 1 --resp-accept 0.5), air zeroing kept (--air auto). Same solver and
#          operator as g_esp_air.
#   2×2 factorial (2026-11-23 late, after base beat g_esp_air): binning {4 resp bins, 1 gated bin 50 %} × prior
#   {LLR only, full = LLR + wavelet mix 0.3 + DFW + static lock}; g_esp_air = (r4, full), base = (gated, LLR).
#   basefull: gated bin + full prior. r4llr: 4 resp bins + LLR only. Static field (--field truth default) like both.
# Usage: run_ctrl.sh <crime|base|basefull|r4llr> <name> <timeout s> <extra args...>
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/precond_compare
PC=data/hasty/phantom/factory_arms/ph0001/tpyr_score/pc
B=rustlib/target/admm/snap/recon_p2np
kind=$1; n=$2; t=$3; shift 3
[ -f $R/$n/DONE ] && { echo "skip $n"; exit 0; }
mkdir -p $R/$n
OP="--adc-part out --keep 1 --grid-scale 0.5 --kcrop --vcoils auto --coil-energy 0.999 --smaps espirit \
 --phys full --gnl --conc-fit deg4 --nseg auto --lam-method lanczos --report-detail --score-pc $PC \
 --warm 0 --warm-full 60 --pool-metric none --init warm --every 5 --precond zmshr1 --iters 60 --step-guard 2 --air auto"
case $kind in
  crime) COM="$OP --resp-bins 4 --prior llr --kappa 1 --wavelet-mix 0.3 --dfw sure --static-lock 20:10:5 --lock-mode group --synthetic --synth-x pc --plan host" ;;
  base)  COM="$OP --resp-bins 1 --resp-accept 0.5 --prior llr --kappa 1 --holdout 25 --holdout-res" ;;
  basefull) COM="$OP --resp-bins 1 --resp-accept 0.5 --prior llr --kappa 1 --wavelet-mix 0.3 --dfw sure --static-lock 20:10:5 --lock-mode group --holdout 25 --holdout-res" ;;
  r4llr) COM="$OP --resp-bins 4 --prior llr --kappa 1 --holdout 25 --holdout-res" ;;
  *) echo "kind crime|base|basefull|r4llr"; exit 2 ;;
esac
flock $LOCK timeout $t systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0 /usr/bin/time -v $B $S $COM "$@" --out $R/$n >> $R/$n/log.txt 2>&1
rc=$?; echo "exit $rc" >> $R/$n/log.txt; [ $rc = 0 ] && ! grep -q "^stopped" $R/$n/log.txt && touch $R/$n/DONE
echo "$n exit $rc"
