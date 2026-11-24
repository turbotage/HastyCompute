#!/bin/bash
# ADMM validation (2026-11-23; research/precond_sampling_theory.md §2–§5, hasty_recon::admm, recon_p2np --prior-mode admm).
# Config = precond_compare/run_guard.sh (the production-like dev config): measured noisy data, --adc-part out, all
# readouts (--keep 1), --holdout 25 --holdout-res, 4 resp bins x 20 cardiac frames x 7 encodes, 128x72x60
# (--grid-scale 0.5 --kcrop), vcoils auto 0.999, ESPIRiT maps (out-half calibration, the default with --adc-part out),
# full physics + GNL + deg4, nseg auto, LLR k1 + db4 wavelet mix 0.3 + DFW sure + static lock group, pooled start
# --warm-full 60 --pool-metric none (each arm computes its own, in-process), pc score.
# Solver: --prior-mode admm, --precond none (P²nP's preconditioner is not used by ADMM; none = no circulants built).
# Binary: the snapshot rustlib/target/admm/snap/recon_p2np (built from target/admm; a rebuild does not change it).
# HASTY_KSTORE_REUSE_ONLY=1: the existing k-space store (key 0492a3fd8094221a, same arguments as g_esp) is reused;
# the run fails in seconds instead of writing a new 44 GB store if the key does not match.
# Usage: run_admm.sh <name> <timeout s> <extra args...>     (outputs in data/hasty/phantom/factory_arms/ph0001/admm/<name>)
# Pause: touch data/hasty/phantom/factory_arms/ph0001/admm/<name>/STOP (x, z, u, ρ saved at the end of the current outer
# iteration in state_<tag>.hsk); resume: rerun the same command (the log is appended; the pooled start is reloaded from
# its warm_full_done checkpoints, the setup is recomputed).
# Host memory cap ADMM_MEM (default 32G; 24G OOM-killed both arms on 2026-11-23: P²nP already peaks at 22.8–23.3 GiB).
# GPU test of the data step (device vs host reference): run_admm.sh gputest 900 (log admm/gputest.log). The arms wait for
# its result before taking the GPU lock and do not start if it failed (ADMM_NO_GATE=1: skip the gate).
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/admm
PC=data/hasty/phantom/factory_arms/ph0001/tpyr_score/pc
B=rustlib/target/admm/snap/recon_p2np
n=$1; t=$2; shift 2
mkdir -p $R
if [ "$n" = gputest ]; then
  flock $LOCK timeout $t systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 rustlib/target/admm/snap/admm_gpu --nocapture --test-threads 1 >> $R/gputest.log 2>&1
  rc=$?; echo "exit $rc" >> $R/gputest.log; echo "gputest exit $rc"; exit $rc
fi
[ -f $R/$n/DONE ] && { echo "skip $n"; exit 0; }
mkdir -p $R/$n
if [ -z "$ADMM_NO_GATE" ]; then
  while ! grep -q "^exit " $R/gputest.log 2>/dev/null; do sleep 60; done
  if ! grep -q "^exit 0$" $R/gputest.log; then echo "gputest failed ($(grep '^exit' $R/gputest.log | tail -1)): $n not started" | tee -a $R/$n/log.txt; exit 3; fi
fi
COM="--adc-part out --keep 1 --resp-bins 4 --grid-scale 0.5 --kcrop --vcoils auto --coil-energy 0.999 --smaps espirit \
 --phys full --gnl --conc-fit deg4 --nseg auto --prior llr --kappa 1 --wavelet-mix 0.3 --dfw sure \
 --static-lock 20:10:5 --lock-mode group --lam-method lanczos --report-detail --score-pc $PC \
 --holdout 25 --holdout-res --warm 0 --warm-full 60 --pool-metric none --init warm \
 --prior-mode admm --precond none"
export HASTY_KSTORE_REUSE_ONLY=1
flock $LOCK timeout $t systemd-run --user --scope -q -p MemoryMax=${ADMM_MEM:-32G} -p MemorySwapMax=0 /usr/bin/time -v $B $S $COM "$@" --out $R/$n >> $R/$n/log.txt 2>&1
rc=$?; echo "exit $rc" >> $R/$n/log.txt; [ $rc = 0 ] && ! grep -q "^stopped" $R/$n/log.txt && touch $R/$n/DONE
echo "$n exit $rc"
