#!/bin/bash
# research/solver_choice.md §5 / results.md "Fixed-objective solver test" (revised after review).
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
R=data/hasty/phantom/factory_arms/ph0001/precond_compare
# dry start of the metric-prox arm (3 iterations): λ at setup, air from the start stack, the root
FX_ITERS=3 bash $E/run_fx.sh fx_dry_bfo 3600 --precond zmshr1 --prox-metric
grep -q "^exit 0" $R/fx_dry_bfo/log.txt || { echo "dry start failed: stop"; exit 1; }
bash $E/run_fx.sh fx_admm_a 7200 --precond none --prior-mode admm --admm-rho 1 --admm-inner 3 --admm-metric clip
bash $E/run_fx.sh fx_admm_b 7200 --precond none --prior-mode admm --admm-rho 1 --admm-inner 3 --admm-metric clip
bash $E/run_fx.sh fx_bfo 7200 --precond zmshr1 --prox-metric
bash $E/run_fx.sh fx_p2np 7200 --precond zmshr1
