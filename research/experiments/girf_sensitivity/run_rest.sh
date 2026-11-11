#!/bin/bash
# remaining MC runs of the study (after the estimator was fixed: temporal unwrapping, no adaptive window cut)
cd "$(dirname "$0")"
PY=../../../rustlib/.venv/bin/python
L=../../../data/hasty/seq/studies/girf_sensitivity_2026-11-11/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY mc.py --tag acc_res_10min_v2 --truth resonance --snr 1000 --navg 50 --ndraw 24 --workers 1 --no-r > $L/mc_acc_res_10min_v2.log 2>&1 &
$PY mc.py --tag R_rand_s300 --truth rand --snr 300 --navg 50 --ndraw 24 --workers 5 > $L/mc_R_rand_s300.log 2>&1
$PY mc.py --tag R_rand_T200_60 --truth rand --snr 1000 --navg 150 --ndraw 24 --tadc 0.2 --workers 6 > $L/mc_R_rand_T200_60.log 2>&1
$PY mc_refit.py --tag R_rand_T200 --ndraw 16 --workers 6 > $L/mc_refit_T200.log 2>&1
wait
