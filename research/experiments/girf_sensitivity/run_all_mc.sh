#!/bin/bash
# the R Monte Carlo runs of the study, sequential, 6 single-threaded workers (resumable: rerun to continue)
cd "$(dirname "$0")"
PY=../../../rustlib/.venv/bin/python
L=../../../data/hasty/seq/studies/girf_sensitivity_2026-11-11/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { tag=$1; shift; $PY mc.py --tag $tag --workers ${W:-6} "$@" >> $L/mc_$tag.log 2>&1; }
run R_rand_10   --truth rand --snr 1000 --navg 50  --ndraw 48
run R_rand_T200 --truth rand --snr 1000 --navg 25 --ndraw 32 --tadc 0.2
run R_res_10    --truth resonance --snr 1000 --navg 50 --ndraw 24
run R_rand_nonuis --truth rand --snr 1000 --navg 50 --ndraw 24 --nuis off
true                                                         
run R_rand_s300 --truth rand --snr 300  --navg 50  --ndraw 24
run R_rand_T200_60 --truth rand --snr 1000 --navg 150 --ndraw 24 --tadc 0.2
true                                                         
true                                                                   
