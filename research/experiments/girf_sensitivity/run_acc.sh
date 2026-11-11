#!/bin/bash
# accuracy-only MC runs (no R), girf.rs resonance chain; 6 runs in parallel, one worker each (resumable)
cd "$(dirname "$0")"
PY=../../../rustlib/.venv/bin/python
L=../../../data/hasty/seq/studies/girf_sensitivity_2026-11-11/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
a() { tag=$1; shift; $PY mc.py --tag $tag --workers 1 --no-r --truth resonance --snr 1000 "$@" > $L/mc_$tag.log 2>&1; }
a acc_res_10min --navg 50 --ndraw 24 &
a acc_res_20min --navg 100 --ndraw 24 &
a acc_res_60min --navg 300 --ndraw 24 &
a acc_res_10min_ssalt --navg 50 --ndraw 24 --ss-alt &
a acc_res_10min_off --navg 50 --ndraw 24 --nuis off &
a acc_res_10min_only_b0_rms_hz --navg 50 --ndraw 12 --nuis only:b0_rms_hz &
wait
for n in only:center only:coilw only:df0_hz only:eps_ss only:dp_rand only:k_ss; do t=$(echo $n | tr ':' '_'); a acc_res_10min_$t --navg 50 --ndraw 12 --nuis $n & done
wait
