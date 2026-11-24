#!/bin/bash
# Gated-cell campaign (results.md "Gated-cell campaign", revised after review): one changed factor per run.
# 1. gc_llr: LLR only, static ratemap (reference for the ratemap question)
# 2. gc_llr_rm: + true ratemap per frame (frame mode; startup self-check aborts on a builder/slot bug)
# 3. only if 2 finished: the prior components on the true-ratemap operator (the decided dev model,
#    memory ratemap_per_frame), each against gc_llr_rm
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
R=data/hasty/phantom/factory_arms/ph0001/precond_compare
bash $E/run_gc.sh gc_llr 3600
bash $E/run_gc.sh gc_llr_rm 5400 --ratemap truth-bc
if [ ! -f $R/gc_llr_rm/DONE ]; then echo "gc_llr_rm did not finish: components not started"; exit 1; fi
bash $E/run_gc.sh gc_rm_wav 5400 --ratemap truth-bc --wavelet-mix 0.3
bash $E/run_gc.sh gc_rm_dfw 5400 --ratemap truth-bc --dfw sure
bash $E/run_gc.sh gc_rm_lock 5400 --ratemap truth-bc --static-lock 1.5:10:5 --lock-mode group
