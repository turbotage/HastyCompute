#!/bin/bash
# 2026-11-24, pre-registered in results.md "Run-to-run spread": two IDENTICAL gated LLR runs (the gc_llr config) to
# 200 iterations. Question: do they converge to the same estimate (pc, held-out, f), i.e. is the pc spread of 60
# iterations (0.262 vs 0.173) trajectory noise of an unconverged solver? Decision: converged and equal → evaluate every
# arm at convergence; not equal → the solver / estimator is not reproducible: change the algorithm before cluster runs.
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
GC_ITERS=200 bash $E/run_t.sh conv_a 7200
GC_ITERS=200 bash $E/run_t.sh conv_b 7200
