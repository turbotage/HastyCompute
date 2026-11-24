#!/bin/bash
# 2026-11-24, pre-registered in results.md: does the production segment rule (svdn, energy, auto:noise → L = 4)
# lose against the old dev operator (fixed basis, count weights, L = 5)? gc_llr_fixed5 = gc_llr with only the segment
# options changed. Also gc_llr_fixed5 vs c_base_llr_ga05 = same operator, only the scoring truth changed.
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
GC_SEG="--nseg 5 --seg-basis fixed --seg-weight count" bash $E/run_gc2.sh gc_llr_fixed5 3600
