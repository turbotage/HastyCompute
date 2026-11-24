#!/bin/bash
# 2026-11-24, pre-registered in results.md: with the true ratemap the measured residual is still 27.6× the noise floor.
# gc_rm_tmaps = gc_llr_rm with the TRUE coil maps (oracle): how much of the remaining mismatch is the ESPIRiT maps?
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
GC_SMAPS=truth bash $E/run_gc3.sh gc_rm_tmaps 5400 --ratemap truth-bc
