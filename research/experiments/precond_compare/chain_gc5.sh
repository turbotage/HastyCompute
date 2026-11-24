#!/bin/bash
# 2026-11-24, pre-registered in results.md "Encode trains and frames": every 7-encode train (79 ms) spans 2–3 cardiac
# frames (42.6 ms), so per-readout binning gives each encode of a frame different rotations (encode-specific
# undersampling, which does not cancel in the phase differences). gc_rm_bg = gc_llr_rm + --bin-group (each rotation's
# 7 encodes in the frame of its velocity time). gc_llr_rep = gc_llr repeated (run-to-run spread of the pc score).
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
bash $E/run_gc3.sh gc_rm_bg 5400 --ratemap truth-bc --bin-group
bash $E/run_gc3.sh gc_llr_rep 3600
