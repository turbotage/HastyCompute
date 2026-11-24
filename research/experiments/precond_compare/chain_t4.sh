#!/bin/bash
# 2026-11-24: gc_rm_tmaps again. With --smaps truth the support is the whole box (air included), where the f32-stored
# per-frame field differed from the static f64 one by 7e-5 in the degenerate self-check. RateFrame.om is now f64
# (snap_t2): the degenerate frame model is exactly the static field.
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
GC_BIN=rustlib/target/admm/snap_t2/recon_p2np GC_SMAPS=truth bash $E/run_t.sh gc_rm_tmaps 5400 --ratemap truth-bc
