#!/bin/bash
# 2026-11-24, pre-registered: gc_llr_svdn5 = gc_llr with L = 5 fixed (svdn basis, energy weights kept). With
# gc_llr (svdn L 4) and gc_llr_fixed5 (fixed L 5): is the loss the rule's L or the svdn basis / energy weights?
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
GC_SEG="--nseg 5 --seg-basis svdn --seg-weight energy" bash $E/run_gc3.sh gc_llr_svdn5 3600
