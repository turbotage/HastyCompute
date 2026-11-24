#!/bin/bash
# 2026-11-24: dry start of the --temporal additions (gating, held-out, ConstOverFrames warm start, per-knot true
# ratemaps on the knot-major engine, box-average scoring, startup self-check): 2 warm-start + 2 solver iterations.
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
GC_ITERS=2 GC_WARM=2 bash $E/run_t.sh dry_t_cubic_rm 2400 --temporal cubic --ratemap truth-bc
