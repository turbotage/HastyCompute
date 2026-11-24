#!/bin/bash
# 2026-11-24, pre-registered in results.md "Exact-time temporal model": one factor vs gc_llr_rm (binned 20 frames,
# per-frame true ratemap, LLR, gated): the time model. Knot-major engine, per-knot true ratemaps, scored on frame box
# averages. Binary snap_t1 (reviewed; self-check on two encodes with a one-knot-shift control).
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
bash $E/run_t.sh t_cubic_rm 7200 --temporal cubic --ratemap truth-bc
bash $E/run_t.sh t_linear_rm 7200 --temporal linear --ratemap truth-bc
