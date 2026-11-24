#!/bin/bash
# 2026-11-23 22:40: per_set_segs test passed with the control-engine floor (per_set_segs_test2.log); start the
# P²nP reference (g_esp_air config) with the TRUE ratemap per frame (--ratemap truth-bc, frame mode).
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
RG_BIN=rustlib/target/admm/snap_rm/recon_p2np RG_MEM=40G bash $E/run_guard.sh g_esp_air_rm 9000 espirit --precond zmshr1 --iters 60 --step-guard 2 --air auto --ratemap truth-bc
