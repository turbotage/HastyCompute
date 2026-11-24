#!/bin/bash
# 2026-11-23: rerun of the two ADMM arms of chain_air.sh after both were OOM-killed at the 24G cgroup cap.
# Same arguments; cap 32G (run_admm.sh ADMM_MEM); the pooled start is reloaded from warm_full_done checkpoints.
cd /home/turbotage/Documents/GitHub/HastyCompute
A=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/admm/run_admm.sh
ADMM_NO_GATE=1 bash $A A_none_air 10800 --iters 30 --every 3 --admm-metric none --admm-inner 3 --admm-rho 1 --air auto
ADMM_NO_GATE=1 bash $A B_clip_air 10800 --iters 30 --every 3 --admm-metric clip --admm-inner 3 --admm-rho 1 --air auto
