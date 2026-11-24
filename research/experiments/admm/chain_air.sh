#!/bin/bash
# 2026-11-23: every arm with the air prior (--air auto: air map from the iterate, zeroed in every frame before and after
# the prior). Sequential so the order is fixed. Reference P²nP (ESPIRiT), then ADMM none / clip, then P²nP true maps.
cd /home/turbotage/Documents/GitHub/HastyCompute
G=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/precond_compare/run_guard.sh
A=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/admm/run_admm.sh
bash $G g_esp_air 7200 espirit --precond zmshr1 --iters 60 --step-guard 2 --air auto
ADMM_NO_GATE=1 bash $A A_none_air 10800 --iters 30 --every 3 --admm-metric none --admm-inner 3 --admm-rho 1 --air auto
ADMM_NO_GATE=1 bash $A B_clip_air 10800 --iters 30 --every 3 --admm-metric clip --admm-inner 3 --admm-rho 1 --air auto
bash $G g_truth_air 7200 truth --precond zmshr1 --iters 60 --step-guard 2 --air auto
