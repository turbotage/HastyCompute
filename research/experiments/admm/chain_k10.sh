#!/bin/bash
# 2026-11-23, pre-registered. Question: with the data step solved better (K = 10 inner PCG steps per set instead of 3),
# do the no-metric and clip arms reach the same image, and does the x–z gap stop growing?
# Decision: same image (pc and held-out within ~1 %, small ‖z_none − z_clip‖) and bounded gap → the K = 3 differences were
# the inexact x-step; pick the metric by wall time. Otherwise the gap/held-out drift is not the x-step → next test is an
# increasing-ρ schedule.
# One factor changed vs chain_air2.sh: --admm-inner 3 → 10 (and 21 outer iterations, since each costs ~3× more).
cd /home/turbotage/Documents/GitHub/HastyCompute
A=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/admm/run_admm.sh
ADMM_NO_GATE=1 bash $A A10_none_air 10800 --iters 21 --every 3 --admm-metric none --admm-inner 10 --admm-rho 1 --air auto
ADMM_NO_GATE=1 bash $A B10_clip_air 10800 --iters 21 --every 3 --admm-metric clip --admm-inner 10 --admm-rho 1 --air auto
