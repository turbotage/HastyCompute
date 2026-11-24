#!/bin/bash
# 2026-11-24 01:05, pre-registered in results.md: which prior component loses? Gated cell (1 bin, 50 % quietest
# breathing, 20 cardiac frames), LLR k1 base (= c_base_llr_ga05) plus ONE component per run, static field, old binary.
# Lock: group mode at the documented μ 1.5 (research/static_lock.md) and at the μ 20 the dev runs used (a quadratic-mode value).
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
bash $E/run_ctrl.sh base k_ga05_wav 3600 --wavelet-mix 0.3
bash $E/run_ctrl.sh base k_ga05_dfw 3600 --dfw sure
bash $E/run_ctrl.sh base k_ga05_lock1p5 3600 --static-lock 1.5:10:5 --lock-mode group
bash $E/run_ctrl.sh base k_ga05_lock20 3600 --static-lock 20:10:5 --lock-mode group
