#!/bin/bash
# 2026-11-23 late: 2×2 factorial binning × prior (see run_ctrl.sh), queued behind g_esp_air_rm on the GPU lock.
cd /home/turbotage/Documents/GitHub/HastyCompute
E=research/experiments/precond_compare
bash $E/run_ctrl.sh basefull f_ga05_full 3600
bash $E/run_ctrl.sh r4llr f_r4_llr 7200
