#!/usr/bin/env bash
# Breathing tether, arteries only + solid organs kept (2026-11-28): factory_test18/ph0001's respiratory motion with the vessel tether limited to
# arteries (veins move with their tissue) into data/hasty/scratch/motion_tether/root (cardiac from motion_root.sh).
# Question: does the intrahepatic IVC follow the liver (≥ 70 % of the surrounding liver's SI motion), the aorta stay
# static (≤ 1 mm), no kink > 5° / stretch < 0.85 along the IVC, det J ≥ 0.24?
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
P=data/hasty/phantom/factory_test18/ph0001
O=data/hasty/scratch/motion_tether
T=rustlib/tools/phantom
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= rustlib/.venv/bin/python -I"
m=root
$PY $T/resp_motion.py $P/atlas/tissue_2mm.hv $O/$m --dsi 13.297317164990922 --dap 7.942143517142021 --dlr 2.60638965858329 --graph $P/graph.npz --tether on > jobs/logs/motion_tether_resp_$m.log 2>&1 || { echo "resp failed"; tail -5 jobs/logs/motion_tether_resp_$m.log; exit 1; }
grep -E "tether|det" jobs/logs/motion_tether_resp_$m.log | tail -3 | cut -c1-300
V=$O/$m/phantom
ln -sf "$(realpath $O/$m/resp_motion.hv)" $V/motion/resp_motion.hv
rustlib/.venv/bin/python -I jobs/tools/vessel_motion_check.py $V $O/$m/check 2>&1 | tail -11 | cut -c1-330
echo motion resp done
rustlib/.venv/bin/python -I jobs/tools/resp_det.py $O/$m/resp_motion.hv $P/atlas/tissue_2mm.hv
