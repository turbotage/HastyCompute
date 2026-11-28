#!/usr/bin/env bash
# Tethering of the cardiac and respiratory motion (2026-11-28): factory_test18/ph0001's motion regenerated with
# --tether off (as before) and on, into data/hasty/scratch/motion_tether/{off,on}, then the vessel kinematics check.
# CPU only (the GPU runs the LBM); one job at a time.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
P=data/hasty/phantom/factory_test18/ph0001
O=data/hasty/scratch/motion_tether
T=rustlib/tools/phantom
TT=rustlib/crates/hasty-phantom/data/atlas_tissues_3t.json
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= rustlib/.venv/bin/python -I"
for m in ${MODES:-off on}; do
  mkdir -p $O/$m
  $PY $T/cardiac_motion.py $P/atlas/tissue_2mm.hv $TT $P $P/flow $O/$m --dav 15.74324723568622 --twist 7.870988712062912 --fe 0.671623941907945 --tether $m > jobs/logs/motion_tether_card_$m.log 2>&1 || { echo "cardiac $m failed"; tail -5 jobs/logs/motion_tether_card_$m.log; exit 1; }
  grep -E "tether|outside the heart|EF" jobs/logs/motion_tether_card_$m.log | tail -4
  $PY $T/resp_motion.py $P/atlas/tissue_2mm.hv $O/$m --dsi 13.297317164990922 --dap 7.942143517142021 --dlr 2.60638965858329 --graph $P/graph.npz --tether $m > jobs/logs/motion_tether_resp_$m.log 2>&1 || { echo "resp $m failed"; tail -5 jobs/logs/motion_tether_resp_$m.log; exit 1; }
  grep -E "tether" jobs/logs/motion_tether_resp_$m.log | tail -2
  # the vessel check reads <phantom>/motion: a view of the phantom with this motion
  V=$O/$m/phantom
  mkdir -p $V/motion
  ln -sf "$(realpath $P/graph.json)" $V/graph.json
  ln -sf "$(realpath $P/graph.npz)" $V/graph.npz
  ln -sf "$(realpath $O/$m/cardiac_motion.hv)" $V/motion/cardiac_motion.hv
  ln -sf "$(realpath $O/$m/resp_motion.hv)" $V/motion/resp_motion.hv
  rustlib/.venv/bin/python -I jobs/tools/vessel_motion_check.py $V $O/$m/check 2>&1 | tail -11 | cut -c1-330
done
echo motion tether done
