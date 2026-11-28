#!/usr/bin/env bash
# Great-artery root hold (2026-11-28): factory_test18/ph0001's cardiac motion with the tether + root hold (default now)
# into data/hasty/scratch/motion_tether/root; breathing reused from .../on; vessel kinematics check against .../on.
# Question: does the aortic root move ≈ 0.77 / 0.45 / 0.25 × the AV descent at the annulus / ostia / mid-ascending
# (CMR 11.6 mm, CTA 6.2 / 3.8 mm at ≈ 15 mm descent) with the ascending-aorta stretch ≤ 10 %, chambers still exact?
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
P=data/hasty/phantom/factory_test18/ph0001
O=data/hasty/scratch/motion_tether
T=rustlib/tools/phantom
TT=rustlib/crates/hasty-phantom/data/atlas_tissues_3t.json
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES= rustlib/.venv/bin/python -I"
m=root
mkdir -p $O/$m
$PY $T/cardiac_motion.py $P/atlas/tissue_2mm.hv $TT $P $P/flow $O/$m --dav 15.74324723568622 --twist 7.870988712062912 --fe 0.671623941907945 --tether on "$@" > jobs/logs/motion_tether_card_$m.log 2>&1 || { echo "cardiac $m failed"; tail -5 jobs/logs/motion_tether_card_$m.log; exit 1; }
grep -E "tether|root hold|outside the heart|EF|det " jobs/logs/motion_tether_card_$m.log | grep -v Deprecation | head -8
V=$O/$m/phantom
mkdir -p $V/motion
ln -sf "$(realpath $P/graph.json)" $V/graph.json
ln -sf "$(realpath $P/graph.npz)" $V/graph.npz
ln -sf "$(realpath $O/$m/cardiac_motion.hv)" $V/motion/cardiac_motion.hv
ln -sf "$(realpath $O/on/resp_motion.hv)" $V/motion/resp_motion.hv
rustlib/.venv/bin/python -I jobs/tools/vessel_motion_check.py $V $O/$m/check 2>&1 | tail -11 | cut -c1-330
echo motion root done
rustlib/.venv/bin/python -I jobs/tools/aorta_root_profile.py $V 2>&1 | grep -v "aorta path"
rm -rf "${V:?}/_tmp_check"
