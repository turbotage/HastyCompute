#!/usr/bin/env bash
cd /home/turbotage/Documents/GitHub/HastyCompute
until grep -q "campaign6 done" jobs/logs/campaign6.log 2>/dev/null; do sleep 30; done
S=data/hasty/scratch/clr_lca
flock jobs/locks/gpu.lock systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 rustlib/.venv/bin/python -I rustlib/tools/phantom/clearance.py $S --watch "Left coronary artery|connector: Ascending aorta" > jobs/logs/clr_watch2.log 2>&1
echo "clr_watch2 done" >> jobs/logs/clr_watch2.log
