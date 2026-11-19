#!/bin/bash
# Chain of the pre-registered dev run 2 arms B2a (4 mm, robust init) → B2b (2 mm, --init-from B2a) → score.
# Each arm takes flock on the GPU lock inside dev_run2.sh; rerunning this script resumes (finished stages are skipped
# by recon_p2np's checkpoints; pause with `touch <out>/STOP`).
#   usage: dev_run2_chain.sh <gpu lock file>
cd /home/turbotage/Documents/GitHub/HastyCompute
O=data/hasty/phantom/factory_arms/ph0001/vknot/dev_k8
export BIN=$O/recon_p2np_vspline
L=$1
research/experiments/vknot/dev_run2.sh B2a $L || exit 1
grep -q "stopped " $O/B2a.log && exit 0
research/experiments/vknot/dev_run2.sh B2b $L || exit 1
grep -q "stopped " $O/B2b.log && exit 0
research/experiments/vknot/dev_run2.sh score
