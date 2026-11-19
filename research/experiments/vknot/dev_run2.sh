#!/bin/bash
# Dev run 2 (pre-registered 2026-11-18, research/velocity_model_sr.md §F): the parameter-spline velocity model
# (recon_p2np --vmodel knots: m, ψ, v in the 6 × 16 basis, cubic-interpolated operator frames, priors on, robust init)
# against complex knot images (splinerecon) with and without priors, at the same basis and data.
#
# QUESTION  Does the velocity model with priors recover the velocity phase (encode 1 − 0) at least as well as complex
#           knot images with their own wavelet + LLR prior, at --keep 8, in the deep lumen and the dynamics?
# DECISION  B2 ≥ A2 (slope and corr, deep lumen and dynamics, ≥ 2 of 3 breathing states) → the velocity model is the
#           recon path (next: full scan / 1 mm box, learned prior in (|z|, v)). B2 < A2 in the dynamics only → the
#           operator frames / the parameter basis (frames_caveat.json: F_op 48 leaves 0.0065 of the dynamic velocity)
#           or the priors' strength on v; B2 < A2 everywhere → the init or the solver (in-driver RMSE per iteration).
#
#   basis   6 linear respiratory knots (bellows quantiles) × 16 periodic cubic cardiac knots (the target)
#   data    ph0001 sim_flow_10min, 2 mm, --keep 8 (deviation: 1/8 of the readouts), 12 SV virtual coils (deviation:
#           82.8 % energy, target ≥ 99.9 %), physics full (B0 10 fixed segments + concomitant + GNL, degree-4 fit), W 10
#   arms    A  (exists, dev_run.sh A): splinerecon, encodes 0–1, warm 10 + knot-preconditioned CGLS 40, no prior
#           A2 splinerecon --arms ncgcap (wavelet db4 L3 + LLR B8, prior-aware metric), 41 gradients, encodes 0–1
#           B2 recon_p2np --vmodel knots (defaults: --vop frames:48, priors on), pyramid: level 1 at 4 mm (grid-scale
#              0.5, robust init = complex param-knot CGLS + per-knot dual-VENC decode + spline fit), level 2 at 2 mm
#              --init-from level 1
#   score   score_pair.py (encode 1 − 0 velocity-phase slope / corr per breathing state's own truth; depth 0–2, ≥ 5,
#           dynamics ≥ 2) on A, A2, B2; B2 also the in-driver velocity RMSE (truth for scoring only)
# Usage (detached, GPU serialised): dev_run2.sh <arm A2|B2a|B2b|score> <gpu lock file>
set -e
cd /home/turbotage/Documents/GitHub/HastyCompute
ARM=${1:-score}; LOCK=${2:-/dev/null}
# BIN: a snapshot of recon_p2np (other agents rebuild the workspace while a run is going)
BIN=${BIN:-rustlib/target/release/recon_p2np}
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
O=data/hasty/phantom/factory_arms/ph0001/vknot/dev_k8
I=data/hasty/phantom/factory_arms/ph0001/spline2d
mkdir -p $O
case $ARM in
A2)
  flock $LOCK rustlib/target/release/splinerecon $S --rknots 6 --rdeg 1 --cknots 16 --cdeg 3 --resp bellows --width 10 --nseg 10 --vcoils 12 \
    --phys conc --conc-fit deg4 --keep 8 --encodes 0,1 --warm 10 --precond knot --arms ncgcap --prior-dev cpu --llr-budget-gb 0.25 --budget 41 --save-at 10,20,40 \
    --eval "0.077:$I/idx_r0.07_f32.npy,0.2:$I/idx_r0.181_f32.npy,0.499:$I/idx_r0.493_f32.npy" --out $O/A2 > $O/A2.log 2>&1
  for f in $O/A2_*_r0.077_* ; do mv "$f" "${f/_r0.077_/_r0.07_}"; done
  for f in $O/A2_*_r0.2_* ; do mv "$f" "${f/_r0.2_/_r0.181_}"; done
  for f in $O/A2_*_r0.499_* ; do mv "$f" "${f/_r0.499_/_r0.493_}"; done
  ;;
B2a)
  # level 1: 4 mm (grid-scale 0.5, k-space cropped to the level's ball), robust init inside the driver
  flock $LOCK $BIN $S --vmodel knots --resp bellows --gnl --phys full --conc-fit deg4 \
    --nseg 10 --vcoils 12 --keep 8 --grid-scale 0.5 --kcrop --warm 10 --iters 40 --precond zmshr1m --every 5 --lam-iters 6 \
    --out $O/B2a >> $O/B2a.log 2>&1
  ;;
B2b)
  # level 2: 2 mm from level 1
  flock $LOCK $BIN $S --vmodel knots --resp bellows --gnl --phys full --conc-fit deg4 \
    --nseg 10 --vcoils 12 --keep 8 --init-from $O/B2a --iters 20 --precond zmshr1m --every 5 --lam-iters 4 --no-full-save \
    --eval "0.07@0.077:$I/idx_r0.07_f32.npy,0.181@0.2:$I/idx_r0.181_f32.npy,0.493@0.499:$I/idx_r0.493_f32.npy" --eval-every 10 \
    --out $O/B2b >> $O/B2b.log 2>&1
  ;;
score)
  P=$(ls $O/B2b/knots_*_r0.07_it*_lumen.* | head -1 | sed 's/_r0.07_it.*//')
  rustlib/.venv/bin/python -I research/experiments/spline2d/score_pair.py $O/A_spline $O/A2_ncgcap $P > $O/scores2.txt 2>&1
  grep -h "vk it" $O/B2a.log $O/B2b.log >> $O/scores2.txt
  ;;
esac
