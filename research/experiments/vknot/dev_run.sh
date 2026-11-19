#!/bin/bash
# Dev run (pre-registered 2026-11-18): does the knot velocity model (recon_p2np --vmodel knots) recover velocity better
# than complex knot images (splinerecon) at the same basis and data?
#   basis   6 linear respiratory knots (bellows quantiles) × 16 periodic cubic cardiac knots (the target)
#   data    ph0001 sim_flow_10min, 2 mm, the same --keep N readout subset in both arms (hash of the readout index)
#   physics B0 (10 fixed segments) + concomitant + GNL + degree-4 concomitant fit, W 10, truth maps; coils: 12 SV-
#           compressed virtual coils in both arms (deviation from the target ≥ 99.9 % = 31 coils: the knot operator's FFT
#           work is K · L · coils per encode, 4 mm smoke 48.6 s per gradient at 31 coils)
#   arms    A: splinerecon, encodes 0–1, warm 10 + knot-preconditioned CGLS (G⁻¹ ⊗ C⁻¹), 40 iterations, no prior
#           B: recon_p2np --vmodel knots, all 7 encodes, warm 10 (SENSE PCGLS), P²nP ZMSHR1-M in the Gauss–Newton
#              metric (per-knot Gram weights), VENC continuation 10 + 30 iterations, no prior
#   score   score_pair.py: encode 1 − 0 velocity-phase slope / corr against each breathing state's own truth
#           (r 0.07 / 0.181 / 0.493, evaluated at their median bellows amplitudes 0.077 / 0.200 / 0.499), deep lumen
#           (depth ≥ 5), edge band (0–2), dynamics (depth ≥ 2); B also: in-driver velocity RMSE (knot interpolant,
#           20 frames, median breathing amplitude) — truth used for scoring only
#   decision B ≥ A in the dynamics and the deep lumen at every state → the knot velocity model is the path (next:
#           priors + knot lock at this scale); B < A in the dynamics only → the between-knot interpolation caveat
#           (interp_caveat.py: 16 cubic knots 0.52 vs complex knots 0.13 relative) dominates → more knots / operator
#           frames before priors; B < A everywhere → diagnose the solver (convergence, metric) first
# Usage (detached, GPU serialised): research/experiments/vknot/dev_run.sh <keep N> <arm A|B> <lock file>
set -e
cd /home/turbotage/Documents/GitHub/HastyCompute
KEEP=${1:-8}; ARM=${2:-B}; LOCK=${3:-/dev/null}
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
O=data/hasty/phantom/factory_arms/ph0001/vknot/dev_k$KEEP
I=data/hasty/phantom/factory_arms/ph0001/spline2d
mkdir -p $O
if [ "$ARM" = A ]; then
  flock $LOCK rustlib/target/release/splinerecon $S --rknots 6 --rdeg 1 --cknots 16 --cdeg 3 --resp bellows --width 10 --nseg 10 --vcoils 12 \
    --phys conc --conc-fit deg4 --keep $KEEP --encodes 0,1 --warm 10 --precond knot --iters 40 --save-at 10,20,40 \
    --eval "0.077:$I/idx_r0.07_f32.npy,0.2:$I/idx_r0.181_f32.npy,0.499:$I/idx_r0.493_f32.npy" --out $O/A_spline > $O/A_spline.log 2>&1
  # files carry the bellows amplitude: rename to the true-state labels of score_pair.py
  for f in $O/A_spline_r0.077_* ; do mv "$f" "${f/_r0.077_/_r0.07_}"; done
  for f in $O/A_spline_r0.2_* ; do mv "$f" "${f/_r0.2_/_r0.181_}"; done
  for f in $O/A_spline_r0.499_* ; do mv "$f" "${f/_r0.499_/_r0.493_}"; done
else
  flock $LOCK rustlib/target/release/recon_p2np $S --vmodel knots --knots2d 6:1,16:3 --resp bellows --gnl --phys full --conc-fit deg4 \
    --nseg 10 --vcoils 12 --keep $KEEP --warm 10 --iters 40 --vcont 10 --precond zmshr1m --every 5 --lam-iters 6 --no-full-save \
    --eval "0.07@0.077:$I/idx_r0.07_f32.npy,0.181@0.2:$I/idx_r0.181_f32.npy,0.493@0.499:$I/idx_r0.493_f32.npy" --eval-every 20 \
    --out $O/B_vknot > $O/B_vknot.log 2>&1
fi
