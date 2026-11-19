#!/bin/bash
# P1–P2 solver arms (research/convergence_strategy.md § Results P1–P2): one undersampling level = one splinerecon run
# over encodes 0–1 with every arm. Usage: run_level.sh <tag> <keep N> [budget] [arms] [extra splinerecon args...]
# ENCODES=0 for one encode. Run as a detached user service (survives an editor disconnect), host RAM capped:
#   systemd-run --user --unit=p12_<tag> --collect -p MemoryMax=60G -p MemorySwapMax=0 \
#     --working-directory=$PWD research/experiments/convergence_p12/run_level.sh <tag> <keep>
cd /home/turbotage/Documents/GitHub/HastyCompute
O=data/hasty/phantom/factory_arms/ph0001/spline2d
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
EV="0.07:$O/idx_r0.07_f32.npy,0.181:$O/idx_r0.181_f32.npy,0.493:$O/idx_r0.493_f32.npy"
TAG=$1; KEEP=$2; BUDGET=${3:-41}; ARMS=${4:-}; ARMS=${ARMS:-pcg,ncg,ncg0,ncgcap,ncgI,ncgc,zmshr1,fista,admm0.3,admm1,admm3,admm10}
shift 4 2>/dev/null || shift $#
exec rustlib/target/release/splinerecon $S --rknots 4 --rdeg 1 --cknots 10 --resp truth --coils 8 --no-gnl --seg-basis fixed --nseg 10 --conc-fit quad --warm 10 --precond knot --heldout 20 --encodes ${ENCODES:-0,1} \
  --phys b0 --eval $EV --keep $KEEP --arms $ARMS --budget $BUDGET --save-at 5,10,20,40 "$@" --out $O/p12_$TAG > $O/p12_$TAG.log 2>&1
