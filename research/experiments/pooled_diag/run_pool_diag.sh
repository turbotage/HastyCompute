#!/bin/bash
# research/recon_diag_2026-11-22.md: the pooled (one image per encode, all 20 frames) solve on the noise-free inverse
# crime, static object (--synth-x pd) vs PD + blood dynamics (--synth-x pc), at several sampling levels (--keep K =
# 1/K of the readouts). Pooled PCG 60 iterations with the per-encode circulant; the image error vs x★ is printed every
# 5 iterations ("pooled image vs x★"), the pc velocity score at it 0 of a 1-iteration P²nP run.
# Usage: run_pool_diag.sh <stage> (test | a | b). Done markers skip finished runs; STOP: touch <run dir>/STOP.
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/pool_diag
PC=data/hasty/phantom/factory_arms/ph0001/tpyr_score/pc
B=rustlib/target/release/recon_p2np
PHYS="--gnl --phys full --conc-fit deg4 --nseg 6"
COM="--grid-scale 0.5 --kcrop --vcoils 8 --lam-method lanczos --report-detail --score-pc $PC --synthetic --plan host --warm 0"
gpu() { # gpu <name> <timeout s> <args...>
  local n=$1 t=$2; shift 2
  [ -f $R/$n/DONE ] && { echo "skip $n"; return; }
  mkdir -p $R/$n
  echo "== $n $(date +%T)"
  flock $LOCK timeout $t systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 env NOISE_SCALE=0 HASTY_PRECOND_DIAG=1 $EV $B $S "$@" --out $R/$n > $R/$n/log.txt 2>&1
  local rc=$?; echo "exit $rc" >> $R/$n/log.txt; [ $rc = 0 ] && ! grep -q "^stopped" $R/$n/log.txt && touch $R/$n/DONE
  echo "   $n exit $rc $(date +%T)"
}
pool() { # pool <name> <keep> <pd|pc> <extra...>: pooled 60 iterations, scored at it 0
  local n=$1 k=$2 x=$3; shift 3
  gpu $n 1200 $PHYS --precond ${PRE:-zmshr1m} $COM --keep $k --synth-x $x --warm-full ${IT:-60} --init warm --iters 1 --no-full-save --no-vel-save 1 "$@"
}
case $1 in
test)
  mkdir -p $R
  # the test binary of `cargo test --release -p hasty-recon --test grouped_sets --no-run` (no cargo in systemd units)
  T=$(ls -t rustlib/target/release/deps/grouped_sets-* | grep -v '\.d$' | head -1)
  for n in pooled_pcg grouped_setops_match_per_set; do
    flock $LOCK timeout 900 systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 $T --test-threads 1 --nocapture $n
  done > $R/test_grouped_sets.log 2>&1
  echo "test exit $?"
  ;;
a)
  pool a_k16_pd 16 pd
  pool a_k16_pc 16 pc
  pool a_k4_pd 4 pd
  pool a_k4_pc 4 pc
  pool a_k2_pd 2 pd
  pool a_k2_pc 2 pc
  ;;
b)
  # bisect at keep 16, static object, 30 pooled iterations: plain model (no ratemap/B0 segments, no concomitant, no GNL)
  # with the circulant, plain model with plain CG, full model with plain CG; --adj-test prints ⟨Ex,y⟩ vs ⟨x,Eᴴy⟩
  export IT=30
  PHYS="--phys none" pool b_plain 16 pd --adj-test
  # (b_plain_cg / b_full_cg of 2026-11-22 passed a second --precond that was ignored: they repeat b_plain / a_k16_pd)
  ;;
c)
  # plain CG (no metric) for both models; GNL alone; the ratemap/concomitant physics alone
  export IT=30
  PRE=none pool c_full_none 16 pd
  PRE=none PHYS="--phys none" pool c_plain_none 16 pd
  PHYS="--gnl --phys none" pool c_gnl 16 pd
  PHYS="--phys full --conc-fit deg4 --nseg 6" pool c_phys_nognl 16 pd
  ;;
d)
  # after the 3D support floor of the circulant (precond.rs): the full model with the circulant again
  export IT=30
  pool d_full_circ3d 16 pd
  ;;
e)
  # (d_full_circ3d: a 3D support floor of the circulant, since reverted: its edge level was below the ε floor, no
  # change.) A milder circulant (floor ε = 1e-2, 1e-1 of λ_max instead of 1e-3), then circulant vs plain CG at 1/4
  # and 1/2 of the readouts
  export IT=30
  EV=HASTY_PRECOND_EPS=1e-2 pool e_k16_eps1e2 16 pd
  EV=HASTY_PRECOND_EPS=1e-1 pool e_k16_eps1e1 16 pd
  pool e_k4_circ 4 pd
  pool e_k4_none 4 pd --pool-metric none
  pool e_k2_circ 2 pd
  pool e_k2_none 2 pd --pool-metric none
  ;;
esac
