#!/bin/bash
# research/precond_estimator.md §5 (E1–E4): estimator-preserving pooled preconditioners on the solvable dev case
# (research/recon_diag_2026-11-22.md §5): ph0001 sim_flow_10min, 128 × 72 × 60 grid, kcrop, 8 virtual coils, full
# physics (6 ratemap/B0 segments, concomitant degree 4, GNL), 20 cardiac frames × 7 encodes, --keep 2 = 26 674 readouts,
# 162–214 per (frame, encode) set (pooled per encode: 6.3 samples per voxel). Noise-free inverse crime.
#
# Stages (each skips when its DONE marker exists; STOP: touch <run dir>/STOP, a rerun repeats the stage):
#   build  CPU: release build of recon_p2np + the hasty-recon tests (-j 4), CPU unit tests of ppre / lanczos
#   test   GPU: ppre_gpu (device metric = host reference, deflation), precond_host (touched precond.rs)
#   e1     static object, μ = 0, pooled PCG of encode 0, 40 iterations of circ / none / p1 at field scale 0, 0.5, 1:
#          Ritz report of P Q (κ_eff, where the top / bottom Ritz vectors sit), probed vs nominal symbol per k shell
#   e2     static object, ridge μ ∈ {1e-2, 1e-3} λ̄, encode 0: every candidate to ‖r‖/‖r₀‖ ≤ 1e-6 (cap 200), distance to
#          the reference x̂ (p1n + 16-vector deflation to 1e-7) → Q1 (same image?) and Q2 (iterations to 1 % / 0.1 %)
#   e2b    as e2 at μ = 1e-3 with the band-limit to the sampled ball (estimator variant)
#   e3     dynamic object (PD + blood), μ = 1e-3: the short list
#   pnp    per-frame P²nP, LLR κ 1, 60 iterations, --score-pc: (A) today's pipeline (pooled plain CG 60 + zmshr1),
#          (B) pooled p1n start (μ = 1e-3) + zmshr1m in the p1n metric with 16-vector deflation, and zmshr1
#   e4     the best variant at field scale 0.5 (E4; field scale 1 = e2)
# Usage: run_campaign.sh <stage>.  GPU budget (estimated): test 2, e1 6, e2 20, e2b 6, e3 8, pnp 23, e4 4 min.
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/precond_est
PC=data/hasty/phantom/factory_arms/ph0001/tpyr_score/pc
B=rustlib/target/release/recon_p2np
PHYS="--gnl --phys full --conc-fit deg4 --nseg 6"
COM="--grid-scale 0.5 --kcrop --keep 2 --vcoils 8 --lam-method lanczos --report-detail --score-pc $PC --synthetic --plan host --warm 0 --no-full-save --no-vel-save 1"
gpu() { # gpu <name> <timeout s> <args...>
  local n=$1 t=$2; shift 2
  [ -f $R/$n/DONE ] && { echo "skip $n"; return; }
  mkdir -p $R/$n
  echo "== $n $(date +%T)"
  flock $LOCK systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 timeout $t env NOISE_SCALE=0 HASTY_PRECOND_DIAG=1 $EV $B $S "$@" --out $R/$n > $R/$n/log.txt 2>&1
  local rc=$?; echo "exit $rc" >> $R/$n/log.txt; [ $rc = 0 ] && ! grep -q "^stopped" $R/$n/log.txt && touch $R/$n/DONE
  echo "   $n exit $rc $(date +%T)"
}
study() { # study <name> <timeout> <pd|pc> <runs> <extra...>
  local n=$1 t=$2 x=$3 runs=$4; shift 4
  gpu $n $t $PHYS $COM --precond zmshr1m --synth-x $x --iters 1 --pp-study "$runs" "$@"
}
case $1 in
build)
  cd rustlib
  nice cargo build --release -j 4 -p hasty-bench --bin recon_p2np && \
  nice cargo test --release -j 4 -p hasty-recon --lib -- ppre lanczos && \
  nice cargo test --release -j 4 -p hasty-recon --test ppre_gpu --test precond_host --no-run
  ;;
test)
  mkdir -p $R
  for t in ppre_gpu precond_host; do
    T=$(ls -t rustlib/target/release/deps/$t-* | grep -v '\.d$' | head -1)
    flock $LOCK systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 timeout 600 $T --test-threads 1 --nocapture
  done > $R/test_gpu.log 2>&1
  echo "test exit $?"
  ;;
e1)
  RUNS="circ=circ:40;none=none:40;p1=p1:40;p1n=p1n:40"
  for fs in 1 0.5 0; do
    study e1_fs$fs 900 pd "$RUNS" --field-scale $fs
  done
  ;;
e2)
  # (after e1: J = 8 probes are noise-dominated per mode, split-half 0.96 at radius 2 → reference / deflation base p0,
  # p1n at smoothing radius 6; p1, p12 dropped)
  # (2026-11-22 partial e2 at μ 1e-3: every Fourier metric stalls at ‖r‖/‖r₀‖ ~1e-2 and the image error grows; plain CG
  # and Jacobi p2 reach 2e-4 / 0.087 in 150 iterations → the reference is now Jacobi to 1e-6; the dead Fourier
  # variants are kept short as controls)
  RUNS="ref=p2:700:1e-6;none=none:300:1e-6;p2=p2:300:1e-6;circ=circ:60;p0=p0:60;p1n=p1n:60"
  study e2_pd 2700 pd "$RUNS" --pp-mu 1e-3,1e-2 --pp-smooth 6
  ;;
e5)
  # bounded-dynamic-range circulant pe<ε>_<p>: P = D^{-1/2} F⁻¹ (λ_nom/λ_max + ε)^{-p} F D^{-1/2}, ε ∈ {1e-2, 3e-2,
  # 1e-1, 3e-1}, p ∈ {0.5, 0.75}; ridge μ 1e-3 λ̄ (the e2 estimator); then deflation r = 8, 32 on the best (BEST)
  # (shortened 2026-11-22 for the budget: μ 1e-2, 100 iterations each, compared at equal iterations with plain CG /
  # Jacobi; no converged reference exists, e2 / §6.0c)
  R5="none=none:100;p2=p2:100"
  for e in 1e-2 3e-2 1e-1 3e-1; do for q in 0.5 0.75; do R5="$R5;pe${e}_$q=pe${e}_$q:100"; done; done
  study e5_pd 900 pd "$R5" --pp-mu 1e-2 --pp-harvest 0
  ;;
e5d)
  BEST=${BEST:?BEST=pe<ε>_<p> from e5}
  study e5d_pd 1500 pd "ref=p2:700:1e-6;b=$BEST:60;bd8=$BEST+d8:200:1e-6;bd32=$BEST+d32,4:200:1e-6;bb=$BEST:200:1e-6" --pp-mu 1e-3 --pp-harvest 60
  ;;
e2b)
  RUNS="p1n=p1n:60;ref=p1n+d16:400:1e-7;none=none:200:1e-6;circ=circ:200:1e-6;p1nb=p1n:200:1e-6"
  study e2b_pd_band 1200 pd "$RUNS" --pp-mu 1e-3 --pp-band
  ;;
e3)
  # dynamic object (PD + blood), μ 1e-3: reference Jacobi, plain CG, and the winner WIN (default p2 = Jacobi)
  WIN=${WIN:-p2}
  RUNS="ref=p2:700:1e-6;none=none:300:1e-6;win=$WIN:300:1e-6"
  study e3_pc 1500 pc "$RUNS" --pp-mu 1e-3
  ;;
g1)
  # §7.4.1: the E1 Ritz report against the shift map s = Δt |∇f| / k_max (s > 0.3), current circulant, μ = 0
  study g1_fs1 600 pd "circ=circ:40;none=none:40"
  ;;
g2)
  # §7.4.2: free recentring U P Uᴴ of the circulant (and of a bounded one), ridge 1e-3, Jacobi reference
  study g2_pd 1200 pd "ref=p2:700:1e-6;circu=circ+u:150:1e-6;pe1u=pe1e-1_0.5+u:150:1e-6;circ=circ:60" --pp-mu 1e-3
  ;;
g3)
  # §7.4.3, the main candidate: coil + segment surrogate, flexible outer PCG; plain CG and Jacobi as the bars to beat
  # (accept: wall time to 1e-3 of x̂ ≥ 2× better than both)
  # (after e2: at μ 1e-3 Jacobi does not converge in 700 iterations (‖r‖/‖r₀‖ 7.6e-5) → μ 1e-2, and the surrogate
  # run itself is the reference (any converged SPD solve is x̂); time to a residual level = iterations × (ms Q + ms P))
  study g3_pd 1500 pd "ref=sur3k10:150:1e-6;none=none:150:1e-6;p2=p2:150:1e-6;c3k10=cheb3k10:100:1e-6;s3k5=sur3k5:100:1e-6" --pp-mu 1e-2
  ;;
g3chk)
  # the surrogate against the TRUE operator N (the setup check ‖(N − cÑ)x‖/‖Nx‖ over 8 conventions), L_s = 3, 4, 6,
  # concomitant fields on (deg 4, the dev model) and off (--phys b0: B0 segments only, GNL kept)
  study g3chk_full 600 pd "c3=sur3k1:1;c4=sur4k1:1;c6=sur6k1:1" --pp-mu 1e-2
  PHYS="--gnl --phys b0 --nseg 6" study g3chk_b0 600 pd "c3=sur3k1:1;c6=sur6k1:1" --pp-mu 1e-2
  ;;
g3gnl)
  # isolate GNL: b0 segments WITHOUT GNL, and the plain model WITH GNL (the passing plain check had neither)
  PHYS="--phys b0 --nseg 6" study g3chk_b0_nognl 600 pd "c6=sur6k1:1" --pp-mu 1e-2
  PHYS="--gnl --phys none" study g3chk_plain_gnl 600 pd "c1=sur1k1:1" --pp-mu 1e-2
  ;;
g3dec)
  # decomposition check of the stored segment tables (b0, no GNL)
  PHYS="--phys b0 --nseg 6" study g3chk_b0_nognl 600 pd "c6=sur6k1:1" --pp-mu 1e-2
  ;;
g3pc)
  study g3_pc 1500 pc "ref=p2:700:1e-6;none=none:300:1e-6;p2=p2:300:1e-6;s3k10=sur3k10:150:1e-6" --pp-mu 1e-3
  ;;
g4)
  # §7.4.4: deflation on top of the fixed (Chebyshev) surrogate preconditioner
  study g4_pd 1500 pd "ref=p2:700:1e-6;c3=cheb3k10:60;c3d8=cheb3k10+d8:150:1e-6;c3d32=cheb3k10+d32,4:150:1e-6" --pp-mu 1e-3 --pp-harvest 60
  ;;
pnp)
  # (A) today's best pipeline at 60 iterations: pooled plain CG start, ZMSHR1 without metric
  gpu pnp_A 1500 $PHYS $COM --synth-x pc --precond zmshr1 --warm-full 60 --pool-metric none --init warm --iters 60 --every 3
  # (B) pooled start with the winner WIN of e2 / g3 (ridge 1e-3 λ̄, ${PP_IT:-60} iterations), then ZMSHR1 without
  # metric (the Fourier metrics failed in e2; a surrogate inside P²nP needs the Chebyshev form, §7.4.6)
  gpu pnp_B 2100 $PHYS $COM --synth-x pc --precond zmshr1 --warm-full ${PP_IT:-60} --pp-pool ${WIN:-p2} --pp-mu 1e-3 --init warm --iters 60 --every 3
  ;;
e4)
  # the best variant at field scale 0.5 (edit BEST after e2)
  BEST=${BEST:-p1n}
  study e4_fs0.5 1200 pd "$BEST=$BEST:60;ref=$BEST+d16:400:1e-7;cand=$BEST:200:1e-6;none=none:200:1e-6" --pp-mu 1e-3 --field-scale 0.5
  ;;
esac
