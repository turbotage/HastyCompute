#!/bin/bash
# §13 of temporal_pyramid.md: validated velocity score (--score-pc), held-out k-space (--holdout / --holdout-eval),
# controls (noise-free data, inverse crime with x★ = PD + truth blood, ≥ 50 iterations) and the nested anchor test.
# Dev scale as run_dev.sh. Usage: run_score.sh <stage> (smoke | diag | anchor). Every step is skipped when its
# done marker exists; STOP: touch <run dir>/STOP (recon_p2np checkpoints; rerun the same stage resumes).
cd /home/turbotage/Documents/GitHub/HastyCompute
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
S=data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan
R=data/hasty/phantom/factory_arms/ph0001/tpyr_score
PC=$R/pc   # pc_native.py output
B=rustlib/target/release/recon_p2np
COM="--gnl --phys full --conc-fit deg4 --grid-scale 0.5 --kcrop --keep 16 --vcoils 8 --nseg 6 --precond zmshr1m --lam-method lanczos --report-detail --score-pc $PC --holdout 8"
gpu() { # gpu <name> <timeout s> <args...>: one recon_p2np under the lock, in a memory-capped scope
  local n=$1 t=$2; shift 2
  [ -f $R/$n/DONE ] && { echo "skip $n"; return; }
  mkdir -p $R/$n
  echo "== $n $(date +%T)"
  flock $LOCK timeout $t systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 env HASTY_X=1 $EV $B $S "$@" --out $R/$n > $R/$n/log.txt 2>&1
  local rc=$?; echo "exit $rc" >> $R/$n/log.txt; [ $rc = 0 ] && ! grep -q "^stopped" $R/$n/log.txt && touch $R/$n/DONE
  echo "   $n exit $rc $(date +%T)"
}
ev() { # ev <name> <solve run dir> <recon npy>: held-out data term of one recon (f at it 1: x, then zero)
  EV= gpu $1 600 $COM_EV --holdout-eval 8 --cc-load $2/coil_basis.txt --init-from $3 --init warm,zero --iters 1 --every 1 --no-full-save
}
COM_EV=${COM/--holdout 8/}
EV=
case $1 in
smoke)
  gpu smoke 900 $COM --warm 3 --warm-full 2 --init warm --iters 1 --no-vel-save 1
  ev smoke_ev $R/smoke $(ls $R/smoke/recon_*.hv | head -1)
  EV=NOISE_SCALE=0 gpu smoke_crime 900 $COM --synthetic --synth-x pc --plan host --warm 3 --warm-full 2 --init warm --iters 1 --no-full-save
  ;;
diag)
  # measured data: pooled x̄ converged (60 pooled PCG), five starts, the short run (12 iterations)
  gpu meas12 1500 $COM --warm 3 --warm-full 60 --init warm,zero,meanres:1:circ,meanres:1:pcg3,meanres:2:circ --iters 12 --every 3 --no-full-save
  # noise-free measured data (HASTY_NOISE_SIGMA=0: the clean sim data, no noise added), 51 iterations
  EV=HASTY_NOISE_SIGMA=0 gpu clean 1500 $COM --warm 3 --warm-full 60 --init warm --iters 51 --every 3 --no-full-save
  # inverse crime, the recon's own operator, x★ = PD + truth blood (realistic object): noise-free, then at the measured SNR
  EV=NOISE_SCALE=0 gpu crime0 1800 $COM --synthetic --synth-x pc --plan host --warm 3 --warm-full 60 --init warm,zero --iters 51 --every 3 --no-full-save
  # noisy inverse crime at the measured SNR: noise scaled by the data-norm ratio √(f(0) synthetic / f(0) measured)
  F0S=$(grep "it   1: f" $R/crime0/log.txt | sed -n 2p | sed 's/.*f = \([0-9.e+]*\),.*/\1/')
  F0M=$(grep "it   1: f" $R/meas12/log.txt | sed -n 2p | sed 's/.*f = \([0-9.e+]*\),.*/\1/')
  NS=$(python3 -c "print(($F0S/$F0M)**0.5)"); echo "crime1 NOISE_SCALE $NS (f0 synthetic $F0S, measured $F0M)"
  EV=NOISE_SCALE=$NS gpu crime1 1200 $COM --synthetic --synth-x pc --plan host --warm 3 --warm-full 60 --init warm --iters 51 --every 3 --no-full-save
  # 51 iterations: the best two starts at it 12 by the pc score RMSE, plus the pooled start (the anchor test's reference)
  BEST=$(python3 - $R/meas12 <<'PY'
import csv, glob, os, sys
m = {"": "warm", "_zero": "zero", "_mr1c": "meanres:1:circ", "_mr1p3": "meanres:1:pcg3", "_mr2c": "meanres:2:circ"}
r = []
for f in glob.glob(os.path.join(sys.argv[1], "pcscore_seg_zmshr1m_k1*.csv")):
    suf = os.path.basename(f)[len("pcscore_seg_zmshr1m_k1"):-4]
    rows = [x for x in csv.DictReader(open(f)) if x["it"] == "12"]
    if suf in m and rows:
        r.append((float(rows[-1]["rmse"]), m[suf]))
best = [s for _, s in sorted(r)[:2]]
print(",".join(best + ([] if "warm" in best else ["warm"])))
PY
)
  echo "51-iteration starts: $BEST"
  gpu meas51 2400 $COM --warm 3 --warm-full 60 --init $BEST --iters 51 --every 3 --no-vel-save 1
  ;;
anchor)
  # coarse level: 5 nested cardiac frames (4 fine frames each), 51 iterations from the pooled start; then the 20-frame
  # level anchored to it (Theorem 5 case B), same start and iterations; the unanchored reference is meas51's warm run
  gpu coarse5 1500 $COM --card-frames 5 --warm 3 --warm-full 60 --init warm --iters 51 --every 3 --no-vel-save 1
  gpu fine20 2400 $COM --warm 3 --warm-full 60 --init warm --anchor $R/coarse5:10,$R/coarse5:100,$R/coarse5:1000 --iters 51 --every 3 --no-vel-save 1
  for f in $R/fine20/recon_*.hv $R/meas51/recon_*.hv $R/coarse5/recon_*.hv; do
    d=${f%/*}; b=$(basename $f .hv); ev ev_$(basename $d)_$b $d $f
  done
  ;;
k4)
  # 4x the data (--keep 4): is the inverse crime well posed there? then the five starts, 12 iterations
  C4=${COM/--keep 16/--keep 4}
  EV=NOISE_SCALE=0 gpu k4crime0 1500 $C4 --synthetic --synth-x pc --plan host --warm 3 --init zero --iters 30 --every 3 --no-full-save
  gpu k4meas12 1800 $C4 --warm 3 --warm-full 60 --init warm,zero,meanres:1:circ,meanres:1:pcg3,meanres:2:circ --iters 12 --every 3 --no-full-save
  ;;
anchor4)
  # Theorem 5 case B at --keep 4: coarse level = 5 nested cardiac frames (4 fine frames each) from the pooled start,
  # 30 iterations; the 20-frame level from the same pooled start, 51 iterations, without / with the anchor to it
  C4=${COM/--keep 16/--keep 4}
  gpu k4coarse5 1500 $C4 --card-frames 5 --warm 3 --warm-full 60 --init warm --iters 30 --every 3 --no-vel-save 1
  gpu k4fine20 2400 $C4 --warm 3 --warm-full 60 --init ${INIT4:-warm} --anchor none,$R/k4coarse5:100,$R/k4coarse5:1000 --iters 51 --every 3 --no-vel-save 1
  COM_EV=${C4/--holdout 8/}
  for f in $R/k4fine20/recon_*.hv $R/k4coarse5/recon_*.hv; do
    d=${f%/*}; b=$(basename $f .hv); ev ev_$(basename $d)_$b $d $f
  done
  ;;
esac
