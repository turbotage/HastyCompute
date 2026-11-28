#!/usr/bin/env bash
# Deterministic blood sub-source packing (bsig det = 1, default), dev input with the beat + coronary library, 1 s of
# scan in 40-bin chunks, navigator on; the run's own self-checks (prefix sum of the first slots, total per bin) assert.
#  c1           resp field, 1 worker                                     → the self-checks pass
#  n1 vs n2     no resp field (fits two workers on one 24 GB card), 1 vs 2 workers → bit-identical
#  b1 vs b2     --blood-blend on --blood-advect off (two frames per bin: the frame offset), two runs → bit-identical
# Results so far (2026-11-27, first version): two det runs bit-identical; det vs SIMPH_BSIG_ATOMIC=1 rms 1.2e-6 of rms.
# Outputs: data/hasty/scratch/bsig_det (program output); logs: jobs/logs.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
P=data/hasty/phantom/factory_arms/ph0001
SF=$P/sim_flow_dev_respfield
O=data/hasty/scratch/bsig_det
L=jobs/logs
LOCK=jobs/locks/gpu.lock
NEW=rustlib/target/release/simphantom
export HASTY_IO=npy SIMPH_CHUNK_BINS=40
BASE=(--seq flow --flow-run data/hasty/seq/flow_thorax --flip 12 --scan-time 1.0 --beats "$P/lbm/run_lib" --beats-ref "$P/lbm/run_cor" --beat-cache 2)
RUN="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0"
rm -rf "${O:?}"
mkdir -p "$O"
run() { # tag args…
  local t=$1; shift
  /usr/bin/time -f "$t wall %e s" flock "$LOCK" $RUN $NEW "$SF/sim_input.hv" "$O/$t" "${BASE[@]}" "$@" > "$L/bsig_$t.log" 2>&1
  echo "$t: $(grep -E '^simulation ' "$L/bsig_$t.log") $(grep -m1 -oE 'packed to [0-9.]+ %' "$L/bsig_$t.log") $(grep -A1 'panicked' "$L/bsig_$t.log" | head -2 | tr '\n' ' ')"
}
run c1 --resp-field "$SF/resp_field.hv" --gpus 0
run n1 --gpus 0
run n2 --gpus 0,0
run b1 --blood-blend on --blood-advect off --gpus 0
run b2 --blood-blend on --blood-advect off --gpus 0
rustlib/.venv/bin/python -I - "$O" <<'EOF'
import sys, os, numpy as np
O = sys.argv[1]
def ulps(a, b):
    ai = a.view(np.int32).astype(np.int64); bi = b.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, -2**31 - ai, ai); bi = np.where(bi < 0, -2**31 - bi, bi)
    return np.abs(ai - bi)
def cmp(x, y):
    for n in ["data_clean.npy", "nav_encode.npy", "nav_tail.npy"]:
        fa, fb = os.path.join(O, x, n), os.path.join(O, y, n)
        if not (os.path.exists(fa) and os.path.exists(fb)):
            print(f"{x} vs {y} {n}: MISSING"); continue
        a = np.load(fa).view(np.float32).ravel(); b = np.load(fb).view(np.float32).ravel()
        d = np.abs(a - b); r = np.sqrt(np.mean(a.astype(np.float64) ** 2)); u = ulps(a, b)
        print(f"{x} vs {y} {n}: max |diff| {d.max() / r:.2e} of rms, rms diff {np.sqrt(np.mean(d.astype(np.float64)**2)) / r:.2e}, differing {100 * np.mean(u > 0):.3f} %, max ulp {u.max()}")
cmp("n1", "n2"); cmp("b1", "b2"); cmp("n1", "b1")
EOF
echo bsig test done
