#!/usr/bin/env bash
# simphantom multi-GPU equivalence, part 2:
#  (r)  finish the interrupted n2r run (2 of 3 chunks done by 2 workers) with 1 worker → compare with n1;
#  (a)  the old binary twice with the beat library (old vs old2): the run-to-run spread of the existing code
#       (bsig packs blood sub-sources through an atomic counter: source order, hence f32 rounding, varies);
#  (b)  no beat library (deterministic sources): old_nb vs n1_nb vs n2_nb must agree to ≈ 1 ulp (f64 atomic order).
# Outputs in data/hasty/scratch/mgpu (program output), logs in jobs/logs.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
P=data/hasty/phantom/factory_arms/ph0001
SF=$P/sim_flow_dev_respfield
O=data/hasty/scratch/mgpu
L=jobs/logs
LOCK=jobs/locks/gpu.lock
NEW=rustlib/target/release/simphantom
OLD=rustlib/target/snap/simphantom_old
export HASTY_IO=npy SIMPH_CHUNK_BINS=40
BASE=(--seq flow --flow-run data/hasty/seq/flow_thorax --flip 12 --scan-time 1.0 --resp-field "$SF/resp_field.hv")
BEATS=(--beats "$P/lbm/run_lib" --beats-ref "$P/lbm/run_cor" --beat-cache 2)
RUN="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0"
run() { # tag binary args…
  local t=$1 b=$2; shift 2
  /usr/bin/time -f "$t wall %e s, max RSS %M kB" flock "$LOCK" $RUN $b "$SF/sim_input.hv" "$O/$t" "$@" > "$L/mgpu_$t.log" 2>&1
  tail -1 "$L/mgpu_$t.log"; grep -E "chunks of|^  bins|simulation |panicked|rror" "$L/mgpu_$t.log" | head -8
}
run n2r "$NEW" "${BASE[@]}" "${BEATS[@]}" --gpus 0
rm -rf "${O:?}/old2" "${O:?}/old_nb" "${O:?}/n1_nb" "${O:?}/n2_nb"
run old2 "$OLD" "${BASE[@]}" "${BEATS[@]}"
run old_nb "$OLD" "${BASE[@]}"
run n1_nb "$NEW" "${BASE[@]}" --gpus 0
run n2_nb "$NEW" "${BASE[@]}" --gpus 0,0
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
        print(f"{x} vs {y} {n}: max |diff| {d.max() / r:.2e} of rms, rms diff {np.sqrt(np.mean(d.astype(np.float64)**2)) / r:.2e}, "
              f"differing {100 * np.mean(u > 0):.2f} %, > 1 ulp {100 * np.mean(u > 1):.3f} %, max ulp {u.max()}")
cmp("n1", "n2r")
cmp("old", "old2")
cmp("old", "n1")
cmp("old_nb", "n1_nb")
cmp("old_nb", "n2_nb")
cmp("n1_nb", "n2_nb")
EOF
echo test2 done
