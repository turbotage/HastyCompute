#!/usr/bin/env bash
# dsum register blocking: SIMPH_DSUM_RB = 1 (dsum, reference), 2, 4 (dsum_rb) on the deterministic dev config (no beat
# library: no atomic source packing), 1 s of scan, navigator on. Same arithmetic per pair and source order, so the
# outputs must agree to ≈ 1 ulp (f64 atomic order only); the simulation seconds give the speed-up.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
P=data/hasty/phantom/factory_arms/ph0001
SF=$P/sim_flow_dev_respfield
O=data/hasty/scratch/dsum_rb
L=jobs/logs
LOCK=jobs/locks/gpu.lock
export HASTY_IO=npy
ARGS=(--seq flow --flow-run data/hasty/seq/flow_thorax --flip 12 --scan-time 1.0 --resp-field "$SF/resp_field.hv" --gpus 0)
rm -rf "${O:?}"
mkdir -p "$O"
for rb in 1 2 4; do
  SIMPH_DSUM_RB=$rb SIMPH_TIMING=1 flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 \
    rustlib/target/release/simphantom "$SF/sim_input.hv" "$O/rb$rb" "${ARGS[@]}" > "$L/dsum_rb$rb.log" 2>&1
  echo "rb $rb: $(grep -E '^simulation ' "$L/dsum_rb$rb.log") $(grep -E 'panicked|rror' "$L/dsum_rb$rb.log" | head -2)"
  grep -E "^    bin 100:" "$L/dsum_rb$rb.log" | tail -1
done
rustlib/.venv/bin/python -I - "$O" <<'EOF'
import sys, os, numpy as np
O = sys.argv[1]
def ulps(a, b):
    ai = a.view(np.int32).astype(np.int64); bi = b.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, -2**31 - ai, ai); bi = np.where(bi < 0, -2**31 - bi, bi)
    return np.abs(ai - bi)
for t in ["rb2", "rb4"]:
    for n in ["data_clean.npy", "nav_encode.npy", "nav_tail.npy"]:
        fa, fb = os.path.join(O, "rb1", n), os.path.join(O, t, n)
        if not (os.path.exists(fa) and os.path.exists(fb)):
            print(f"{t} {n}: MISSING"); continue
        a = np.load(fa).view(np.float32).ravel(); b = np.load(fb).view(np.float32).ravel()
        r = np.sqrt(np.mean(a.astype(np.float64) ** 2)); u = ulps(a, b)
        print(f"rb1 vs {t} {n}: max |diff| {np.abs(a - b).max() / r:.2e} of rms, differing {100 * np.mean(u > 0):.3f} %, > 1 ulp {100 * np.mean(u > 1):.4f} %, max ulp {u.max()}")
EOF
echo rb test done
