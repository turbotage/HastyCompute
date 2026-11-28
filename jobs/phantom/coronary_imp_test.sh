#!/usr/bin/env bash
# Coronary intramyocardial pressure (network_flow --coronary-imp κ) on the full17 test phantom: κ = 0 must reproduce the
# stored network exactly (regression of the two-pass restructure); κ 0.5 / 0.75 / 1.0 → the LCA / RCA diastolic share
# (target LCA ≈ 0.70–0.75, RCA lower). Outputs in data/hasty/scratch/net_imp_*, logs in jobs/logs.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
PH=data/hasty/phantom/factory_test/ph0001
S=data/hasty/scratch
PY="systemd-run --user --scope -q -p MemoryMax=30G -p MemorySwapMax=0 rustlib/.venv/bin/python -I"
# wait until no other heavy CPU job (lbm_case / make_phantom) runs
while ps -eo cmd | grep -qE "[l]bm_case.py|[m]ake_phantom.py"; do sleep 20; done
ARGS=$(rustlib/.venv/bin/python -I -c "
import json; p=json.load(open('$PH/params.json')); h=p['heart']; c=p['circulation']
print(' '.join(str(x) for x in ['--hr',h['hr'],'--sv',h['sv_ml'],'--p-sys',c['p_sys'],'--p-pul',c['p_pul'],'--ctot-scale',c['ctot_scale'],'--terr-sigma',c['terr_sigma'],'--seed',p['seed']]))")
echo "network args: $ARGS"
for k in ${KAPPAS:-0 0.5 0.75 1.0}; do
  O=$S/net_imp_$k
  rm -rf "${O:?}"
  # shellcheck disable=SC2086
  $PY rustlib/tools/phantom/network_flow.py $PH $O $ARGS --coronary-imp $k > jobs/logs/net_imp_$k.log 2>&1 || { echo "κ $k failed"; tail -5 jobs/logs/net_imp_$k.log; continue; }
  echo "κ $k: $(grep -E '^Left coronary|^Right coronary|intramyocardial' jobs/logs/net_imp_$k.log | tr '\n' ' ')"
done
rustlib/.venv/bin/python -I - <<'EOF'
import numpy as np
a = np.load("data/hasty/phantom/factory_test/ph0001/flow/network_flow.npz")["q"]
b = np.load("data/hasty/scratch/net_imp_0/network_flow.npz")["q"]
print(f"regression κ 0 vs stored: shape {a.shape} {b.shape}, max |Δq| {np.abs(a - b).max():.3e} (max |q| {np.abs(a).max():.3e}), identical {np.array_equal(a, b)}")
EOF
echo coronary test done
