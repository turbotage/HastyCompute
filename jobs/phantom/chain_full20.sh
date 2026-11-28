#!/usr/bin/env bash
# full20 (2026-11-28): full19 + vessel realism round 2, then the refined heart again.
#   graph export_full20: radius taper where one vessel continues another (radius_taper.py: degree-2 joints and the
#   parent's through-line at branch nodes; steps ≥ 15 % at r ≥ 2 mm: 5 → 0 in the graph), the same taper again after the
#   per-name radius jitter in make_phantom (the IVC stepped 8.2 | 10.1 mm in factory_test19).
#   motion: breathing vessel rod (systemic + portal veins r ≥ 4 mm follow the organs they touch, smoothed along the
#   vessel over ℓ 30 mm: IVC 3.8 → 6.1 renal → 8.6 coeliac → 10–11.6 intrahepatic, monotone; was a 0.3 mm still spot)
#   and the below-FOV breathing tether per edge, great arteries only (median r ≥ 4 mm; the SMA moves with the mesentery).
# Pre-registered question: are full20's valve metrics (lbm_full20_05ra) within the run-to-run spread (lbm_full18_05r
# run vs run_late) of full19's (lbm_full19_05ra)? Decision: yes → full20 is the test phantom; no → trace it (the taper
# changes the ascending aorta | arch joint near the aortic valve) before replacing.
# Output: jobs/logs/chain_full20.log (the scoped steps' output does not reach the unit's journal).
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
exec >> jobs/logs/chain_full20.log 2>&1
echo "chain full20 start $(date -Is)"
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
E=data/hasty/external/zanatomy/export
X19=data/hasty/phantom/factory_test/export_full19
X0=data/hasty/scratch/vc_geom/export_taper0
X=data/hasty/phantom/factory_test/export_full20
OUT=data/hasty/phantom/factory_test20
TP=$OUT/ph0001
S=data/hasty/scratch
L=jobs/logs
LOCK=jobs/locks/gpu.lock
mkdir -p $X0 $X
$PY $T/vessel_graph.py $E --out $X0 --joint-taper-mm 0 > $X0/vessel_graph.log 2>&1 || { echo "taper0 failed"; tail -5 $X0/vessel_graph.log; exit 1; }
rustlib/.venv/bin/python -I - <<EOF
import numpy as np
a = np.load("$X19/graph.npz"); b = np.load("$X0/graph.npz")
d = {k: (float(np.abs(a["radius_atlas" if k == "radius" else k].astype(np.float64) - b[k].astype(np.float64)).max()) if a[k].shape == b[k].shape else "shape") for k in b.files}
print("self-check taper 0 vs export_full19 (max |Δ| per array; radius vs full19's pre-Murray radius_atlas):", d)
assert all(v == 0.0 for v in d.values()), "taper 0 does not reproduce export_full19"
EOF
[ $? -eq 0 ] || { echo "self-check failed"; exit 1; }
$PY $T/vessel_graph.py $E --out $X > $X/vessel_graph.log 2>&1 || { echo "full20 graph failed"; tail -5 $X/vessel_graph.log; exit 1; }
grep -E "^joint taper|^de-overlap" $X/vessel_graph.log | cut -c1-600
$PY $T/network_flow.py "$X" "$X/flow" > /dev/null || { echo network failed; exit 1; }
for _ in 1 2; do $PY $T/murray_floor.py "$X" "$X/flow" > /dev/null; $PY $T/network_flow.py "$X" "$X/flow" > $X/network.log 2>&1; done
grep -E "residual|Kirchhoff" $X/network.log | head -3 | cut -c1-300
rm -rf "${X0:?}"
rm -rf "${TP:?}"
flock "$LOCK" $PY $T/make_phantom.py 1 --out $OUT --graph "$X" --truth "" --clearance --allow-overlap > $L/chain_full20_make.log 2>&1 || { echo "make_phantom failed"; tail -8 $L/chain_full20_make.log; exit 1; }
grep -E "stroke volume|clearance|cardiac motion|done" $L/chain_full20_make.log | cut -c1-300
flock "$LOCK" $PY $T/clearance.py $TP --audit-only > $TP/clearance_audit.log 2>&1 || true
grep -E "^audit|^   pairs|^   within|^     [0-9]" $TP/clearance_audit.log | head -20
$PY $T/flow_check.py $TP | tail -12 || true
grep -E "tether|root hold" $TP/motion/*.log 2>/dev/null | head -4
rustlib/.venv/bin/python -I jobs/tools/vessel_motion_check.py $TP $TP/motion/figs_vessels 2>&1 | tail -12 | cut -c1-330
rustlib/.venv/bin/python -I jobs/tools/aorta_root_profile.py $TP --dav-mm "$(rustlib/.venv/bin/python -I -c "import json; print(json.load(open('$TP/params.json'))['heart']['dav_mm'])")" 2>&1 | grep -v "aorta path"
rm -rf "${TP:?}/_tmp_check"
rustlib/.venv/bin/python -I jobs/tools/resp_det.py $TP/motion/resp_motion.hv $TP/atlas/tissue_2mm.hv | head -3
rustlib/.venv/bin/python -I jobs/tools/vessel_resp_neighbours.py $TP $TP/atlas/tissue_2mm.hv "^Inferior vena cava" --step-mm 10 | cut -c1-150
rustlib/.venv/bin/python -I -c "import json; j = json.load(open('$TP/motion/resp_motion.json')); print('resp: tether', j.get('tether'), '| vessel rod', j.get('vessel_rod'))" | cut -c1-500
R="$PY jobs/tools/vessel_render.py"
$R $TP $S/vc_geom/ivc_full20_phantom.png --center -10,-20,1175 --half 55,40,95 --vox 0.6 --rmin-mm 1.5 --views ant,right,post,ant-right --title "factory_test20/ph0001 (after clearance): IVC region, r ≥ 1.5 mm" | tail -1 | cut -c1-60
$R $TP $S/vc_geom/ivc_full20_motion.png --center -10,-20,1175 --half 55,40,95 --vox 0.8 --rmin-mm 3 --views ant,right --motion $TP --states 0:0,0:1,0.3:0 --title "factory_test20/ph0001: IVC region at end-expiration, end-inspiration, systole (r ≥ 3 mm)" | tail -1 | cut -c1-60
O=$S/lbm_full20_05ra
mkdir -p $O
$PY $T/lbm_case.py $TP $O/case --dx-mm 0.5 --refine all --refine-down-mm 25 > $O/case.log 2>&1 || { echo "lbm_case failed"; tail -5 $O/case.log; exit 1; }
tail -3 $O/case.log | cut -c1-300
flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0 rustlib/target/release/cardiac_lbm $O/case $O/run --cycles 6 --warmup 3 --quiet > $L/chain_full20_run.log 2>&1 || { echo "cardiac_lbm failed"; tail -5 $L/chain_full20_run.log; exit 1; }
tail -2 $L/chain_full20_run.log
$PY $T/lbm_report.py $O/case $O/run > $L/chain_full20_report.log 2>&1 || { echo "lbm_report failed"; tail -5 $L/chain_full20_report.log; exit 1; }
grep -E "^(valve gate|density gate|tree balance)" $L/chain_full20_report.log | cut -c1-300
echo chain full20 done
