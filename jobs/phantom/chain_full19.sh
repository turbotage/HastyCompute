#!/usr/bin/env bash
# full19 (2026-11-28): vessel tree realism + motion realism, then the refined heart again.
#   graph export_full19: the hepatic portal tree is rooted in the liver (no bridge / relocation into the IVC: it was a
#   portosystemic shunt, a hairpin in the viewer), the venae cavae and pulmonary veins end where they enter their atrium
#   (no outward connectors, IVC stub 15 mm shorter); network: portal sink at RA + 4 mmHg, hepatic veins carry 25 % CO.
#   motion: tether (bone + spine-bound great vessels) + great-artery root hold (kept fraction of the AV descent 0.77 at
#   the annulus → 0.45 ostia → 0.25 mid-ascending → 0 at 100 mm).
# Steps: test phantom seed 1 (clearance, carve) into data/hasty/phantom/factory_test19 → audit, flow check, vessel
# kinematics check → 0.5 mm LBM case with all four valves refined (0.25 mm, 25 mm downstream) → 6 cycles → report,
# compared with lbm_full18_05ra (same refinement on the full18 phantom).
# Pre-registered question: with the new tree and motion, are the valve metrics within the full18 refined run's
# run-to-run spread (lbm_full18_05r run_late vs run)? Decision: yes → full19 replaces full18 as the test phantom;
# a shift beyond the spread → trace it (root motion moves the aortic valve plane 4 mm less) before replacing.
set -uo pipefail
cd /home/turbotage/Documents/GitHub/HastyCompute
PY="systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 rustlib/.venv/bin/python -I"
T=rustlib/tools/phantom
X=data/hasty/phantom/factory_test/export_full19
OUT=data/hasty/phantom/factory_test19
TP=$OUT/ph0001
S=data/hasty/scratch
L=jobs/logs
LOCK=jobs/locks/gpu.lock
rm -rf "${TP:?}"
flock "$LOCK" $PY $T/make_phantom.py 1 --out $OUT --graph "$X" --truth "" --clearance --allow-overlap > $L/chain_full19_make.log 2>&1 || { echo "make_phantom failed"; tail -8 $L/chain_full19_make.log; exit 1; }
grep -E "stroke volume|clearance|cardiac motion|done" $L/chain_full19_make.log | cut -c1-300
flock "$LOCK" $PY $T/clearance.py $TP --audit-only > $TP/clearance_audit.log 2>&1 || true
grep -E "^audit|^   pairs|^   within|^     [0-9]" $TP/clearance_audit.log | head -20
$PY $T/flow_check.py $TP | tail -12 || true
grep -E "tether|root hold" $TP/motion/*.log 2>/dev/null | head -4
rustlib/.venv/bin/python -I jobs/tools/vessel_motion_check.py $TP $TP/motion/figs_vessels 2>&1 | tail -12 | cut -c1-330
rustlib/.venv/bin/python -I jobs/tools/aorta_root_profile.py $TP --dav-mm "$(rustlib/.venv/bin/python -I -c "import json; print(json.load(open('$TP/params.json'))['heart']['dav_mm'])")" 2>&1 | grep -v "aorta path"
rm -rf "${TP:?}/_tmp_check"
rustlib/.venv/bin/python -I jobs/tools/resp_det.py $TP/motion/resp_motion.hv $TP/atlas/tissue_2mm.hv | head -3
R="$PY jobs/tools/vessel_render.py"
$R $TP $S/vc_geom/ivc_full19_phantom.png --center -10,-20,1175 --half 55,40,95 --vox 0.6 --rmin-mm 1.5 --views ant,right,post,ant-right --title "factory_test19/ph0001 (after clearance): IVC region, r ≥ 1.5 mm" | tail -1 | cut -c1-60
$R $TP $S/vc_geom/ivc_full19_motion.png --center -10,-20,1175 --half 55,40,95 --vox 0.8 --rmin-mm 3 --views ant,right --motion $TP --states 0:0,0:1,0.3:0 --title "factory_test19/ph0001: IVC region at end-expiration, end-inspiration, systole (r ≥ 3 mm)" | tail -1 | cut -c1-60
O=$S/lbm_full19_05ra
mkdir -p $O
$PY $T/lbm_case.py $TP $O/case --dx-mm 0.5 --refine all --refine-down-mm 25 > $O/case.log 2>&1 || { echo "lbm_case failed"; tail -5 $O/case.log; exit 1; }
tail -3 $O/case.log | cut -c1-300
flock "$LOCK" systemd-run --user --scope -q -p MemoryMax=32G -p MemorySwapMax=0 rustlib/target/release/cardiac_lbm $O/case $O/run --cycles 6 --warmup 3 --quiet > $L/chain_full19_run.log 2>&1 || { echo "cardiac_lbm failed"; tail -5 $L/chain_full19_run.log; exit 1; }
tail -2 $L/chain_full19_run.log
$PY $T/lbm_report.py $O/case $O/run > $L/chain_full19_report.log 2>&1 || { echo "lbm_report failed"; tail -5 $L/chain_full19_report.log; exit 1; }
grep -E "^(valve gate|density gate|tree balance)" $L/chain_full19_report.log | cut -c1-300
echo chain full19 done
