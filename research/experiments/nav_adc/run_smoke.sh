#!/bin/bash
# Navigator-ADC smoke test (research/sequence_design.md §10): simphantom on the dev phantom input, 2 s of the
# flow_thorax scan with the navigator on / off (ideal chain), plus 0.5 s with --girf resonance --girf-hist 3;
# then check_nav.py (files, counts, times, k vs design moments, readout data unchanged). Verdict:
#   data/hasty/phantom/factory_arms/ph0001/nav_smoke/verdict.txt and research/experiments/nav_adc/verdict.txt
set -u
R=/home/turbotage/Documents/GitHub/HastyCompute
cd $R
E=research/experiments/nav_adc
LOCK=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/gpu.lock
B=rustlib/target/seqnav/release/simphantom
IN=data/hasty/phantom/factory_arms/ph0001/sim_flow_dev_respfield/sim_input.hv
O=data/hasty/phantom/factory_arms/ph0001/nav_smoke
mkdir -p $O
V=$O/verdict.txt
echo "nav_smoke queued $(date)" > $V
cat > $O/gpu_part.sh <<EOS
#!/bin/bash
cd $R
echo "GPU start \$(date)"
$B $IN $O/nav_on --seq flow --flow-run data/hasty/seq/flow_thorax --scan-time 2 --no-maps > $O/nav_on.log 2>&1; echo "nav_on exit \$?"
$B $IN $O/nav_off --seq flow --flow-run data/hasty/seq/flow_thorax --scan-time 2 --no-maps --nav off > $O/nav_off.log 2>&1; echo "nav_off exit \$?"
$B $IN $O/nav_girf --seq flow --flow-run data/hasty/seq/flow_thorax --scan-time 0.5 --no-maps --girf resonance --girf-hist 3 > $O/nav_girf.log 2>&1; echo "nav_girf exit \$?"
echo "GPU end \$(date)"
EOS
chmod +x $O/gpu_part.sh
flock $LOCK systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 timeout 1800 $O/gpu_part.sh >> $V 2>&1
echo "--- checks $(date)" >> $V
python3 -I $E/check_nav.py >> $V 2>&1 && echo "VERDICT: PASS" >> $V || echo "VERDICT: FAIL (see above)" >> $V
for f in nav_on nav_off nav_girf; do echo "--- $f.log tail" >> $V; tail -6 $O/$f.log >> $V 2>/dev/null; done
cp $V $E/verdict.txt
