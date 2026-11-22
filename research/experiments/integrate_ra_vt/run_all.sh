#!/bin/bash
# chain: the rerun of vt_static_order2, the seg_* regression + sanitizer, then opbench
TESTS=vt_static_order2 /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/integrate_ra_vt/run_tests.sh
/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/integrate_ra_vt/run_regress.sh
/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/integrate_ra_vt/run_opbench.sh
touch /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/integrate_ra_vt/all.done
