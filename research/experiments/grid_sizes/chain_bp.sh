#!/bin/bash
# after the baseline (gs-a): the tuned cases (b), the nsys profiles (p), the planner demonstration (c)
while systemctl --user is-active --quiet gs-a; do sleep 30; done
BIN=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/bin/opbench_gs2 TAG=b CASES_FILE=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes/cases_b.txt /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes/run_cases.sh
BIN=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/bin/opbench_gs2 TAG=p CASES_FILE=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes/cases_p.txt /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes/run_prof.sh
BIN=/tmp/claude-1000/-home-turbotage-Documents-GitHub-HastyCompute/c25caa1d-d864-44cf-9019-66f29ac9cb24/scratchpad/bin/opbench_gs3 TAG=c CASES_FILE=/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes/cases_c.txt /home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes/run_cases.sh
/home/turbotage/Documents/GitHub/HastyCompute/research/experiments/grid_sizes/run_reg.sh
