#!/bin/bash
# Detached remainder of the 2-D basis test (runs as a systemd user service, independent of any editor session):
# wait for a running splinerecon, then the missing arms, the gradient-pass timings, the scoring.
cd /home/turbotage/Documents/GitHub/HastyCompute
O=data/hasty/phantom/factory_arms/ph0001/spline2d
while pgrep -x splinerecon >/dev/null; do sleep 30; done
research/experiments/spline2d/run_arms.sh
research/experiments/spline2d/time_bases.sh > $O/timing.log 2>&1
nice -n 5 rustlib/.venv/bin/python -I research/experiments/spline2d/score.py > $O/score.log 2>&1
echo "pipeline done $(date)" >> $O/pipeline.log
