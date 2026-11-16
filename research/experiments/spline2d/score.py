"""Score the respiratory × cardiac basis arms (run_arms.sh) with frame_fit.py at each breathing state's own truth.

    rustlib/.venv/bin/python -I research/experiments/spline2d/score.py [arm ...]

Per arm, state r* and saved iteration: frame_fit lines pooled depth 0-2 (vessel edges), dynamics depth >= 2 / >= 5 and
ROI dynamics depth >= 5 (slope / corr / wrapped error, truth dyn rms). Writes scores.csv next to the outputs.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
O = REPO / "data/hasty/phantom/factory_arms/ph0001/spline2d"
S = REPO / "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan"
FF = REPO / "research/experiments/recon_diag/frame_fit.py"
PY = REPO / "rustlib/.venv/bin/python"
TRUTH = {"0.07": "truth_pc_frames_r0.07.npz", "0.181": "truth_pc_frames_r0.181.npz", "0.493": "truth_pc_frames.npz"}
arms = sys.argv[1:] or ["bins8", "lin6", "lin4"]
pat = {
    "pooled_0_2": r"depth 0-2: slope ([-\d.]+) corr ([-\d.]+) wrapped phase error rms ([\d.]+)",
    "dyn_2": r"dynamics, depth 2-99 \(\d+ voxels\): slope ([-\d.]+) corr ([-\d.]+) wrapped error rms ([\d.]+) rad \(truth dyn rms ([\d.]+)\)",
    "dyn_5": r"dynamics, depth 5-99 \(\d+ voxels\): slope ([-\d.]+) corr ([-\d.]+) wrapped error rms ([\d.]+) rad \(truth dyn rms ([\d.]+)\)",
    "roi_5": r"ROI dynamics, depth >= 5 \(\d+ blocks of 8 mm\): slope ([-\d.]+) corr ([-\d.]+) wrapped error rms ([\d.]+) rad \(truth dyn rms ([\d.]+)\)",
}
rows = []
for arm in arms:
    for r, tf in TRUTH.items():
        for it in (5, 10, 20):
            f = O / f"{arm}_r{r}_it{it}_lumen.npy"
            if not f.exists():
                continue
            out = subprocess.run([str(PY), "-I", str(FF), str(S), str(f)], env={**os.environ, "FRAME_FIT_TRUTH": tf}, capture_output=True, text=True).stdout
            row = {"arm": arm, "r": r, "it": it}
            for k, p in pat.items():
                m = re.search(p, out)
                row[k] = "/".join(m.groups()[:3]) if m else "?"
                if m and len(m.groups()) > 3:
                    row["truth_dyn"] = m.group(4)
            rows.append(row)
            print(row, flush=True)
keys = ["arm", "r", "it", "pooled_0_2", "dyn_2", "dyn_5", "roi_5", "truth_dyn"]
with open(O / "scores.csv", "w") as fh:
    fh.write(",".join(keys) + "\n")
    for row in rows:
        fh.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
print(f"wrote {O / 'scores.csv'}")
