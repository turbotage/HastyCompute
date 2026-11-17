"""Full-enumeration verification of the K = 7 lattice designs from search2.log (exact-margin search), at slack
f = 1 / 0.92 / 0.85, against coprime p = 2 (2026-11-17). Prints the same columns as enc_verify.py.

    nice -n 10 rustlib/.venv/bin/python research/experiments/encoding/enc_verify7.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from enc_lib import designs, max_m1  # noqa: E402
from enc_verify import run, scale_of  # noqa: E402

PICKS = {
    "K7 lattice @7.96": [[0, 0, 0], [2, -2, 2], [0, 1, 1], [-2, -2, 0], [1, -1, 2], [3, -2, -3], [-2, 1, 1]],
    "K7 lattice @8.28": [[0, 0, 0], [1, 2, 1], [1, -2, -1], [1, -1, 2], [2, 0, -2], [3, -3, 2], [-2, -2, 2]],
    "K7 lattice @9.84": [[0, 0, 0], [0, 0, 2], [1, -2, 1], [-3, -2, -3], [-1, 0, 3], [-2, 3, 1], [3, 1, -2]],
    "K7 lattice @7.02": [[0, 0, 0], [0, -1, -1], [0, 2, -2], [0, 1, 0], [2, 1, 1], [2, 0, -1], [3, -1, -1]],
}
if __name__ == "__main__":
    ref_m1 = max_m1(math.pi * designs()["coprime7 p=2"])
    for name, z in PICKS.items():
        z = np.array(z, float)
        run(name, z, scale_of(z), (1.0, 0.92, 0.85), ref_m1)
