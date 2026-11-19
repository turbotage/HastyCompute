"""eps-rank of one readout's range (SVD of a point-source sketch): the lower bound on the number of virtual samples
any compression of that readout (orthogonal or interpolative) needs at a given relative error. Uniform sources =
white image (worst on average); signal = ratemap components drawn ~ |amplitude|.

run: rustlib/.venv/bin/python -I research/experiments/sample_compress/eps_rank.py case readout_index(0..5 of id_readouts' draw)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_common as sc  # noqa: E402
from id_readouts import CASES  # noqa: E402

OUT = os.path.join(HERE, "out")


def main():
    name = sys.argv[1]
    i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    mode = sys.argv[3] if len(sys.argv) > 3 else "uni"
    case = [c for c in CASES if c[0] == name][0]
    _, scale, roi, half = case[:4]
    rng = np.random.default_rng(7)
    ros = np.sort(rng.choice(np.arange(0, sc.META["n_readouts"], 7), 6, replace=False))
    rd = sc.load_readouts([ros[i]])[0]
    if len(case) > 4:
        rd = sc.upsample(rd, case[4])
    wb = sc.offres_band(half, sc.BOX_CENTRE)
    B, dwell = sc.band(rd, roi, half, scale, wb)
    landau = float((B * dwell / (2 * np.pi)).sum())
    t0 = time.time()
    P = int(min(1.5 * landau + 600, 5000))
    Vs = sc.columns(rd, np.random.default_rng(500 + i), P, roi, half, sc.BOX_CENTRE, scale, wb, mode=mode)
    sv = np.linalg.svd(Vs, compute_uv=False)
    tail = np.sqrt(np.cumsum((sv ** 2)[::-1])[::-1] / (sv ** 2).sum())
    er = {str(e): int(np.argmax(tail < e)) if (tail < e).any() else None for e in (1e-3, 1e-4, 1e-5)}
    path = os.path.join(OUT, "eps_rank.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res[f"{name}/{int(ros[i])}/{mode}"] = {"n": len(rd[1]), "landau": landau, "columns": P, "eps_rank": er,
                                           "reduction_bound_1e-4": len(rd[1]) / er["0.0001"] if er["0.0001"] else None,
                                           "seconds": time.time() - t0}
    json.dump(res, open(path, "w"), indent=1)
    print(name, ros[i], mode, res[f"{name}/{int(ros[i])}/{mode}"], flush=True)


if __name__ == "__main__":
    main()
