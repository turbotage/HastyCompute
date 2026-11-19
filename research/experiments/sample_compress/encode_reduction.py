"""Whole-encode reduction of the per-readout time decimation (selection rule of sc_common.select_warped, all encode-0
readouts) and the k-space density left after it (direction 2: what a dense-core treatment could still address).

Only the subset sizes are computed here (no interpolator): the accuracy of each (os, q) is measured per readout in
id_readouts.py / dense_check.py.

run: rustlib/.venv/bin/python -I research/experiments/sample_compress/encode_reduction.py [ro_stride]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_common as sc  # noqa: E402

OUT = os.path.join(HERE, "out")
CASES = [("box160_2mm_cube", 1.0, "cube", 0.08, 1), ("box160_2mm_ball", 1.0, "ball", 0.08, 1),
         ("box160_1mm_cube", 2.0, "cube", 0.08, 1), ("box160_1mm_ball", 2.0, "ball", 0.08, 1),
         ("box128_1mm_cube", 2.0, "cube", 0.064, 1),
         ("box160_1mm_cube_dwell1us", 2.0, "cube", 0.08, 2), ("box160_1mm_ball_dwell1us", 2.0, "ball", 0.08, 2)]
OSS = [1.25, 1.35, 1.5, 2.0]
SHELLS = [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    t0 = time.time()
    s, ro_idx = sc.common.encode_samples(0, ro_stride=stride)       # k (rad/m), tau
    nro, na, _ = s.shape
    k = s[..., :3].astype(np.float64)
    tau = s[..., 3].astype(np.float64)
    del s
    res = {"readouts": int(nro), "stride": stride, "cases": {}}
    for name, scale, roi, half, up in CASES:
        wb = sc.offres_band(half, sc.BOX_CENTRE)
        N = int(round(2 * half / (0.001 if scale == 2.0 else 0.002)))     # box voxels per axis
        tot_n, tot_S = 0, {o: 0 for o in OSS}
        shell_n = np.zeros(len(SHELLS))
        shell_S = np.zeros(len(SHELLS))
        cells = {i: [] for i in range(len(SHELLS))}
        for r in range(nro):
            kr, tr = k[r], tau[r]
            if up > 1:
                f = np.arange((na - 1) * up + 1) / up
                kr = np.stack([np.interp(f, np.arange(na), kr[:, a]) for a in range(3)], 1)
                tr = np.interp(f, np.arange(na), tr)
            B, dwell = sc.band((kr, tr, None, None), roi, half, scale, wb)
            n = len(tr)
            tot_n += n
            for o in OSS:
                S = sc.select_warped(B, dwell, o)
                tot_S[o] += len(S)
                if o == 1.35:
                    u = kr[S] / (2 * np.pi) * scale * 2 * (2 * half)                # 2N cells of the box grid
                    rr = np.linalg.norm(u, axis=1) / N
                    rall = np.linalg.norm(kr, axis=1) / (2 * np.pi) * scale * 2 * (2 * half) / N
                    cid = np.floor(u + N).astype(np.int64)
                    cid = (cid[:, 0] * 2 * N + cid[:, 1]) * 2 * N + cid[:, 2]
                    for i, (a, b) in enumerate(SHELLS):
                        m = (rr >= a) & (rr < b)
                        shell_S[i] += m.sum()
                        shell_n[i] += ((rall >= a) & (rall < b)).sum()
                        if r % 4 == 0:
                            cells[i].append(np.stack([cid[m], np.full(m.sum(), r)], 1))
        dens = []
        for i, (a, b) in enumerate(SHELLS):
            c = np.concatenate(cells[i]) if cells[i] else np.zeros((0, 2), np.int64)
            ncell = len(np.unique(c[:, 0])) if len(c) else 1
            npair = len(np.unique(c[:, 0] * nro + c[:, 1])) if len(c) else 1
            dens.append({"shell": f"{a}-{b}", "frac_samples_before": float(shell_n[i] / tot_n), "frac_samples_after": float(shell_S[i] / sum(shell_S)),
                         "reduction_in_shell": float(shell_n[i] / max(1, shell_S[i])),
                         "after_samples_per_readout_visit": float(len(c) / max(1, npair)),
                         "after_readouts_per_occupied_cell_quarter_readouts": float(npair / max(1, ncell))})
        res["cases"][name] = {"reduction": {str(o): tot_n / tot_S[o] for o in OSS}, "shells_os1.35": dens}
        print(name, {o: round(tot_n / tot_S[o], 3) for o in OSS}, f"({time.time()-t0:.0f}s)", flush=True)
        for d in dens:
            print("   ", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in d.items()}, flush=True)
        json.dump(res, open(os.path.join(OUT, f"encode_reduction_s{stride}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
