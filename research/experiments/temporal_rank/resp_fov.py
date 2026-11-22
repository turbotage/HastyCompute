"""Respiratory rank over the whole field of view (research/temporal_rank.md, run 4).

    rustlib/.venv/bin/python resp_fov.py [--amp A] [--R 24]

The k-space simulator's static-tissue sources (sim_flow_10min/sim_input.hv tissue: 11.9 M sources on a 1 mm lattice,
PD / T1 / T2*) are moved by the respiratory displacement u(X, r) = A r sum_k B_k(X) and splatted (trilinear) onto the
reconstruction grid (254 x 144 x 114, 2 mm) at R breathing states (equal-probability quantile midpoints of the scan's
breathing trace). Cardiac motion and blood are left out: this isolates the breathing axis of the whole image (chest
wall, liver dome, lung bases). Signal per source: PD sin(a)(1 - E1)/(1 - cos(a) E1) exp(-TE/T2*); the encodes only
differ by the respiratory tissue velocity, which this run leaves out (rdot = 0; see joint_rank --rdot), so one real
image per state.
"""
import json
import sys
import time

import numpy as np

from common import LEVELS, OUT, PH, TE, k_at, orth, resp_hats, ss
from experiment import eig_desc
from joint_rank import resp_field, sample_B

A = sys.argv[1:]
AMP = float(A[A.index("--amp") + 1]) if "--amp" in A else 1.0
R = int(A[A.index("--R") + 1]) if "--R" in A else 24


def main():
    t0 = time.time()
    from hastycompute import value_format as vf
    SI = vf.open_lazy(PH / "sim_flow_10min/sim_input.hv")
    meta = json.loads((PH / "sim_flow_10min/scan/meta.json").read_text())
    pos = np.asarray(SI["tissue"]["pos"].read())
    sig = (np.asarray(SI["tissue"]["pd"].read()) * ss(np.asarray(SI["tissue"]["t1"].read())) * np.exp(-TE / np.maximum(np.asarray(SI["tissue"]["t2s"].read()), 1e-4))).astype(np.float32)
    r_tr = np.asarray(SI["breathing"]["r"].read())[:600000]
    dims = np.array(meta["recon_dims"])  # x y z
    res = float(meta["res"][0])
    centre = np.array(meta["presc_center_subject"])
    lo = centre - (dims // 2) * res  # centre of voxel 0 (i -> centre + (i - n/2) res)
    B, borg, bvox = resp_field()
    U = sample_B(B, borg, bvox, pos)  # [N, 3]
    print(f"{len(pos)} sources, displacement field sampled {time.time() - t0:.0f} s", flush=True)
    qs = (np.arange(R) + 0.5) / R
    rs = np.quantile(r_tr, qs)
    nv = int(np.prod(dims))
    X = np.zeros((nv, R), np.float32)
    for j, r in enumerate(rs):
        p = (pos + AMP * r * U - lo) / res
        i0 = np.floor(p).astype(np.int64)
        fr = (p - i0).astype(np.float32)
        acc = np.zeros(nv, np.float64)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ix, iy, iz = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
                    w = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * (fr[:, 2] if dz else 1 - fr[:, 2])
                    ok = (ix >= 0) & (ix < dims[0]) & (iy >= 0) & (iy < dims[1]) & (iz >= 0) & (iz < dims[2])
                    flat = ix[ok] + dims[0] * (iy[ok] + dims[1] * iz[ok])
                    acc += np.bincount(flat, (w * sig)[ok], minlength=nv)
        X[:, j] = acc
        print(f"state {j} r={r:.3f} {time.time() - t0:.0f} s", flush=True)
    G = (X.astype(np.float64).T @ X.astype(np.float64))
    w, Uv = eig_desc(G)
    out = {"amp": AMP, "R": R, "r_states": rs.tolist(), "levels": LEVELS, "spectrum": (w / w[0]).tolist(), "K": k_at(w, LEVELS)}
    a = X.mean(1)
    sigm = a > 0.1 * np.percentile(a, 99)
    rows = []
    for k in range(1, min(R, 12) + 1):
        Phi = Uv[:, :k]
        E = X - (X @ Phi) @ Phi.T
        rows.append(dict(basis="svd", K=k, rel=float(np.linalg.norm(E[sigm]) / np.linalg.norm(X[sigm])), vox_p99=float(np.percentile(np.linalg.norm(E[sigm], axis=1) / np.maximum(np.linalg.norm(X[sigm], axis=1), 1e-12), 99))))
    for nk in [2, 3, 4, 6, 8, 12]:
        Phi = orth(resp_hats(rs, np.quantile(r_tr, np.linspace(0, 1, nk))))
        E = X - (X @ Phi) @ Phi.T
        rows.append(dict(basis="linear hats (quantile knots)", K=nk, rel=float(np.linalg.norm(E[sigm]) / np.linalg.norm(X[sigm])), vox_p99=float(np.percentile(np.linalg.norm(E[sigm], axis=1) / np.maximum(np.linalg.norm(X[sigm], axis=1), 1e-12), 99))))
    out["rows"] = rows
    (OUT / f"resp_fov_amp{AMP}.json").write_text(json.dumps(out, indent=1))
    print("K", out["K"])
    for r in rows:
        print(f"{r['basis']:30s} K={r['K']:2d} rel {r['rel']:.3g} vox_p99 {r['vox_p99']:.3g}")
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
