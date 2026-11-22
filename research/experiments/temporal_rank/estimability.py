"""Run 8: can the SVD basis be estimated from measurable data? (research/temporal_rank.md)

Target: the continuous-motion truth (truth velocity, continuous-motion magnitude) on 2 mm voxels. Bases (shared over
the 7 encodes) learned from degraded sources and applied to the target:
- oracle: the target's own Gram;
- lowres N mm: the same scene on N mm voxels (complex mean of the 1 mm signal over N^3), as from a low-resolution
  frame-resolved reconstruction of the densely sampled k-space centre;
- noisy SNR s: the target plus white complex noise of std = (mean lumen magnitude) / s per voxel and frame;
- other regime: the bases of other runs (network-flow overlay; per-vessel delays 50 ms) applied to this target;
- generic: periodic cubic B-splines and Fourier at equal K (already in the other runs, repeated here).

    rustlib/.venv/bin/python estimability.py
"""
import json
import pickle
import time

import numpy as np

from common import CACHE, OUT, cardiac_basis, encode_matrix, fourier_basis, orth
from experiment import Context, eig_desc, grams
from rank_lib import Fields, Vessels, build_X, eval_basis
from regimes import Overlay
from experiment import truth_refs

KS = [8, 12, 16, 20, 24, 32]


def boxn(a, n):
    z, y, x = (s // n * n for s in a.shape)
    a = a[:z, :y, :x]
    return a.reshape(z // n, n, y // n, n, x // n, n).mean(axis=(1, 3, 5))


def lowres_gram(F, sm, n, vel=None):
    K = encode_matrix()
    cols = []
    for f in range(F.nf):
        v = F.v[f].astype(np.float32) if vel is None else vel(f)
        m = sm[f].astype(np.float32)
        row = []
        for e in range(7):
            ph = K[e, 0] * v[0] + K[e, 1] * v[1] + K[e, 2] * v[2]
            row.append(boxn(m * np.exp(-1j * ph), n).reshape(-1))
        cols.append(np.stack(row))
    X = np.stack(cols, -1)  # [7, V_n, T]
    return grams(X).sum(0)


def main():
    t0 = time.time()
    F = Fields()
    b2, cat, vmap = pickle.loads((CACHE / "context.pkl").read_bytes())
    sm = np.load(CACHE / "smooth_mag.npy")
    mag = lambda f: sm[f].astype(np.float32)  # noqa: E731
    X = build_X(F, mag=mag)
    ctx = Context(X, b2, vmap)
    vt, Qt = truth_refs(X, ctx)
    print(f"target built {time.time() - t0:.0f} s", flush=True)
    sources = {}
    w, U = eig_desc(grams(X).sum(0))
    sources["oracle (target's own SVD)"] = U
    for n in [4, 6, 8]:
        w, U = eig_desc(lowres_gram(F, sm, n))
        sources[f"low resolution {n} mm"] = U
        print(f"lowres {n} {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(5)
    mlum = np.abs(X[0][ctx.bl]).mean()
    for snr in [20, 10, 5]:
        sig = mlum / snr
        G = np.zeros((X.shape[2], X.shape[2]), np.complex128)
        for e in range(7):
            Xn = X[e] + (sig / np.sqrt(2) * (rng.standard_normal(X[e].shape) + 1j * rng.standard_normal(X[e].shape))).astype(np.complex64)
            G += Xn.conj().T @ Xn
        w, U = eig_desc(G)
        sources[f"noisy, lumen SNR {snr} per frame"] = U
        print(f"noisy {snr} {time.time() - t0:.0f} s", flush=True)
    # other regimes: the network-flow overlay with and without 50 ms delays (velocity of the vessel blood replaced)
    Ve = Vessels()
    ov = Overlay(F, Ve)
    for name, kn in [("other regime: network-flow overlay", {}), ("other regime: overlay + 50 ms delays + coronary 30 ms", {"tau_ms": 50, "coronary": {"ramp_ms": 30, "peak_ratio": 2.5}})]:
        Xo = build_X(F, vel=ov.velocity(ov.waveforms(kn)), mag=mag)
        w, U = eig_desc(grams(Xo).sum(0))
        sources[name] = U
        del Xo
        print(f"{name} {time.time() - t0:.0f} s", flush=True)
    # the staircase-magnitude truth (the phantom's own frames)
    Xs = build_X(F)
    w, U = eig_desc(grams(Xs).sum(0))
    sources["staircase-magnitude truth"] = U
    del Xs
    phase = np.arange(F.nf) / F.nf
    rows = []
    for k in KS:
        for name, U in sources.items():
            r = eval_basis(X, [U[:, :k]] * 7, ctx.sig, ctx.bl, ctx.b2, ctx.vmap, vt, Qt, ctx.idxB, ctx.catB)
            r.update(source=name, K=k)
            rows.append(r)
        for name, B in [("generic: cubic B-splines", cardiac_basis(phase, k, 3)), ("generic: Fourier", fourier_basis(phase, k))]:
            r = eval_basis(X, [orth(B).astype(np.complex128)] * 7, ctx.sig, ctx.bl, ctx.b2, ctx.vmap, vt, Qt, ctx.idxB, ctx.catB)
            r.update(source=name, K=k)
            rows.append(r)
        print(f"K {k} {time.time() - t0:.0f} s", flush=True)
        for r in rows[-len(sources) - 2:]:
            q = r.get("Q", {})
            print(f"  {r['source']:52s} rel_blood {r['rel_blood']:.4f} dv_mean {r['dv_mean']:.2f} dv_p99 {r['dv_p99']:.1f} RCA {q.get('RCA', (np.nan,))[0]:.3f} LAD {q.get('LAD', (np.nan,))[0]:.3f} LCx {q.get('LCx', (np.nan,))[0]:.3f}", flush=True)
    (OUT / "estimability.json").write_text(json.dumps({"rows": rows}, indent=1))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
