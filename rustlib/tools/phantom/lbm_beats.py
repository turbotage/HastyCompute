"""Beat library of a cardiac LBM run (`cardiac_lbm --beats`): per cycle, phase bin and occupied output cell the coverage
w, mean velocity, covariance, third central moments, wall shear stress and (with `--spin`) the spin history: mean and std
of the relative M_z s and cov(s, u) (`<run>/beats/beat_<cycle>.bin`, f16; 23, 25 or 28 fields by version).

    rustlib/.venv/bin/python -I rustlib/tools/phantom/lbm_beats.py <run> [check] [--other <run2>]

check: the beats' sample-weighted mean against flow_bins (the averaged statistics of the same cycles), and with
--other a second run of the same case (e.g. stopped and resumed) beat by beat.

As a module:
- `BeatLibrary(run)`: `beats` (cycle numbers), `read(cycle)` → {bin: (cells, values [n, nf] f32, samples)},
  `dense(cycle, bin)` → coverage [Z, Y, X], mean [3, Z, Y, X], cov [6, Z, Y, X], m3 [10, Z, Y, X], wss [3, Z, Y, X]
  (+ mz, mz_std [Z, Y, X], c_su [3, Z, Y, X] with a spin history) on the run's output cell grid (`grid()`: origin of
  cell 0, spacing, dims x y z).
- `coarsen(w, mean, cov, m3, k)`: exact moments of the mixture over k³ cells (an MR voxel of k cells).
- `sigma_points(mean, cov, m3)`: 7 velocity samples per cell (κ = 3: the centre weight is 0, the six points of
  simphantom's `bsig`) with weights reproducing mean and covariance exactly and the diagonal third moments (skewness)
  along the covariance's principal axes — sub-voxel sources for the k-space simulator, so the voxel signal
  Σ w e^{i k·v} carries the intravoxel distribution and its asymmetry.
"""
import json
import sys
from pathlib import Path

import numpy as np

TRI = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2), (0, 1, 1), (1, 1, 2), (0, 2, 2), (1, 2, 2), (0, 1, 2)]
PAIR = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def cov_mat(c):
    """[6, …] (xx yy zz xy xz yz) → [3, 3, …]"""
    M = np.empty((3, 3) + c.shape[1:], c.dtype)
    for k, (i, j) in enumerate(PAIR):
        M[i, j] = c[k]
        M[j, i] = c[k]
    return M


def m3_tensor(m):
    """[10, …] → symmetric [3, 3, 3, …]"""
    T = np.empty((3, 3, 3) + m.shape[1:], m.dtype)
    import itertools

    for k, idx in enumerate(TRI):
        for p in set(itertools.permutations(idx)):
            T[p] = m[k]
    return T


class BeatLibrary:
    def __init__(self, run):
        self.run = Path(run)
        self.meta = json.loads((self.run / "flow_bins.json").read_text())
        self.index = json.loads((self.run / "beats" / "index.json").read_text())
        self.nf = len(self.index["fields"])
        self.bins = int(self.index["bins"])
        self.period = float(self.index["period"])
        self.beats = sorted(int(p.stem.split("_")[1]) for p in (self.run / "beats").glob("beat_*.bin"))
        m = self.meta
        self.fct, self.dx = int(m["out_factor"]), float(m["dx"])
        self.c1 = 8 // self.fct
        self.cpt = int(m["cells_per_tile"])
        txyz = np.asarray(m["tile_xyz"], np.int64).reshape(-1, 3)
        c = np.arange(self.cpt)
        self.gx = (txyz[:, 0:1] * self.c1 + (c % self.c1)[None]).ravel()
        self.gy = (txyz[:, 1:2] * self.c1 + ((c // self.c1) % self.c1)[None]).ravel()
        self.gz = (txyz[:, 2:3] * self.c1 + (c // (self.c1 * self.c1))[None]).ravel()
        self.lo = np.array([self.gx.min(), self.gy.min(), self.gz.min()])
        self.dims = np.array([self.gx.max(), self.gy.max(), self.gz.max()]) - self.lo + 1

    def grid(self):
        """origin (centre of cell 0, m), spacing (m), dims (x, y, z)"""
        o = np.asarray(self.meta["origin"], float) + (self.lo * self.fct + 0.5 * self.fct - 0.5) * self.dx
        return o, self.fct * self.dx, self.dims

    def read(self, cycle):
        raw = (self.run / "beats" / f"beat_{cycle:04d}.bin").read_bytes()
        out, p = {}, 0
        while p < len(raw):
            b, n = np.frombuffer(raw, np.uint32, 2, p)
            s = np.frombuffer(raw, np.float32, 1, p + 8)[0]
            p += 12
            idx = np.frombuffer(raw, np.uint32, n, p)
            p += 4 * n
            v = np.frombuffer(raw, np.float16, n * self.nf, p).reshape(n, self.nf).astype(np.float32)
            p += 2 * n * self.nf
            out[int(b)] = (idx, v, float(s))
        return out

    def dense(self, cycle, b, data=None):
        data = data or self.read(cycle)
        idx, v, _ = data[b]
        X, Y, Z = self.dims
        flat = ((self.gz[idx] - self.lo[2]) * Y + (self.gy[idx] - self.lo[1])) * X + (self.gx[idx] - self.lo[0])
        F = np.zeros((self.nf, Z * Y * X), np.float32)
        F[:, flat] = v.T
        F = F.reshape(self.nf, Z, Y, X)
        d = {"w": F[0], "mean": F[1:4], "cov": F[4:10], "m3": F[10:20], "wss": F[20:23]}
        if self.nf >= 25:
            d |= {"mz": F[23], "mz_std": F[24]}
        if self.nf >= 28:
            d["c_su"] = F[25:28]
        return d


def coarsen(w, mean, cov, m3, k):
    """moments of the mixture over k³ cells (weights w = coverage): returns w̄ (fraction), mean, cov, m3"""
    Z, Y, X = w.shape
    Z, Y, X = Z // k * k, Y // k * k, X // k * k

    def box(a):
        a = a[..., :Z, :Y, :X]
        return a.reshape(a.shape[:-3] + (Z // k, k, Y // k, k, X // k, k)).sum(axis=(-5, -3, -1))

    w = w[:Z, :Y, :X]
    W = box(w)
    safe = np.maximum(W, 1e-12)
    # raw moments of the mixture from each cell's central ones: E[u] = Σw μ / Σw, E[uu] = Σw (C + μμ) / Σw, …
    mu = box(w * mean[:, :Z, :Y, :X]) / safe
    e2 = np.stack([box(w * (cov[k_, :Z, :Y, :X] + mean[i, :Z, :Y, :X] * mean[j, :Z, :Y, :X])) for k_, (i, j) in enumerate(PAIR)]) / safe
    C = np.stack([e2[k_] - mu[i] * mu[j] for k_, (i, j) in enumerate(PAIR)])
    Cm = cov_mat(cov[:, :Z, :Y, :X])
    m = mean[:, :Z, :Y, :X]
    e3 = []
    for k_, (i, j, l) in enumerate(TRI):
        # E[u_i u_j u_l] of a cell = m3 + μ_i C_jl + μ_j C_il + μ_l C_ij + μ_i μ_j μ_l
        r = m3[k_, :Z, :Y, :X] + m[i] * Cm[j, l] + m[j] * Cm[i, l] + m[l] * Cm[i, j] + m[i] * m[j] * m[l]
        e3.append(box(w * r) / safe)
    E2 = cov_mat(e2)
    M3 = np.stack([e3[k_] - mu[i] * E2[j, l] - mu[j] * E2[i, l] - mu[l] * E2[i, j] + 2 * mu[i] * mu[j] * mu[l] for k_, (i, j, l) in enumerate(TRI)])
    return W / k**3, mu, C, M3


def sigma_points(mean, cov, m3, kappa=3.0):
    """moment-matched velocity samples for cells [N]: mean [N, 3], cov [N, 6], m3 [N, 10] → points [N, 7, 3], weights
    [N, 7] (sum 1). Along each principal axis e_a of the covariance (eigenvalue λ_a, third moment s_a = m3(e_a, e_a,
    e_a)) two points at μ + x± e_a with weights w± reproduce the variance λ_a and s_a (an asymmetric two-point rule),
    plus the centre: the mean and the covariance are exact, the skewness along the principal axes too"""
    N = len(mean)
    C = cov_mat(cov.T).transpose(2, 0, 1)  # [N, 3, 3]
    lam, E = np.linalg.eigh(C)
    lam = np.maximum(lam, 0.0)
    T = m3_tensor(m3.T).transpose(3, 0, 1, 2)  # [N, 3, 3, 3]
    s = np.einsum("nijk,nia,nja,nka->na", T, E, E, E)  # skew along the axes
    sd = np.sqrt(lam)
    # standardised skew γ = s / σ³; two points at σ(γ/2 ± sqrt(γ²/4 + κ)) with weights giving mean 0, variance σ² (the
    # rest of the weight sits at the centre)
    g = np.where(sd > 1e-9, s / np.maximum(sd, 1e-12) ** 3, 0.0)
    g = np.clip(g, -3.0, 3.0)
    r = np.sqrt(g**2 / 4 + kappa)
    xp, xm = sd * (g / 2 + r), sd * (g / 2 - r)  # xm < 0 < xp
    # weights: wp xp + wm xm = 0, wp xp² + wm xm² = σ² (per axis, out of a share 1/3 each)
    den = np.maximum(xp * (xp - xm), 1e-30)
    wp = np.where(sd > 1e-9, sd**2 / den, 0.0)
    wm = np.where(sd > 1e-9, -wp * xp / np.minimum(xm, -1e-30), 0.0)
    pts = np.repeat(mean[:, None, :], 7, axis=1).copy()
    wts = np.zeros((N, 7))
    for a in range(3):
        pts[:, 1 + 2 * a] += xp[:, a, None] * E[:, :, a]
        pts[:, 2 + 2 * a] += xm[:, a, None] * E[:, :, a]
        wts[:, 1 + 2 * a] = wp[:, a] / 3 * 3  # each axis carries its own variance: weights per axis are absolute
        wts[:, 2 + 2 * a] = wm[:, a] / 3 * 3
    wts[:, 0] = 1.0 - wts[:, 1:].sum(1)
    return pts, wts


if __name__ == "__main__":
    A = sys.argv[1:]
    lib = BeatLibrary(A[0])
    print(f"{lib.run}: beats {lib.beats}, {lib.bins} bins, cells {lib.fct * lib.dx * 1e3:.1f} mm, dims {lib.dims.tolist()}")
    if "check" in A:
        acc_stride = int(lib.meta.get("acc_stride", 10))
        nt = int(lib.meta["tiles"])
        acc = np.memmap(lib.run / "flow_bins.f32", np.float32, "r", shape=(lib.bins, nt, lib.cpt, acc_stride))
        f3 = lib.fct**3
        datas = {c: lib.read(c) for c in lib.beats}
        worst = 0.0
        for b in range(0, lib.bins, 7):
            num = np.zeros((nt * lib.cpt, 3))
            den = np.zeros(nt * lib.cpt)
            for c in lib.beats:
                idx, v, s = datas[c][b]
                ww = v[:, 0] * s * f3
                num[idx] += ww[:, None] * v[:, 1:4]
                den[idx] += ww
            a = np.asarray(acc[b]).reshape(-1, acc_stride)
            ok = (a[:, 0] > 0) & (den > 0)
            ref = a[ok, 1:4] / a[ok, 0:1]
            got = num[ok] / den[ok, None]
            err = np.sqrt(((got - ref) ** 2).sum(1))
            rel = np.sqrt((err**2).mean() / (ref**2).sum(1).mean())
            worst = max(worst, rel)
            print(f"  bin {b:2d}: {ok.sum()} cells, beats' mean vs flow_bins: rms {np.sqrt((err**2).mean()) * 1e3:.2f} mm/s ({rel * 100:.2f} % of rms speed), coverage Σw {den[ok].sum():.0f} vs {a[ok, 0].sum():.0f}")
        print(f"worst relative rms {worst * 100:.2f} % (f16 rounding)")
        if "--other" in A:
            lib2 = BeatLibrary(A[A.index("--other") + 1])
            for c in lib.beats:
                d1, d2 = datas[c], lib2.read(c)
                diffs = []
                for b in d1:
                    i1, v1, _ = d1[b]
                    i2, v2, _ = d2[b]
                    same = np.array_equal(i1, i2)
                    diffs.append((same, np.abs(v1[:, 1:4] - v2[:, 1:4]).max() if same else np.nan))
                print(f"  beat {c}: cells identical in {sum(d[0] for d in diffs)}/{len(diffs)} bins, max |Δ mean| {np.nanmax([d[1] for d in diffs]):.4f} m/s")
