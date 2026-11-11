"""Time-segment bases (torch ports of hasty_nufft::segbasis and recon_p2np's svdn / fixed+energy) on the object voxels.

g(v, t): per-voxel time function the segments represent (callable (t: f64 tensor (nt,), sl) -> (nt, |sl|) complex64).
svdn: weighted rank-L SVD on seg_time_grid (96 samples) with time weights w_n = 1 + (E_n + E_vol)/(C sigma^2) and
energy voxel weights (smoothed |x|^2 sum|S|^2 / median); interpolators by voxel-weighted least squares (fit voxels =
every 3rd support voxel, as recon_p2np). fixed: uniform nodes, D_l = g(., t_l), the same LS interpolators (weighted =
`--seg-weight energy`, unweighted = Sutton-Noll-Fessler histogram LS).
"""
import numpy as np, torch
import pb


def seg_time_grid(tau, w, ns=96):
    """port of hasty_nufft::seg_time_grid: a third uniform, the rest equidistributing sqrt(w); quadrature weights"""
    na = len(tau); ns = int(np.clip(ns, 2, na)); nu = max(ns // 3, 2)
    idx = [int(round((na - 1) * i / (nu - 1))) for i in range(nu)]
    cum = np.cumsum(np.sqrt(np.maximum(w, 0))); acc = cum[-1]
    nw = max(ns - nu, 1)
    if acc > 0:
        j = 0
        for i in range(nw):
            target = acc * (i + 0.5) / nw
            while j + 1 < na and cum[j] < target:
                j += 1
            idx.append(j)
    idx = np.unique(idx)
    tw = np.zeros(len(idx)); g = 0
    for n in range(na):
        while g + 1 < len(idx) and abs(idx[g + 1] - n) <= abs(n - idx[g]):
            g += 1
        tw[g] += max(w[n], 0)
    return tau[idx], tw


def vol_energy(E, kap):
    dV = 4 * np.pi * kap ** 2 * np.abs(np.gradient(kap))
    return (E * dV).sum() / dV.sum()


def noise_time_weights(E, kap, C, sigma):
    return 1.0 + (E + vol_energy(E, kap)) / (C * sigma ** 2)


def smooth_mag(obj, x):
    from scipy.ndimage import uniform_filter1d
    nx, ny, nz = obj.dims
    g = np.zeros(nx * ny * nz, np.float32); g[obj.idx] = np.abs(x)
    a = g.reshape((nx, ny, nz), order='F')
    for ax, r in [(0, 4), (1, 4), (2, 1)]:
        for _ in range(3):
            a = uniform_filter1d(a, 2 * r + 1, axis=ax, mode='nearest')
    return a.ravel(order='F')[obj.idx]


class Basis:
    def __init__(self, D, interp, kind, L, ev=None, nodes=None):
        self.D, self.interp, self.kind, self.L, self.ev, self.nodes = D, interp, kind, L, ev, nodes

    def approx(self, tau, sl):
        """sum_l a_l(tau) D_l(v) for tau (ns,) -> ns x |sl| (interpolators cached per time set)"""
        key = (len(tau), float(tau[0]), float(tau[-1]), float(tau.sum()))
        if getattr(self, '_key', None) != key:
            self._A = self.interp(tau)  # ns x L complex64
            self._key = key
        return self._A @ self.D[:, sl]


def ls_interp_fn(g, Dfit, fit, wv):
    sw = torch.sqrt(wv)
    B = (Dfit.to(torch.complex128) * sw[None, :]).T  # nfit x L (f64: equal-weight nodes at L > 14 are ill-conditioned in f32)
    G = B.conj().T @ B
    G += 1e-9 * torch.trace(G).real / G.shape[0] * torch.eye(G.shape[0], dtype=torch.complex128, device=pb.DEV)
    Gi = torch.linalg.inv(G)

    def interp(tau):
        out = []
        for c0 in range(0, len(tau), 64):
            y = g(tau[c0:c0 + 64], fit).to(torch.complex128) * sw[None, :]  # nt x nfit
            rhs = y @ B.conj()  # nt x L
            out.append((rhs @ Gi.T).to(torch.complex64))
        return torch.cat(out, 0)
    return interp


class Family:
    """weighted Gram once (svdn), bases for any L"""

    def __init__(self, obj, g, tw_tau, tw_w, wv_all, fit_step=3, ns=96):
        self.obj, self.g = obj, g
        self.fit = torch.arange(0, obj.N, fit_step, device=pb.DEV)
        self.wv = torch.tensor(wv_all, dtype=torch.float64, device=pb.DEV)[self.fit]
        ts, q = seg_time_grid(tw_tau, tw_w, ns)
        self.ts = torch.tensor(ts, dtype=torch.float64, device=pb.DEV)
        self.st = torch.tensor(np.sqrt(q), dtype=torch.float32, device=pb.DEV)
        sw = torch.sqrt(self.wv).to(torch.float32)
        Gm = torch.zeros((len(ts), len(ts)), dtype=torch.complex128, device=pb.DEV)
        for c0 in range(0, len(self.fit), 262144):
            sl = self.fit[c0:c0 + 262144]
            rows = g(self.ts, sl).T * sw[c0:c0 + 262144, None] * self.st[None, :]  # nfit x nt
            Gm += (rows.conj().T @ rows).to(torch.complex128)
        ev, V = torch.linalg.eigh(Gm)
        self.ev = torch.clamp(ev.flip(0).real, min=0).cpu().numpy()
        self.V = V.flip(1)

    def tail(self, L):
        return float(np.sqrt(self.ev[L:].sum() / self.ev.sum()))

    def basis(self, L):
        obj = self.obj
        Vl = self.V[:, :L].to(torch.complex64)
        D = torch.zeros((L, obj.N), dtype=torch.complex64, device=pb.DEV)
        for c0 in range(0, obj.N, 262144):
            sl = torch.arange(c0, min(c0 + 262144, obj.N), device=pb.DEV)
            rows = self.g(self.ts, sl).T * self.st[None, :]
            D[:, c0:c0 + len(sl)] = (rows @ Vl).T
        return Basis(D, ls_interp_fn(self.g, D[:, self.fit], self.fit, self.wv), 'svdn', L, ev=self.ev)


def basis_fixed(obj, g, L, t0, t1, wv_all=None, fit_step=3, nodes=None):
    fit = torch.arange(0, obj.N, fit_step, device=pb.DEV)
    tl = np.linspace(t0, t1, L) if nodes is None else np.asarray(nodes)
    ttl = torch.tensor(tl, dtype=torch.float64, device=pb.DEV)
    D = torch.zeros((L, obj.N), dtype=torch.complex64, device=pb.DEV)
    for c0 in range(0, obj.N, 262144):
        sl = torch.arange(c0, min(c0 + 262144, obj.N), device=pb.DEV)
        D[:, c0:c0 + len(sl)] = g(ttl, sl)
    wv = torch.ones(len(fit), dtype=torch.float64, device=pb.DEV) if wv_all is None else torch.tensor(wv_all, dtype=torch.float64, device=pb.DEV)[fit]
    return Basis(D, ls_interp_fn(g, D[:, fit], fit, wv), 'fixed' if wv_all is None else 'fixed_e', L, nodes=tl)


def deim_nodes(basis, tgrid):
    """QDEIM: pivoted QR of the interpolator matrix a(t) (nt x L) on a fine time grid -> L node times whose cardinal
    transform a'(t) = a(t) A(nodes)^{-1} makes the basis interpolatory (D'_l ~ g(., t_l)); returns node times and the
    condition number of A(nodes)"""
    import scipy.linalg as sla
    A = basis.interp(tgrid).cpu().numpy()  # nt x L
    _, _, piv = sla.qr(A.conj().T, pivoting=True, mode='economic')
    nodes = np.sort(piv[:basis.L])
    An = A[nodes]
    return tgrid[nodes].cpu().numpy() if torch.is_tensor(tgrid) else tgrid[nodes], np.linalg.cond(An), nodes
