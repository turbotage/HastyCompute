"""Time-segmentation basis study on the ph0001 field (CPU).

Data-space error of the separable approximation g(v, tau) ~ sum_l a_l(tau) D_l(v) for the realistic image x (warm
start, encode 0), whitened true coil maps, the real trajectory, against the per-sample noise sigma.

  E(n) = sum_c  mean_r | F_{k_rn}[ S_c x delta_n ] |^2 ,  delta_n(v) = g(v, tau_n) - sum_l a_l(tau_n) D_l(v)

near the k-space centre (|k| < KC): 4x4x4-voxel block sums of S_c x delta_n and a direct DFT at the actual sample
positions k_{r,n} of NR readouts (all 32 whitened coils); elsewhere the white model sum_v |x|^2 sum_c|S_c|^2 |delta|^2
(validated against full-resolution FFT shell averages in validate.py). R = sqrt(mean_n E(n) / (C sigma^2)).
"""
import numpy as np, os, json, time
import scipy.sparse as sp
from scipy.ndimage import uniform_filter1d

GBAR = 42.577478e6
TWO_PI = 2 * np.pi


class Case:
    def __init__(self, D, conc=True, r2s=False, fat=False, nr=48, kc=250.0, enc=0, verbose=True):
        S = os.path.join(D, 'sim_flow_10min/scan')
        meta = json.load(open(os.path.join(S, 'meta.json')))
        self.b0 = meta['b0']
        self.dims = tuple(meta['recon_dims'])
        nx, ny, nz = self.dims
        self.res = 0.002
        mm = np.asarray(np.load(os.path.join(D, 'lock_2mm/llr/mask_maps.npy'), mmap_mode='r')) > 0.5
        self.idx = np.flatnonzero(mm.ravel(order='F'))
        nf = len(self.idx)
        self.nf = nf
        i = self.idx % nx; j = (self.idx // nx) % ny; k = self.idx // (nx * ny)
        self.ijk = (i, j, k)
        pos = np.stack([(i - nx / 2) * self.res, (j - ny / 2) * self.res, (k - nz / 2) * self.res], 1)
        self.pos = pos
        ppm = np.asarray(np.load(os.path.join(S, 'field_ppm_recon.npy'), mmap_mode='r')).ravel(order='F')[self.idx]
        self.hz = ppm.astype(np.float64) * GBAR * self.b0 * 1e-6
        self.om = TWO_PI * self.hz
        # trajectory: tau, |k| profile, k of NR readouts spread over the scan
        smp = np.load(os.path.join(S, 'samples.npy'), mmap_mode='r')
        k0 = np.array(smp[0], dtype=np.float64)
        self.tau = k0[:, 3]
        self.na = len(self.tau)
        self.kap = np.linalg.norm(k0[:, :3], axis=1)
        ros = np.linspace(0, smp.shape[0] - 1, nr).astype(int)
        self.kr = np.stack([np.array(smp[r], dtype=np.float64)[:, :3] for r in ros], 0)  # nr x na x 3
        # concomitant (isotropic rotation average of the Maxwell terms, from the readout gradient; offset at ADC start
        # irrelevant): psi_c(v, tau) = W(tau) (z^2/3 + (x^2+y^2)/24),  W = gamma/B0 int |G|^2
        gam = TWO_PI * GBAR
        G = np.linalg.norm(np.gradient(k0[:, :3], self.tau, axis=0), axis=1) / gam
        dt = np.diff(self.tau, prepend=self.tau[0])
        self.Wc = gam / self.b0 * np.cumsum(G ** 2 * dt)
        self.mc = pos[:, 2] ** 2 / 3 + (pos[:, 0] ** 2 + pos[:, 1] ** 2) / 24
        self.conc = conc
        # R2*
        self.r2 = None
        if r2s:
            self.r2 = np.asarray(np.load(os.path.join(S, 'truth_r2s.npy'), mmap_mode='r')).ravel(order='F')[self.idx].astype(np.float64)
        # fat
        self.ff = None
        a = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048]); a /= a.sum()
        self.fat_amp = a
        self.fat_om = TWO_PI * GBAR * self.b0 * np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]) * 1e-6
        if fat:
            ffa = np.load(os.path.join(S, 'truth_fat_fraction.npy'), mmap_mode='r')
            self.ff = np.clip(np.asarray(ffa[..., 0]).ravel(order='F')[self.idx].astype(np.float64), 0, 1)
        # image and coils (whitened)
        xw = np.load(os.path.join(D, 'fix_2mm/llr_s/warm_full.npy'), mmap_mode='r')
        self.x = np.asarray(xw[..., enc]).ravel(order='F')[self.idx].astype(np.complex64)
        acc = np.zeros(self.dims, np.float32)
        for e in range(xw.shape[3]):
            acc += np.abs(np.asarray(xw[..., e]))
        for ax, r in [(0, 4), (1, 4), (2, 1)]:
            for _ in range(3):
                acc = uniform_filter1d(acc, 2 * r + 1, axis=ax, mode='nearest')
        self.mag_s = acc.ravel(order='F')[self.idx] / xw.shape[3]  # smoothed |x| (mean over encodes)
        psi = np.array(meta['psi'], dtype=np.float64)
        ev, U = np.linalg.eigh(psi)
        Wh = (U / np.sqrt(ev)) @ U.T  # psi^{-1/2}
        sm = np.load(os.path.join(S, 'smaps_true.npy'), mmap_mode='r')
        nc = sm.shape[3]
        Sraw = np.empty((nc, nf), np.complex64)
        for c in range(nc):
            Sraw[c] = np.asarray(sm[..., c]).ravel(order='F')[self.idx]
        self.S = (Wh.astype(np.complex64) @ Sraw)  # whitened maps, nc x nf
        del Sraw
        self.nc = nc
        self.cE = (np.abs(self.S) ** 2).sum(0).astype(np.float64)
        self.wE = np.abs(self.x).astype(np.float64) ** 2 * self.cE  # energy the operator sees per voxel
        self.wEs = self.mag_s.astype(np.float64) ** 2 * self.cE      # smoothed (data-derived, recon-available) version
        # measured whitened data energy per ADC sample (coil sum, mean over readouts) and the noise
        dcl = np.load(os.path.join(S, 'data_clean.npy'), mmap_mode='r')
        Emeas = np.zeros(self.na); ss = 0.0
        ros2 = np.linspace(0, dcl.shape[0] - 1, 200).astype(int)
        for r in ros2:
            yr = np.array(dcl[r]).astype(np.complex128)
            ss += (np.abs(yr) ** 2).sum()
            Emeas += (np.abs(yr @ Wh.T) ** 2).sum(1)
        self.Emeas = Emeas / len(ros2)
        self.rms_raw = np.sqrt(ss / (len(ros2) * self.na * nc))
        self.sigma = meta['noise_rel'] * self.rms_raw  # per whitened coil, data units
        # block structure (4x4x4) for the near-centre DFT
        bi, bj, bk = i // 4, j // 4, k // 4
        nb = (-(-nx // 4), -(-ny // 4), -(-nz // 4))
        bid = bi + nb[0] * (bj + nb[1] * bk)
        ub, inv = np.unique(bid, return_inverse=True)
        self.nblk = len(ub)
        self.Bmat = sp.csr_matrix((np.ones(nf, np.float32), (inv, np.arange(nf))), shape=(len(ub), nf))
        ub_i = ub % nb[0]; ub_j = (ub // nb[0]) % nb[1]; ub_k = ub // (nb[0] * nb[1])
        self.bpos = np.stack([(4 * ub_i + 1.5 - nx / 2) * self.res, (4 * ub_j + 1.5 - ny / 2) * self.res, (4 * ub_k + 1.5 - nz / 2) * self.res], 1)
        # evaluation indices: near centre (kap < kc) dense, elsewhere every 40th
        nc_idx = np.flatnonzero(self.kap < kc)
        sel = lambda m: m < 10 or (m < 40 and m % 2 == 0) or m % 4 == 0
        near = [n for n in nc_idx if n < self.na // 2 and sel(n)] + [n for n in nc_idx if n >= self.na // 2 and sel(self.na - 1 - n)]
        self.near = np.array(sorted(set(near)))
        far = np.arange(0, self.na, 60)
        far = far[self.kap[far] >= kc]
        self.far = far
        self.kc = kc
        self.SX = (self.S * self.x[None, :])  # nc x nf, c64
        self._bg = {}
        if verbose:
            print(f'case: {nf} fit voxels, {self.nblk} blocks, near-centre n {len(self.near)}, far n {len(self.far)}, conc max {np.abs(self.Wc[-1] * self.mc).max() / TWO_PI:.2f} cycles, sigma {self.sigma:.2f}, fat {fat}, r2s {r2s}', flush=True)

    # ---------------------------------------------------------------- the time function
    def g(self, t, sel=None):
        """g(v, t) over the fit voxels (or index subset sel), complex128"""
        om = self.om if sel is None else self.om[sel]
        ph = om * t
        if self.conc:
            n = np.interp(t, self.tau, self.Wc)
            ph = ph + n * (self.mc if sel is None else self.mc[sel])
        z = np.exp(-1j * ph)
        if self.r2 is not None:
            r2 = self.r2 if sel is None else self.r2[sel]
            z *= np.exp(-r2 * (t - self.tau[0]))
        if self.ff is not None:
            ff = self.ff if sel is None else self.ff[sel]
            c = np.sum(self.fat_amp * np.exp(-1j * self.fat_om * t))
            z *= (1 - ff) + ff * c
        return z

    def gseg(self, t, sel=None):
        """the function the segments represent: g without the per-sample (Taylor) quadratic part Q(v)(t - tref)"""
        z = self.g(t, sel)
        Q = getattr(self, 'taylorQ', None)
        if Q is not None:
            q = Q if sel is None else Q[sel]
            z = z * np.exp(1j * q * (t - self.tref))
        return z

    def fatc(self, t):
        return np.sum(self.fat_amp * np.exp(-1j * self.fat_om * t))

    # ---------------------------------------------------------------- near-centre DFT
    def phase(self, n):
        k = self.kr[:, n, :]  # nr x 3
        return np.exp(-1j * (self.bpos @ k.T)).astype(np.complex64)  # nblk x nr

    def bg(self, n):
        """block images of S_c x g(tau_n) (nc x nblk), cached"""
        if n not in self._bg:
            z = (self.SX * self.g(self.tau[n]).astype(np.complex64)[None, :])
            self._bg[n] = (self.Bmat @ z.T).T.astype(np.complex64)
        return self._bg[n]

    def signal_near(self):
        """model signal energy per near-centre sample (coil sum, mean over the NR readouts)"""
        out = []
        for n in self.near:
            F = self.bg(n) @ self.phase(n)
            out.append((np.abs(F) ** 2).sum(0).mean())
        return np.array(out)


def fit_subset(case, step=3):
    return np.arange(0, case.nf, step)


def ls_interp(case, Dsub, sub, w, ts):
    """weighted LS interpolators a(t) for diagonals Dsub (L x |sub|) at times ts: (L x len(ts))"""
    sw = np.sqrt(w) if w is not None else np.ones(len(sub))
    B = (Dsub * sw[None, :]).T.astype(np.complex128)  # |sub| x L
    G = B.conj().T @ B
    tr = np.trace(G).real / G.shape[0]
    G += 1e-9 * tr * np.eye(G.shape[0])
    C = np.linalg.cholesky(G)
    out = np.empty((Dsub.shape[0], len(ts)), np.complex128)
    for q, t in enumerate(ts):
        y = case.gseg(t, sub) * sw
        rhs = B.conj().T @ y
        out[:, q] = np.linalg.solve(C.conj().T, np.linalg.solve(C, rhs))
    return out


def voxel_weights(case, kind, sub):
    if kind == 'count':
        return None
    if kind == 'mass':  # recon_p2np --seg-weight mass: smoothed magnitude / median, clamped [0.3, 3]
        m = case.mag_s[sub].astype(np.float64)
        return np.clip(m / np.median(m), 0.3, 3.0)
    if kind == 'energy':  # smoothed |x|^2 x whitened coil energy (recon-available)
        return case.wEs[sub] / np.median(case.wEs[sub])
    if kind == 'energy_true':  # the exact |x|^2 x coil energy
        return case.wE[sub] / np.median(case.wE[sub])
    raise ValueError(kind)


def time_weights(case, kind, ts):
    """per-time weight (before quadrature)"""
    if kind == 'none':
        return np.ones(len(ts))
    if kind == 'data':  # measured data energy at that ADC time (svdw)
        return np.interp(ts, case.tau, case.Emeas)
    if kind == 'snr':  # data energy / noise + 1 (noise-referenced: error matters where signal >> noise; floor 1)
        return np.interp(ts, case.tau, case.Emeas) / (case.nc * case.sigma ** 2) + 1.0
    raise ValueError(kind)


def time_grid(case, kind, ns):
    """(times, quadrature weights) for the SVD Gram"""
    t0, t1 = case.tau[0], case.tau[-1]
    if kind == 'uniform':  # recon_p2np: 96 uniform samples, no quadrature weights
        ts = np.linspace(t0, t1, ns)
        return ts, np.ones(ns)
    if kind == 'ends':  # dense near both ends (ADC-sample resolution in the first/last 60 samples), quadrature dt
        n = case.na
        m = ns // 3
        a = np.arange(0, 60, max(1, 60 // m))
        mid = np.linspace(60, n - 61, ns - 2 * len(a)).round().astype(int)
        ii = np.unique(np.concatenate([a, mid, n - 1 - a]))
        ts = case.tau[ii]
        q = np.gradient(ts)
        return ts, q / q.mean()
    raise ValueError(kind)


class Basis:
    def __init__(self, D, ts_fn):
        self.D = D  # L x nf (c64)
        self.a = ts_fn  # t array -> L x len


def basis_fixed(case, L, nodes='uniform', vw='count', step=3):
    t0, t1 = case.tau[0], case.tau[-1]
    if nodes == 'uniform':
        tl = np.linspace(t0, t1, L) if L > 1 else np.array([0.5 * (t0 + t1)])
    elif nodes == 'cheb':  # Chebyshev-Lobatto: clustered at both ends (where the k-space centre is sampled)
        tl = 0.5 * (t0 + t1) - 0.5 * (t1 - t0) * np.cos(np.pi * np.arange(L) / (L - 1))
    else:
        tl = np.asarray(nodes)
    D = np.stack([case.gseg(t).astype(np.complex64) for t in tl], 0)
    sub = fit_subset(case, step)
    w = voxel_weights(case, vw, sub)
    Dsub = D[:, sub]
    return Basis(D, lambda ts: ls_interp(case, Dsub, sub, w, ts))


def basis_svd(case, L, vw='count', tw='none', grid='uniform', ns=96, step=3, return_eig=False):
    sub = fit_subset(case, step)
    w = voxel_weights(case, vw, sub)
    sw = np.sqrt(w) if w is not None else np.ones(len(sub))
    ts, q = time_grid(case, grid, ns)
    om_t = time_weights(case, tw, ts) * q
    st = np.sqrt(np.maximum(om_t, 1e-6 * om_t.max()))
    ns = len(ts)
    Gm = np.zeros((ns, ns), np.complex128)
    for c0 in range(0, len(sub), 65536):
        ss = sub[c0:c0 + 65536]
        rows = np.stack([case.g(t, ss) for t in ts], 1) * sw[c0:c0 + 65536, None] * st[None, :]
        Gm += rows.conj().T @ rows
    ev, V = np.linalg.eigh(Gm)
    ev = ev[::-1]; V = V[:, ::-1]
    Vl = V[:, :L]
    D = np.zeros((L, case.nf), np.complex64)
    for c0 in range(0, case.nf, 131072):
        rows = np.stack([case.g(t, np.arange(c0, min(c0 + 131072, case.nf))) for t in ts], 1) * st[None, :]
        D[:, c0:c0 + rows.shape[0]] = (rows @ Vl).T.astype(np.complex64)
    Dsub = D[:, sub]
    b = Basis(D, lambda tt: ls_interp(case, Dsub, sub, w, tt))
    b.ev = ev
    return b


def evaluate(case, basis, extra=False):
    """data-space error per evaluated ADC index, R (error rms / noise rms), proxies"""
    D = basis.D
    L = D.shape[0]
    res = {}
    # near centre: block DFT at the real sample positions
    a_near = basis.a(case.tau[case.near]).astype(np.complex64)
    BD = np.stack([(case.Bmat @ (case.SX * D[l][None, :]).T).T for l in range(L)], 0).astype(np.complex64)  # L x nc x nblk
    En = np.empty(len(case.near)); Yn = np.empty(len(case.near))
    for q, n in enumerate(case.near):
        bimg = case.bg(n) - np.tensordot(a_near[:, q], BD, axes=(0, 0))
        F = bimg @ case.phase(n)
        En[q] = (np.abs(F) ** 2).sum(0).mean()
    # far: white model
    a_far = basis.a(case.tau[case.far]).astype(np.complex64)
    Ef = np.empty(len(case.far)); epsw = np.empty(len(case.far))
    for q, n in enumerate(case.far):
        d = case.g(case.tau[n]).astype(np.complex64) - a_far[:, q] @ D
        e2 = np.abs(d).astype(np.float64) ** 2
        Ef[q] = (case.wE * e2).sum()
    # proportional proxy at near-centre n too (voxel-energy-weighted relative error x measured data energy)
    epsn = np.full(len(case.near), np.nan)
    for q, n in enumerate(case.near):
        if q % 3 and q != len(case.near) - 1:
            continue
        g = case.g(case.tau[n]).astype(np.complex64)
        d = g - a_near[:, q] @ D
        epsn[q] = (case.wE * np.abs(d) ** 2).sum() / (case.wE * np.abs(g) ** 2).sum()
    res.update(near=case.near.tolist(), En=En.tolist(), far=case.far.tolist(), Ef=Ef.tolist(), epsn=epsn.tolist())
    return res


def integrate(case, res, scale):
    """R = sqrt(mean_n E(n) / (C sigma^2)); E in model units, scale = model/measured data energy"""
    n_all = np.arange(case.na)
    near = np.array(res['near']); far = np.array(res['far'])
    nn = np.concatenate([near, far]); ee = np.concatenate([res['En'], res['Ef']])
    o = np.argsort(nn)
    E = np.interp(n_all, nn[o], ee[o])
    noise = case.nc * case.sigma ** 2 * scale
    Rnear = np.sqrt(E[case.kap < case.kc].sum() / case.na / noise)
    Rfar = np.sqrt(E[case.kap >= case.kc].sum() / case.na / noise)
    R = np.sqrt(E.mean() / noise)
    # proportional proxy: eps_w^2(n) x measured data energy (scaled), near n only + far white
    en = np.array(res['epsn']); ok = np.isfinite(en)
    ep = np.interp(n_all, near[ok], en[ok])
    Eprop = ep * case.Emeas * scale
    Rprop_near = np.sqrt(Eprop[case.kap < case.kc].sum() / case.na / noise)
    return dict(R=R, Rnear=Rnear, Rfar=Rfar, Rprop_near=Rprop_near)


# ---------------------------------------------------------------- v2 evaluation: near block DFT + mid/far rss-FFT shells
import scipy.fft as sfft


def setup_shells(case, kn=120.0, h=30.0):
    """|k| of every FFT bin of the full grid, sorted, and the evaluation indices of v2"""
    nx, ny, nz = case.dims
    kx = TWO_PI * sfft.fftfreq(nx, case.res); ky = TWO_PI * sfft.fftfreq(ny, case.res); kz = TWO_PI * sfft.fftfreq(nz, case.res)
    K = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).ravel(order='F')
    o = np.argsort(K)
    case.shell_order = o
    case.shell_k = K[o]
    case.shell_h = h
    case.kn = kn
    case.rss = np.sqrt(case.cE).astype(np.float32)
    case.rssx = (case.rss * case.x).astype(np.complex64)
    na = case.na
    near = np.flatnonzero(case.kap < kn)
    sel = lambda m: m < 10 or (m < 40 and m % 2 == 0) or m % 4 == 0
    case.near = np.array([n for n in near if sel(n if n < na // 2 else na - 1 - n)])
    # mid/far: dense where kap changes fast near the ends, coarse in the middle
    mid = np.flatnonzero(case.kap >= kn)
    lo, hi = mid[0], mid[-1]
    pts = set()
    for start, sgn in [(lo, 1), (hi, -1)]:
        step, m = 4, start
        while lo <= m <= hi and abs(m - start) < 1000:
            pts.add(int(m)); m += sgn * step; step = min(int(step * 1.25) + 1, 60)
    pts.update(range(lo, hi + 1, 60))
    case.far = np.array(sorted(p for p in pts if case.kap[p] >= kn))


def shell_power(case, img_fit):
    """mean |F|^2 over the shell at every case.far kappa, for an image given on the fit voxels"""
    nx, ny, nz = case.dims
    grid = np.zeros(nx * ny * nz, np.complex64)
    grid[case.idx] = img_fit
    F = sfft.fftn(grid.reshape((nx, ny, nz), order='F'), workers=int(os.environ.get('SEG_FFT_WORKERS', '4')))
    P = (np.abs(F.ravel(order='F')) ** 2)[case.shell_order]
    return P


def shell_at(case, P, kap):
    a = np.searchsorted(case.shell_k, kap - case.shell_h / 2); b = np.searchsorted(case.shell_k, kap + case.shell_h / 2)
    return P[a:b].mean()


def evaluate2(case, basis):
    D = basis.D
    L = D.shape[0]
    a_near = basis.a(case.tau[case.near]).astype(np.complex64)
    BD = np.stack([(case.Bmat @ (case.SX * D[l][None, :]).T).T for l in range(L)], 0).astype(np.complex64)
    En = np.empty(len(case.near))
    for q, n in enumerate(case.near):
        bimg = case.bg(n) - np.tensordot(a_near[:, q], BD, axes=(0, 0))
        F = bimg @ case.phase(n)
        En[q] = (np.abs(F) ** 2).sum(0).mean()
    a_far = basis.a(case.tau[case.far]).astype(np.complex64)
    Ef = np.empty(len(case.far)); Ew = np.empty(len(case.far))
    for q, n in enumerate(case.far):
        d = case.g(case.tau[n]).astype(np.complex64) - a_far[:, q] @ D
        P = shell_power(case, case.rssx * d)
        Ef[q] = shell_at(case, P, case.kap[n])
        Ew[q] = (case.wE * np.abs(d) ** 2).sum()
    return dict(near=case.near.tolist(), En=En.tolist(), far=case.far.tolist(), Ef=Ef.tolist(), Ew=Ew.tolist())


def integrate2(case, res, scale):
    n_all = np.arange(case.na)
    near = np.array(res['near']); far = np.array(res['far'])
    nn = np.concatenate([near, far]); ee = np.concatenate([res['En'], res['Ef']])
    o = np.argsort(nn)
    # interpolate log E (E spans decades between samples near the ends)
    E = np.exp(np.interp(n_all, nn[o], np.log(np.maximum(ee[o], 1e-300))))
    noise = case.nc * case.sigma ** 2 * scale
    m_near = case.kap < case.kn
    return dict(R=float(np.sqrt(E.mean() / noise)), Rnear=float(np.sqrt(E[m_near].sum() / case.na / noise)), Rfar=float(np.sqrt(E[~m_near].sum() / case.na / noise)))


# ---------------------------------------------------------------- Rust ports (hasty_nufft::segbasis) and sweep helpers
def seg_time_grid(tau, w, ns):
    """port of hasty_nufft::seg_time_grid"""
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
    """volume-weighted mean data energy over the sampled k-space ball (shell volume per ADC sample)"""
    dV = 4 * np.pi * kap ** 2 * np.abs(np.gradient(kap))
    return (E * dV).sum() / dV.sum()


def mix_weights(case):
    """1 + (E_n + E_vol) / (C sigma^2): the proportional part plus the white floor of a spatially rough error"""
    return 1.0 + (case.Emeas + vol_energy(case.Emeas, case.kap)) / (case.nc * case.sigma ** 2)


def snr_weights(case):
    return 1.0 + case.Emeas / (case.nc * case.sigma ** 2)


class SvdFamily:
    """weighted Gram once, bases for any L by slicing"""
    def __init__(self, case, vw='count', tw='none', grid='uniform', ns=96, step=3):
        self.case = case
        sub = fit_subset(case, step); self.sub = sub
        w = voxel_weights(case, vw, sub); self.w = w
        sw = np.sqrt(w) if w is not None else np.ones(len(sub))
        if grid == 'uniform':
            ts = np.linspace(case.tau[0], case.tau[-1], ns)
            om = time_weights(case, tw, ts)
            st = np.sqrt(np.maximum(om, 1e-6 * om.max()))  # as build_segments_svd_weighted (floor 1e-6)
        elif grid == 'rust':  # seg_time_grid with per-ADC weights
            wa = {'snr': lambda: snr_weights(case), 'mix': lambda: mix_weights(case), 'data': lambda: case.Emeas, 'none': lambda: np.ones(case.na)}[tw]()
            ts, q = seg_time_grid(case.tau, wa, ns)
            st = np.sqrt(q)
        self.ts = ts
        Gm = np.zeros((len(ts), len(ts)), np.complex128)
        for c0 in range(0, len(sub), 65536):
            ss = sub[c0:c0 + 65536]
            rows = np.stack([case.gseg(t, ss) for t in ts], 1) * sw[c0:c0 + 65536, None] * st[None, :]
            Gm += rows.conj().T @ rows
        ev, V = np.linalg.eigh(Gm)
        self.ev = np.maximum(ev[::-1], 0); V = V[:, ::-1]
        self.V = V; self.st = st

    def basis(self, L):
        case = self.case
        Vl = self.V[:, :L]
        if not hasattr(self, 'Dall') or self.Dall.shape[0] < L:
            Lm = max(L, 16)
            Vm = self.V[:, :Lm]
            D = np.zeros((Lm, case.nf), np.complex64)
            for c0 in range(0, case.nf, 131072):
                ii = np.arange(c0, min(c0 + 131072, case.nf))
                rows = np.stack([case.gseg(t, ii) for t in self.ts], 1) * self.st[None, :]
                D[:, c0:c0 + len(ii)] = (rows @ Vm).T.astype(np.complex64)
            self.Dall = D
        D = self.Dall[:L]
        Dsub = D[:, self.sub]
        return Basis(D, lambda tt: ls_interp(case, Dsub, self.sub, self.w, tt))

    def tail(self, L):
        return np.sqrt(self.ev[L:].sum() / self.ev.sum())


def proxies(case, basis, w_kind='energy'):
    """separable proxies over all evaluation n: eps_w^2(n) with smoothed energy voxel weights (fit voxels)"""
    ns = np.unique(np.concatenate([case.near, case.far]))
    A = basis.a(case.tau[ns]).astype(np.complex64)
    w = case.wEs
    eps = np.empty(len(ns))
    for q, n in enumerate(ns):
        g = case.g(case.tau[n]).astype(np.complex64)
        d = g - A[:, q] @ basis.D
        eps[q] = (w * np.abs(d) ** 2).sum() / (w * np.abs(g) ** 2).sum()
    e = np.interp(np.arange(case.na), ns, eps)
    snr = case.Emeas / (case.nc * case.sigma ** 2)
    evol = (case.wEs.sum()) / case.scale_model  # smoothed white-equivalent data energy per sample, measured units
    snr_vol = evol / (case.nc * case.sigma ** 2)
    return dict(P_prop=float(np.sqrt((snr * e).mean())), P_mix=float(np.sqrt(((snr + snr_vol) * e).mean())), P_snr1=float(np.sqrt(((snr + 1) * e).sum() / (snr + 1).sum())), snr_vol=float(snr_vol))


def setup_v3(case, kn=120.0, h=30.0):
    setup_shells(case, kn, h)
    mid = np.flatnonzero(case.kap >= kn)
    lo, hi = mid[0], mid[-1]
    pts = set()
    for start, sgn in [(lo, 1), (hi, -1)]:
        step, m = 4, start
        while lo <= m <= hi and abs(m - start) < 900:
            pts.add(int(m)); m += sgn * step; step = int(step * 1.45) + 1
    pts.update(range(lo, hi + 1, 150))
    case.far = np.array(sorted(p for p in pts if case.kap[p] >= kn))


def evaluate3(case, basis):
    """v2 metric in one pass, with the separable proxies and optional Taylor-2 per-sample quadratic factor"""
    D = basis.D
    L = D.shape[0]
    Q = getattr(case, 'taylorQ', None)
    nn = np.concatenate([case.near, case.far])
    A = basis.a(case.tau[nn]).astype(np.complex64)
    an, af = A[:, :len(case.near)], A[:, len(case.near):]
    mats = [D] if Q is None else [D, (Q[None, :] * D).astype(np.complex64), (Q[None, :] ** 2 * D).astype(np.complex64)]
    BDs = [np.stack([(case.Bmat @ (case.SX * M[l][None, :]).T).T for l in range(L)], 0).astype(np.complex64) for M in mats]
    def approx(a, t):
        z = a @ D
        if Q is not None:
            dt = t - case.tref
            z = z * (1 - 1j * Q * dt - 0.5 * (Q * dt) ** 2)
        return z.astype(np.complex64)
    w = case.wEs
    En = np.empty(len(case.near)); epsn = np.full(len(case.near), np.nan)
    for q, n in enumerate(case.near):
        a = an[:, q]
        comb = np.tensordot(a, BDs[0], axes=(0, 0))
        if Q is not None:
            dt = case.tau[n] - case.tref
            comb = comb - 1j * dt * np.tensordot(a, BDs[1], axes=(0, 0)) - 0.5 * dt ** 2 * np.tensordot(a, BDs[2], axes=(0, 0))
        F = (case.bg(n) - comb) @ case.phase(n)
        En[q] = (np.abs(F) ** 2).sum(0).mean()
        if q % 3 == 0 or q == len(case.near) - 1:
            g = case.g(case.tau[n]).astype(np.complex64)
            d = g - approx(a, case.tau[n])
            epsn[q] = (w * np.abs(d) ** 2).sum() / (w * np.abs(g) ** 2).sum()
    Ef = np.empty(len(case.far)); epsf = np.empty(len(case.far))
    for q, n in enumerate(case.far):
        g = case.g(case.tau[n]).astype(np.complex64)
        d = g - approx(af[:, q], case.tau[n])
        P = shell_power(case, case.rssx * d)
        Ef[q] = shell_at(case, P, case.kap[n])
        epsf[q] = (w * np.abs(d) ** 2).sum() / (w * np.abs(g) ** 2).sum()
    res = dict(near=case.near.tolist(), En=En.tolist(), far=case.far.tolist(), Ef=Ef.tolist())
    # proxies: eps_w^2(n) interpolated over all ADC samples
    okn = np.isfinite(epsn)
    ne = np.concatenate([case.near[okn], case.far]); ee = np.concatenate([epsn[okn], epsf])
    o = np.argsort(ne)
    e = np.exp(np.interp(np.arange(case.na), ne[o], np.log(np.maximum(ee[o], 1e-300))))
    snr = case.Emeas / (case.nc * case.sigma ** 2)
    res['P_prop'] = float(np.sqrt((snr * e).mean()))
    res['P_snr1'] = float(np.sqrt(((snr + 1) * e).mean()))
    return res
