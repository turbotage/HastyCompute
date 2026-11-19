"""Shared pieces of the sample-compression study (research/sample_compression.md).

Per readout, the range of the forward operator restricted to that readout's samples is spanned by the point-source
signatures (columns)
    v_p(t_j) = beta_{b_p}(u_j) * exp(-i [k_j.(r_p) + w_p tau_j + c_j . mono(r_p)])
r_p in the ROI (cube or ball of the box), w_p in the off-resonance + fat band, b_p one of the cardiac knots active on
the readout (u_j = per-sample cardiac knot coordinate, as splinerecon: (phase + tau/RR) * 16 - 0.5), c_j the sample's
concomitant coefficients [z^2, x^2+y^2, xz, yz]. Coil maps, segment diagonals and the respiratory weight (per readout)
multiply the image or the whole readout and do not enlarge this span (S_c x is box-supported; e^{-i w tau} is the w
factor; P_l a_l(tau) is a fit of it).

Compression of a readout = interpolative decomposition along time: a subset S of its samples and a sparse E (n x |S|)
with  v(t_j) ~= sum_{q in S} E_jq v(t_q)  for every column v.  Then for every image x:
    ||A x - y||^2 = ||W^{1/2} (A_S x - yt)||^2 + c,   W = E^H E,  yt = W^{-1} E^H y,  c = ||y||^2 - yt^H W yt
"""
import json
import os
import sys

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "normal_op"))
import common  # noqa: E402

META = json.load(open(os.path.join(common.SCAN, "meta.json")))
RR = META["mean_rr"]
F0 = 42.577e6 * META["b0"] * 1e-6          # Hz per ppm
M_CARD = 16


FAT_PPM = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])     # hasty-qfit FatSpectrum::six_peak
FAT_AMP = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048]) / 0.999
VOX = 0.002
# box centre: signal-weighted centroid of the dynamic (cardiac/vessel) voxels, truth_pc_frames (logical, m from the
# FOV centre); logical = scanner here (rot_presc = I, fov_center = 0)
BOX_CENTRE = np.array([-0.018, 0.0, 0.0])
_RATE = {}


def ratemap_sources(half=0.08, centre=BOX_CENTRE, roi="cube"):
    """signal-weighted ratemap components inside the ROI: positions (m), complex rate z = R + i w (1/s, signal
    e^{-z tau}), weight |amplitude|. Water: R = truth R2*, w = field; fat: the six peaks on the same field and R."""
    key = (half, tuple(centre), roi)
    if key in _RATE:
        return _RATE[key]
    d = common.SCAN
    pw = np.load(os.path.join(d, "truth_pd_water.npy")).astype(np.float64)
    pf = np.load(os.path.join(d, "truth_pd_fat.npy")).astype(np.float64)
    r2 = np.load(os.path.join(d, "truth_r2s.npy")).astype(np.float64)
    fp = np.load(os.path.join(d, "field_ppm_recon.npy")).astype(np.float64)
    dims = np.array(pw.shape)
    g = [(np.arange(n) - (n - 1) / 2) * VOX for n in dims]
    X, Y, Z = np.meshgrid(*g, indexing="ij")
    pos = np.stack([X, Y, Z], -1).reshape(-1, 3)
    rel = (pos - centre[None, :]) / half
    inside = (np.abs(rel).max(1) <= 1) if roi == "cube" else (np.linalg.norm(rel, axis=1) <= 1)
    m = inside & ((pw.ravel() + pf.ravel()) > 0)
    pos, w_, f_, R, om = pos[m], pw.ravel()[m], pf.ravel()[m], r2.ravel()[m], 2 * np.pi * F0 * fp.ravel()[m]
    P = [pos] + [pos] * 6
    Z_ = [R + 1j * om] + [R + 1j * (om + 2 * np.pi * F0 * c) for c in FAT_PPM]
    A = [w_] + [f_ * a for a in FAT_AMP]
    out = (np.concatenate(P), np.concatenate(Z_), np.concatenate(A))
    _RATE[key] = out
    return out


def offres_band(half=0.08, centre=BOX_CENTRE, q=1e-4):
    """signal-weighted band of the ratemap's frequencies in the box: [q, 1-q] weighted quantiles (rad/s), plus the
    weighted R quantile range"""
    _, z, a = ratemap_sources(half, centre)
    def wq(v, qq):
        o = np.argsort(v)
        c = np.cumsum(a[o]) / a.sum()
        return float(v[o][np.searchsorted(c, qq)])
    return wq(z.imag, q), wq(z.imag, 1 - q)


def rate_R_range(half=0.08, centre=BOX_CENTRE, q=1e-4):
    _, z, a = ratemap_sources(half, centre)
    o = np.argsort(z.real)
    c = np.cumsum(a[o]) / a.sum()
    return float(z.real[o][np.searchsorted(c, q)]), float(z.real[o][np.searchsorted(c, 1 - q)])


def load_readouts(ro_list):
    """k (rad/m, physical = logical here), tau (s), conc coefficients w (rad/m^2) [z2, x2+y2, xz, yz], cardiac phase"""
    rec = common.load_any(os.path.join(common.SCAN, "sched_nominal.hv"), lazy=True)
    ro, _ = common.readout_table()
    out = []
    for r in ro_list:
        x = np.asarray(rec[int(r)], dtype=np.float64)                  # [na, 16]
        out.append((x[:, 0:3].copy(), x[:, 13].copy(), x[:, 9:13].copy(), float(ro[r, 4])))
    return out


def cubic_bspline(u):
    """periodic cubic B-spline weights for M_CARD knots at coordinate u (array). returns idx [n,4], w [n,4]"""
    i = np.floor(u).astype(np.int64)
    f = u - i
    w = np.stack([(1 - f) ** 3 / 6, (3 * f ** 3 - 6 * f ** 2 + 4) / 6, (-3 * f ** 3 + 3 * f ** 2 + 3 * f + 1) / 6, f ** 3 / 6], 1)
    idx = (i[:, None] + np.arange(-1, 3)[None, :]) % M_CARD
    return idx, w


def knot_basis(phase, tau, M=M_CARD):
    """dense [n, M] periodic cubic B-spline matrix at the per-sample cardiac coordinate"""
    u = ((phase + tau / RR) * M - 0.5) % M
    idx, w = cubic_bspline(u)
    B = np.zeros((len(u), M))
    np.add.at(B, (np.repeat(np.arange(len(u)), 4), idx.ravel()), w.ravel())
    return B


def mono(r):
    """concomitant monomials [z^2, x^2+y^2, xz, yz] of positions r [P,3]"""
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    return np.stack([z * z, x * x + y * y, x * z, y * z], 1)


def roi_points(rng, P, roi, half, centre, mode="uni"):
    """uni: uniform in the ROI; edge: on the boundary (cube: >= 2 coordinates at +-1 with prob. 0.67 each axis;
    ball: radius 1 with prob. 0.7) -- the band edge, where a local interpolator is worst"""
    if roi == "cube":
        u = rng.uniform(-1, 1, (P, 3))
        if mode == "edge":
            m = rng.random((P, 3)) < 0.67
            u[m] = np.sign(u[m])
    else:  # ball inscribed in the cube
        u = rng.normal(size=(P, 3))
        rad = rng.uniform(0, 1, P) ** (1 / 3)
        if mode == "edge":
            rad = np.where(rng.random(P) < 0.7, 1.0, rad)
        u *= (rad / np.linalg.norm(u, axis=1))[:, None]
    return centre[None, :] + half * u


def draw_sources(rng, P, mode, roi="cube", half=0.08, centre=BOX_CENTRE, wband=None):
    """positions [P,3] and complex rates z [P] (signal e^{-z tau}).
    signal: ratemap components drawn with probability ~ |amplitude| (signal-weighted), jittered inside the voxel;
    uni / edge: positions as roi_points, w uniform in the signal-weighted band (edge: half at the band ends),
    R drawn from the signal-weighted R distribution."""
    pos, z, a = ratemap_sources(half, centre, roi)
    if mode == "signal":
        i = rng.choice(len(a), P, p=a / a.sum())
        r = pos[i] + rng.uniform(-0.5, 0.5, (P, 3)) * VOX
        return r, z[i]
    wb = wband or offres_band(half, centre)
    r = roi_points(rng, P, roi, half, centre, mode)
    om = rng.uniform(*wb, P)
    if mode == "edge":
        om = np.where(rng.random(P) < 0.5, np.where(rng.random(P) < 0.5, wb[0], wb[1]), om)
    R = z.real[rng.choice(len(a), P, p=a / a.sum())]
    return r, R + 1j * om


def signatures(rd, r, z, scale=1.0, knots=True, conc=True, rng=None):
    """[n, P] readout signatures of point sources (r, z): beta_b(u_j) e^{-i k_j.r - z tau_j - i c_j.mono(r)}"""
    k, tau, w, phase = rd
    ph = -1j * (scale * (k @ r.T)) - tau[:, None] * z[None, :]
    if conc:
        ph = ph - 1j * scale ** 2 * (w @ mono(r).T)
    V = np.exp(ph)
    if knots:
        B = knot_basis(phase, tau)
        act = np.nonzero(B.max(0) > 0)[0]
        b = (rng or np.random.default_rng(0)).choice(act, len(z))
        V *= B[:, b]
    return V


def columns(rd, rng, P, roi="cube", half=0.08, centre=BOX_CENTRE, scale=1.0, wband=None, knots=True, conc=True, mode="uni"):
    """[n, P] complex point-source signatures of one readout (see module doc). scale: k multiplier (1 mm proxy = 2;
    the concomitant coefficients then scale with G^2 = scale^2)."""
    r, z = draw_sources(rng, P, mode, roi, half, centre, wband)
    return signatures(rd, r, z, scale, knots, conc, rng)


def train_columns(rd, seed, P, roi, half, scale, wband=None, centre=BOX_CENTRE):
    """training sketch for the local interpolator: half uniform, half on the band edge"""
    return np.concatenate([columns(rd, np.random.default_rng(seed), P // 2, roi, half, centre, scale, wband, mode="uni"),
                           columns(rd, np.random.default_rng(seed + 1), P - P // 2, roi, half, centre, scale, wband, mode="edge")], 1)


def band(rd, roi="cube", half=0.08, scale=1.0, wband=None):
    """instantaneous two-sided band (rad/s) of the ROI signal per sample, and the dwell"""
    k, tau, w, _ = rd
    dt = np.diff(tau)
    kd = np.diff(k, axis=0) / dt[:, None]
    kd = np.concatenate([kd[:1], 0.5 * (kd[1:] + kd[:-1]), kd[-1:]], 0) * scale
    if roi == "cube":
        B = 2 * (np.abs(kd) * half).sum(1)
    else:
        B = 2 * np.linalg.norm(kd, axis=1) * half
    lo, hi = wband or offres_band()
    return B + (hi - lo), float(np.median(dt))


def select_warped(B, dwell, os_, mmax=3, keep_ends=64):
    """subset of sample indices uniform in warped time s(t) = int B dt / 2pi, os_ samples per unit (Nyquist = 1);
    at most mmax original samples between kept ones, and every sample in the first / last keep_ends (the gradient
    ramps at the readout ends, where k barely moves and the instantaneous-band estimate is poor)"""
    B = np.maximum(B, 2 * np.pi / (dwell * mmax * os_))
    s = np.concatenate([[0.0], np.cumsum(B[1:] * dwell / (2 * np.pi))])
    targets = np.arange(0.0, s[-1], 1.0 / os_)
    n = len(B)
    ends = np.r_[np.arange(min(keep_ends, n)), np.arange(max(0, n - keep_ends), n)]
    idx = np.unique(np.concatenate([np.searchsorted(s, targets), [n - 1], ends]))
    return idx[idx < n]


def local_E(Vtr, S, q, ridge=1e-13, chunk=128):
    """sparse local interpolation: row j from the q subset samples nearest to j (by sample index), LS-fitted on the
    training sketch Vtr [n, P]. returns cols [n, q] (positions in S) and coef [n, q]"""
    n = Vtr.shape[0]
    q = min(q, len(S))
    pos = np.searchsorted(S, np.arange(n))                 # first subset index >= j
    start = np.clip(pos - q // 2, 0, len(S) - q)
    cols = start[:, None] + np.arange(q)[None, :]
    coef = np.zeros((n, q), complex)
    VS = Vtr[S]                                            # [|S|, P]
    for j0 in range(0, n, chunk):
        jj = np.arange(j0, min(n, j0 + chunk))
        X = VS[cols[jj]]                                   # [c, q, P]
        Xc = X.conj()
        G = Xc @ X.transpose(0, 2, 1)
        tr = np.einsum("cqq->c", G).real
        G += (ridge * tr)[:, None, None] * np.eye(q)[None]
        rhs = (Xc @ Vtr[jj][:, :, None])[..., 0]
        coef[jj] = np.linalg.solve(G, rhs[..., None])[..., 0]
    # kept samples: exact identity
    where = np.full(n, -1)
    where[S] = np.arange(len(S))
    for j in S:
        c = np.zeros(q, complex)
        c[np.nonzero(cols[j] == where[j])[0][0]] = 1
        coef[j] = c
    return cols, coef


def apply_E(cols, coef, VS):
    """E @ VS, VS = values at the subset [|S|, P]"""
    return np.einsum("jq,jqp->jp", coef, VS[cols])


def E_dense(cols, coef, nS):
    n = cols.shape[0]
    E = np.zeros((n, nS), complex)
    np.add.at(E, (np.repeat(np.arange(n), cols.shape[1]), cols.ravel()), coef.ravel())
    return E


def rel_err(V, Vhat):
    num = np.linalg.norm(V - Vhat, axis=0)
    den = np.linalg.norm(V, axis=0)
    return float(np.linalg.norm(V - Vhat) / np.linalg.norm(V)), float((num / den).max())


def upsample(rd, f=2):
    """the same readout sampled f x faster (dwell / f): cubic-spline interpolation of k, tau, w along the ADC index;
    emulates a 1 mm protocol that keeps the readout duration and halves the dwell (60 M samples per encode)"""
    from scipy.interpolate import CubicSpline
    k, tau, w, ph = rd
    n = len(tau)
    s = np.arange(n)
    sf = np.arange((n - 1) * f + 1) / f
    return CubicSpline(s, k, axis=0)(sf), CubicSpline(s, tau)(sf), CubicSpline(s, w, axis=0)(sf), ph
