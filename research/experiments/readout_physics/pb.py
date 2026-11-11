"""Readout-physics error budget on ph0001 (factory_arms, 2 mm, 3 T): common noise-referenced data-space metric.

  R = || (A - A_model) x || / (sigma sqrt(M C))

A = the data model with the physical term, A_model = without it (omission) or with the proposed handling
(residual); x = the realistic image (warm start of the 10-min scan, encode 0, calibrated to the measured data); real
whitened true coil maps; sigma = the per-sample noise per whitened coil. Evaluated exactly (direct sum, f32 on the GPU,
phases in f64) at the real sample positions of NR readouts spread over the scan and a quadrature subset of the ADC
samples (all samples near |k| = 0, every `stride`-th elsewhere, trapezoid weights).

Designs: `current` (data/hasty/seq/flow_thorax, out-and-back, 7.98 ms) and `v2` (flow_thorax_v2, centre-out
9.49 ms recon window), both synthesised from the design waveform (tr_grad, readout part, about the echo) and rotated
with the rotation matrices of the simulated scan's readouts (readouts.npy, R k_base). The synthesis is checked against
sched_nominal.npy of the simulated current scan (check_design()).

Error images are built per (sample, voxel) by term functions `efun(ctx, sl) -> complex64 tensor (n_s, n_vox_chunk)`
(already multiplied by x); the evaluator applies e^{-i k.r}, the coil maps and sums.
"""
import json, os, sys, time
import numpy as np
import torch
from scipy.ndimage import binary_dilation, uniform_filter1d

REPO = os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute')
ROOT = os.path.join(REPO, 'data/hasty/phantom/factory_arms/ph0001')
SCAN = os.path.join(ROOT, 'sim_flow_10min/scan')
DESIGNS = {'current': os.path.join(REPO, 'data/hasty/seq/flow_thorax'), 'v2': os.path.join(REPO, 'data/hasty/seq/flow_thorax_v2')}
GBAR = 42.577478e6
GAM = 2 * np.pi * GBAR
DEV = torch.device('cuda')
FAT_AMP = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048]); FAT_AMP = FAT_AMP / FAT_AMP.sum()
FAT_PPM = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
LABELS = ['air', 'fat', 'skin', 'muscle', 'bone_cortical', 'bone_marrow', 'cartilage', 'liver', 'gallbladder', 'spleen', 'thymus', 'stomach_wall',
          'gi_content', 'pancreas', 'bowel_wall', 'thyroid', 'lung', 'airway', 'myocardium', 'valve', 'blood_LA', 'blood_LV', 'blood_RA', 'blood_RV',
          'blood_artery', 'blood_vein', 'blood_pulm_artery', 'blood_pulm_vein', 'vessel_wall', 'kidney']


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ───────────────────────────────────────────────────────────────────── designs
class Design:
    """Base readout of a seqdesign run at the ADC dwell, about the echo (time 0 = echo = first k = 0 of the readout).
    k (rad/m), m1 = gamma M1 (rad s/m), m2h = gamma M2 / 2 (rad s^2/m), P = int G G^T over the readout (T^2 s/m^2),
    tau = time since excitation (s). Base (unrotated) frame."""

    def __init__(self, name, dwell=2e-6, b0=3.0):
        run = DESIGNS[name]
        s = json.load(open(run + '/summary.json'))
        b, pr = s['best'], s['problem']
        self.name, self.b0, self.te = name, b0, b['te']
        dt = pr['dt']
        g = np.load(run + '/tr_grad.npy')[0]
        i0 = b['i_ro'] + b.get('i_echo', 0)
        i1 = b['i_ro'] + b.get('recon', [[0, b['i_adc']]])[0][1]
        # piecewise-linear gradient between raster points, exact cumulative moments at the dwell times
        tr = (np.arange(len(g)) - i0) * dt
        T = (i1 - i0) * dt
        na = int(round(T / dwell)) + 0
        t = np.arange(na) * dwell
        up = 20
        tf = np.arange(0, (i1 - i0) * up + 1) * (dt / up)
        gf = np.stack([np.interp(tf, tr[i0:i1 + 1], g[i0:i1 + 1, a]) for a in range(3)], 1)
        def cum(f):
            return np.concatenate([np.zeros((1,) + f.shape[1:]), np.cumsum(0.5 * (f[1:] + f[:-1]) * (dt / up), 0)])
        M0 = cum(gf); M1 = cum(gf * tf[:, None]); M2 = cum(gf * tf[:, None] ** 2)
        P = cum(np.einsum('ta,tb->tab', gf, gf))
        it = lambda A: np.stack([np.interp(t, tf, A.reshape(len(tf), -1)[:, c]) for c in range(A.reshape(len(tf), -1).shape[1])], 1).reshape((na,) + A.shape[1:])
        self.t = t
        self.tau = self.te + t
        self.k = GAM * it(M0)
        self.m1 = GAM * it(M1)
        self.m2h = 0.5 * GAM * it(M2)
        self.P = it(P)
        self.na = na
        self.kabs = np.linalg.norm(self.k, axis=1)
        self.kmax = 2 * np.pi * pr['kmax']
        n0 = (self.k ** 2).sum(1)
        ok = n0 > 1e-6 * n0.max()
        self.t1 = np.where(ok, (self.m1 * self.k).sum(1) / np.where(ok, n0, 1), 0.0)
        self.t2 = np.where(ok, (self.m2h * self.k).sum(1) / np.where(ok, n0, 1), 0.0)  # m2h = t2 k + perp (t2 incl. the 1/2)
        self.m1perp = self.m1 - self.t1[:, None] * self.k
        self.m2hperp = self.m2h - self.t2[:, None] * self.k
        # concomitant coefficients of the readout part (monomials z^2, x^2+y^2, xz, yz), base frame -> rotated per readout
        self.wc = GAM / (2 * b0)

    def conc_w(self, R):
        """per-sample concomitant coefficients [z^2, x^2+y^2, xz, yz] (rad/m^2) of the rotated readout (physical frame)"""
        Pp = np.einsum('ab,nbc,dc->nad', R, self.P, R)
        wc = self.wc
        return np.stack([wc * (Pp[:, 0, 0] + Pp[:, 1, 1]), wc * Pp[:, 2, 2] / 4, -wc * Pp[:, 0, 2], -wc * Pp[:, 1, 2]], 1)

    def sample_select(self, kc=150.0, stride=8, near_all=True):
        """quadrature subset of the ADC samples: every sample with |k| < kc (rad/m), every `stride`-th elsewhere, plus the
        last; trapezoid weights in sample units (sum = na)"""
        sel = np.zeros(self.na, bool)
        sel[::stride] = True
        sel[-1] = True
        if near_all:
            sel |= self.kabs < kc
        n = np.flatnonzero(sel)
        w = np.zeros(len(n))
        d = np.diff(n).astype(float)
        w[:-1] += 0.5 * d; w[1:] += 0.5 * d
        w[0] += 0.5; w[-1] += 0.5
        return n, w


def scan_readouts(nr, offset=0):
    """NR readouts of the simulated scan spread over the 10 min: index, encode, frame, cardiac phase and the rotation R
    with k_phys = R k_base (Procrustes of the scan's nominal k onto the current design's base readout; exact to 1e-7)"""
    ro = np.load(os.path.join(SCAN, 'readouts.npy'), mmap_mode='r')
    s = np.load(os.path.join(SCAN, 'sched_nominal.npy'), mmap_mode='r')
    n = ro.shape[0]
    idx = np.linspace(offset, n - 1, nr).round().astype(int)
    tab = np.asarray(ro[idx])
    kb = Design('current').k
    Rs = []
    for r in idx:
        kr = np.asarray(s[r, :, :3]).astype(np.float64)
        U, _, Vt = np.linalg.svd(kr.T @ kb)
        Rs.append(U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt)
    return dict(idx=idx, enc=tab[:, 1].astype(int), frame=tab[:, 3].astype(int), phase=tab[:, 4], R=np.array(Rs), t_exc=tab[:, 0])


def check_design(des, ridx=(0, 1000, 30000)):
    """synthesised k / tau / readout M1 / concomitant coefficients vs the simulated scan's sched_nominal records
    (current design only; rotation by Procrustes on k)"""
    s = np.load(os.path.join(SCAN, 'sched_nominal.npy'), mmap_mode='r')
    out = []
    for r in ridx:
        rec = np.asarray(s[r]).astype(np.float64)
        U, _, Vt = np.linalg.svd(rec[:, :3].T @ des.k)
        R = U @ Vt
        m = min(des.na, rec.shape[0])
        dk = np.abs(des.k[:m] @ R.T - rec[:m, :3]).max() / des.kmax
        # readout M1 about the echo from the record: m1(tau) - m1(TE) - TE k(tau)
        m1r = rec[:m, 3:6] - rec[0, 3:6] - des.te * rec[:m, :3]
        dm1 = np.abs(des.m1[:m] @ R.T - m1r).max() / np.abs(m1r).max()
        dtau = np.abs(des.tau[:m] - rec[:m, 13]).max()
        wr = rec[:m, 9:13] - rec[0, 9:13]
        dw = np.abs(des.conc_w(R)[:m] - wr).max() / np.abs(wr).max()
        out.append(dict(readout=int(r), dk_rel_kmax=float(dk), dm1_rel=float(dm1), dtau=float(dtau), dw_rel=float(dw)))
    return out


# ───────────────────────────────────────────────────────────────────── object
class Obj:
    """voxels of the dilated object support on the 2 mm recon grid, Fortran order (x fastest)"""

    def __init__(self, dil=1, enc=0, verbose=True):
        meta = json.load(open(os.path.join(SCAN, 'meta.json')))
        self.meta = meta
        self.b0 = meta['b0']
        self.dims = tuple(meta['recon_dims'])
        nx, ny, nz = self.dims
        self.res = 0.002
        ld = lambda f: np.load(os.path.join(SCAN, f), mmap_mode='r')
        pdw = np.asarray(ld('truth_pd_water.npy')); pdf = np.asarray(ld('truth_pd_fat.npy'))
        lum = np.asarray(ld('truth_lumen_fraction.npy'))
        sup = binary_dilation((pdw + pdf > 0) | (lum > 0), iterations=dil)
        self.sup3 = sup
        self.idx = np.flatnonzero(sup.ravel(order='F'))
        N = len(self.idx)
        self.N = N
        i = self.idx % nx; j = (self.idx // nx) % ny; k = self.idx // (nx * ny)
        self.ijk = np.stack([i, j, k], 1)
        self.pos = np.stack([(i - nx // 2) * self.res, (j - ny // 2) * self.res, (k - nz // 2) * self.res], 1)
        g = lambda a: np.asarray(a).ravel(order='F')[self.idx]
        self.hz = g(ld('field_ppm_recon.npy')).astype(np.float64) * GBAR * self.b0 * 1e-6
        self.om = 2 * np.pi * self.hz
        r2 = g(ld('truth_r2s.npy')).astype(np.float64)
        self.r2 = np.where(np.isfinite(r2), r2, 0.0)
        self.lumen = g(lum).astype(np.float32)
        self.pdw = g(pdw); self.pdf = g(pdf)
        ffa = ld('truth_fat_fraction.npy')
        self.ff = np.stack([g(ffa[..., f]) for f in range(ffa.shape[3])], 1).astype(np.float32)  # N x 20
        self.ff = np.clip(np.nan_to_num(self.ff), 0, 1)
        xw = np.load(os.path.join(ROOT, 'fix_2mm/llr_s/warm_full.npy'), mmap_mode='r')
        self.x_raw = g(xw[..., enc]).astype(np.complex64)
        acc = np.zeros(self.dims, np.float32)
        for e in range(xw.shape[3]):
            acc += np.abs(np.asarray(xw[..., e]))
        for ax, r in [(0, 4), (1, 4), (2, 1)]:
            for _ in range(3):
                acc = uniform_filter1d(acc, 2 * r + 1, axis=ax, mode='nearest')
        self.mag_s = g(acc) / xw.shape[3]
        psi = np.array(meta['psi'], dtype=np.float64)
        ev, U = np.linalg.eigh(psi)
        self.Wh = (U / np.sqrt(ev)) @ U.T
        sm = ld('smaps_true.npy')
        nc = sm.shape[3]
        Sraw = np.empty((nc, N), np.complex64)
        for c in range(nc):
            Sraw[c] = g(sm[..., c])
        self.S = (self.Wh.astype(np.complex64) @ Sraw)
        del Sraw
        self.nc = nc
        self.cE = (np.abs(self.S) ** 2).sum(0).astype(np.float64)
        self.sigma = meta['noise_sigma']
        self.scale = 1.0
        self.x = self.x_raw.copy()
        # GPU copies
        self.t_pos = torch.tensor(self.pos, dtype=torch.float64, device=DEV)
        self.t_ST = torch.tensor(self.S.T.copy(), dtype=torch.complex64, device=DEV)  # N x C
        self.t_om = torch.tensor(self.om, dtype=torch.float64, device=DEV)
        self.t_r2 = torch.tensor(self.r2, dtype=torch.float32, device=DEV)
        self.set_x(self.x)
        self.vel = None
        if verbose:
            log(f'object: {N} voxels (support dilated {dil}), {nc} coils, sigma {self.sigma:.2f}, lumen>0 {(self.lumen > 0).sum()}')

    def set_x(self, x):
        self.x = x.astype(np.complex64)
        self.t_x = torch.tensor(self.x, dtype=torch.complex64, device=DEV)
        self.wE = np.abs(self.x).astype(np.float64) ** 2 * self.cE
        self.wEs = (self.mag_s.astype(np.float64) * self.scale) ** 2 * self.cE

    def load_vel(self):
        """truth velocity (logical = physical axes here), N x 3 x 20, float32 (voxel-mean incl. tissue)"""
        if self.vel is None:
            va = np.load(os.path.join(SCAN, 'truth_vel_logical.npy'), mmap_mode='r')
            nx, ny, nz = self.dims
            v = np.empty((self.N, 3, va.shape[4]), np.float32)
            for f in range(va.shape[4]):
                for a in range(3):
                    v[:, a, f] = np.asarray(va[:, :, :, a, f]).ravel(order='F')[self.idx]
            self.vel = v
        return self.vel

    def labels2(self, frame=0):
        """majority tissue label of the 2 mm voxel from the 1 mm truth (frame), N"""
        sys.path.insert(0, os.path.join(REPO, 'python/lib'))
        from hastycompute import value_format as vf
        v = vf.open_lazy(os.path.join(SCAN, 'truth_fine/truth4d.hv'))
        L = np.asarray(v['labels'][frame])  # Z, Y, X (1 mm)
        Z, Y, X = L.shape
        Lb = L.reshape(Z // 2, 2, Y // 2, 2, X // 2, 2).transpose(4, 2, 0, 5, 3, 1).reshape(X // 2, Y // 2, Z // 2, 8)  # x, y, z, 8
        cnt = np.stack([(Lb == l).sum(-1) for l in range(len(LABELS))], -1)
        lab = cnt.argmax(-1).astype(np.uint8)
        return lab.ravel(order='F')[self.idx]


# ───────────────────────────────────────────────────────────────────── evaluator
class Ctx:
    """per-readout sample context on the GPU"""
    pass


def make_ctx(des, R, n, w, frame=None, enc=None):
    c = Ctx()
    c.n = n; c.w = w
    c.tau = torch.tensor(des.tau[n], dtype=torch.float64, device=DEV)
    c.t = torch.tensor(des.t[n], dtype=torch.float64, device=DEV)
    c.k = torch.tensor(des.k[n] @ R.T, dtype=torch.float64, device=DEV)
    c.m1 = torch.tensor(des.m1[n] @ R.T, dtype=torch.float64, device=DEV)
    c.m2h = torch.tensor(des.m2h[n] @ R.T, dtype=torch.float64, device=DEV)
    c.m1perp = torch.tensor(des.m1perp[n] @ R.T, dtype=torch.float64, device=DEV)
    c.m2hperp = torch.tensor(des.m2hperp[n] @ R.T, dtype=torch.float64, device=DEV)
    c.t1 = torch.tensor(des.t1[n], dtype=torch.float64, device=DEV)
    c.t2 = torch.tensor(des.t2[n], dtype=torch.float64, device=DEV)
    c.w4 = torch.tensor(des.conc_w(R)[n], dtype=torch.float64, device=DEV)
    c.R = R; c.frame = frame; c.enc = enc; c.des = des
    return c


def polar(ph):
    """e^{-i ph} as complex64 from an f64 phase"""
    ph = torch.remainder(ph, 2 * np.pi).to(torch.float32)
    return torch.polar(torch.ones_like(ph), -ph)


def eval_terms(obj, des, ros, efuns, vox=None, nsel=None, chunk=None, frame_override=None, progress=False, complex_out=False):
    """error energy per (readout, selected sample), coil-summed, for each named efun (dict name -> f(ctx, sl)).
    vox: voxel indices (into obj) where any efun is non-zero (default all). Returns dict name -> (NR x ns) array and
    the selection (n, w)."""
    if nsel is None:
        nsel = des.sample_select()
    n, w = nsel
    vox = np.arange(obj.N) if vox is None else np.asarray(vox)
    ns = len(n)
    if chunk is None:
        chunk = max(4096, int(float(os.environ.get("PB_CHUNK_ELEMS", "1.6e8")) // max(ns, 1)))
    out = {k: np.zeros((len(ros['idx']), ns)) for k in efuns}
    t0 = time.time()
    tv = torch.tensor(vox, device=DEV)
    for r in range(len(ros['idx'])):
        fr = ros['frame'][r] if frame_override is None else frame_override
        ctx = make_ctx(des, ros['R'][r], n, w, frame=fr, enc=ros['enc'][r])
        Y = {k: torch.zeros((ns, obj.nc), dtype=torch.complex64, device=DEV) for k in efuns}
        for c0 in range(0, len(vox), chunk):
            sl = tv[c0:c0 + chunk]
            E = polar(ctx.k @ obj.t_pos[sl].T)  # ns x nv
            ST = obj.t_ST[sl]
            for name, f in efuns.items():
                img = f(ctx, sl)
                if img is None:
                    continue
                Y[name] += (E * img) @ ST
            del E
        for name in efuns:
            if complex_out:
                if out[name].ndim == 2:
                    out[name] = np.zeros((len(ros['idx']), ns, obj.nc), np.complex64)
                out[name][r] = Y[name].cpu().numpy()
            else:
                out[name][r] = (torch.abs(Y[name]) ** 2).sum(1).cpu().numpy()
        if progress and (r % max(1, len(ros['idx']) // 8) == 0):
            log(f'  readout {r + 1}/{len(ros["idx"])} ({time.time() - t0:.1f}s)')
    return out, (n, w)


def R_of(obj, E, nsel, des, scale_e=1.0, mask_n=None):
    """R = sqrt(sum_n w_n mean_r E / (na C sigma^2)); E in data units^2 (x calibrated). mask_n: restrict the samples
    (still normalised by all na samples, so the parts add in R^2)"""
    n, w = nsel
    e = E.mean(0) * scale_e
    if mask_n is not None:
        e = e * mask_n
    return float(np.sqrt((w * e).sum() / (des.na * obj.nc * obj.sigma ** 2)))


def R_split(obj, E, nsel, des):
    """R and its split over |k| bands (R^2 adds): < 0.1, 0.1-0.5, 0.5-0.9, >= 0.9 kmax"""
    q = des.kabs[nsel[0]] / des.kmax
    bands = [(0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 9)]
    return dict(R=R_of(obj, E, nsel, des), **{f'R_k{a:.1f}-{b:.1f}': R_of(obj, E, nsel, des, mask_n=((q >= a) & (q < b)).astype(float)) for a, b in bands})


# ───────────────────────────────────────────────────────────────────── physics helpers (torch)
def fat_c(tau, b0=3.0):
    """six-peak fat spectrum c(tau) = sum_p a_p e^{-i w_p tau}, complex64 (ns,)"""
    om = 2 * np.pi * GBAR * b0 * FAT_PPM * 1e-6
    tt = tau[:, None] * torch.tensor(om, dtype=torch.float64, device=DEV)[None, :]
    z = (torch.tensor(FAT_AMP, dtype=torch.float64, device=DEV)[None, :] * torch.exp(-1j * tt)).sum(1)
    return z.to(torch.complex64)


def monos(pos):
    """[z^2, x^2+y^2, xz, yz] at positions (nv x 3, f64)"""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return torch.stack([z * z, x * x + y * y, x * z, y * z], 1)


# ───────────────────────────────────────────────────────────────────── truth-based images and effective decay
OUT = os.path.join(ROOT, 'physics_budget_2026-11-10')


def use_truth(obj, design='current', blood_gain=1.0):
    """truth-based band-limited images (truth_image.py) on the object voxels: water, fat (six-peak), blood"""
    z = np.load(os.path.join(OUT, f'truth_x_{design}.npz'))
    g = lambda a: a.ravel(order='F')[obj.idx].astype(np.complex64)
    obj.xw, obj.xf, obj.xb = g(z['water']), g(z['fat']), g(z['blood']) * blood_gain
    obj.t_xw = torch.tensor(obj.xw, device=DEV); obj.t_xf = torch.tensor(obj.xf, device=DEV); obj.t_xb = torch.tensor(obj.xb, device=DEV)
    obj.x_tot = obj.xw + obj.xf + obj.xb
    obj.set_x(obj.x_tot)
    obj.truth_design = design
    return obj


def geff_tables(obj, tr, dt=2.5e-4, tmax=12.5e-3, b0=3.0):
    """per object voxel, from the tissue sources (static pose, in slab): amplitude A_v = sum a_s and the normalised
    effective signal H(v, t) = sum_s a_s e^{-t/T2*_s} e^{-i (w_s - wbar_v) t} / A_v relative to the voxel's model
    field wbar_v (field_ppm_recon), t = 0 .. tmax since excitation; water and fat separately. Also the
    amplitude-weighted mean source field and the PD-weighted table R2* of the same sources.
    Returns dict species -> (t, H (nt x N, complex64), A, wmean (rad/s), r2table)."""
    sys.path.insert(0, os.path.join(REPO, 'python/lib'))
    from hastycompute import value_format as vf
    v = vf.open_lazy(os.path.join(ROOT, 'sim_flow_10min/sim_input.hv'))
    presc = v['presc']; c = np.array(presc['center']); res = float(presc['res']); slab = presc['slab']
    t = v['tissue']
    pos = t['pos'][:].astype(np.float64)
    pd = t['pd'][:].astype(np.float64); t1 = t['t1'][:].astype(np.float64); sh = t['shift_ppm'][:]
    t2s = t['t2s'][:].astype(np.float64); fppm = t['field_ppm'][:].astype(np.float64)
    inslab = np.abs((pos - np.array(slab['center'])) @ np.array(slab['normal'])) <= 0.5 * slab['thk']
    dims = obj.dims
    q2 = np.rint((pos - c) / res + np.array([d // 2 for d in dims])).astype(np.int64)
    ok = np.all((q2 >= 0) & (q2 < np.array(dims)), axis=1) & inslab
    qq = q2[ok, 0] + dims[0] * (q2[ok, 1] + dims[1] * q2[ok, 2])
    li = np.searchsorted(obj.idx, qq)
    inobj = (li < obj.N) & (obj.idx[np.minimum(li, obj.N - 1)] == qq)
    e1 = np.exp(-tr / np.maximum(t1[ok], 1e-4)); fl = np.deg2rad(12.0)
    a = pd[ok] * np.sin(fl) * (1 - e1) / (1 - np.cos(fl) * e1)
    om_s = 2 * np.pi * GBAR * b0 * 1e-6 * fppm[ok]
    r2s = 1.0 / np.maximum(t2s[ok], 1e-5)
    fat = sh[ok] != 0
    ts = np.arange(0, tmax + 1e-12, dt)
    out = {}
    for nm, sel in [('water', ~fat & inobj), ('fat', fat & inobj)]:
        ii = torch.tensor(li[sel], device=DEV)
        aa = torch.tensor(a[sel], dtype=torch.float64, device=DEV)
        dw = torch.tensor(om_s[sel] - obj.om[li[sel]], dtype=torch.float64, device=DEV)
        rr = torch.tensor(r2s[sel], dtype=torch.float64, device=DEV)
        A = torch.zeros(obj.N, dtype=torch.float64, device=DEV).index_add_(0, ii, aa)
        H = torch.zeros((len(ts), obj.N), dtype=torch.complex64, device=DEV)
        for it, tt in enumerate(ts):
            z = (aa * torch.exp(-rr * tt)) * torch.exp(-1j * dw * tt)
            H[it] = (torch.zeros(obj.N, dtype=torch.complex128, device=DEV).index_add_(0, ii, z) / torch.clamp(A, min=1e-30)).to(torch.complex64)
        wm = torch.zeros(obj.N, dtype=torch.float64, device=DEV).index_add_(0, ii, aa * torch.tensor(om_s[sel], dtype=torch.float64, device=DEV)) / torch.clamp(A, min=1e-30)
        pdw = torch.tensor(pd[ok][sel], dtype=torch.float64, device=DEV)
        P = torch.zeros(obj.N, dtype=torch.float64, device=DEV).index_add_(0, ii, pdw)
        r2t = torch.zeros(obj.N, dtype=torch.float64, device=DEV).index_add_(0, ii, pdw * rr) / torch.clamp(P, min=1e-30)
        H[:, (A == 0)] = 1.0
        out[nm] = (ts, H, A.cpu().numpy(), wm.cpu().numpy(), r2t.cpu().numpy())
        log(f'g_eff {nm}: {int(sel.sum())} sources -> {int((A > 0).sum().item())} voxels')
    return out


def interp_t(table_t, H, tau):
    """linear interpolation of H (nt x nv, torch) at times tau (ns, torch f64) -> ns x nv"""
    dt = float(table_t[1] - table_t[0])
    u = torch.clamp(tau / dt, 0, len(table_t) - 1.000001)
    i0 = torch.floor(u).long(); fr = (u - i0).to(torch.float32)[:, None]
    return H[i0] * (1 - fr) + H[i0 + 1] * fr
