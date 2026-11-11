"""R of residual trajectories from the Gram matrices (rgram.py), with projections onto what the reconstruction can
absorb or fit:
  raw      : R as in the budget (fixed image x);
  affine   : minus the best global k -> A k + b, phase c (the same for every readout): an affine map of the image
             coordinates and a global linear / constant phase, i.e. absorbed by x exactly (continuous image model);
  enc      : affine + per-encode k offset b_e and phase c_e (the per-encode linear background-phase polynomial of the
             velocity model, coherent over rotations);
  field    : enc + a degree-1 field error over the readout (dk = c t, dphi = w t: a constant gradient offset is a
             linear B0 field), i.e. the B0 map's degree <= 1 part refitted from the flow data;
  refined  : field + recon-side GIRF refinement directions (default: per-axis delay of the played gradient, per-axis
             B0-term (k0) amplitude), fitted from the scan's own data.
Fisher information / CRLB of the refinement parameters for the whole 10-min scan from the same quadratic form.
"""
import os
import numpy as np
import v2seq

OUT = v2seq.os.path.join(v2seq.REPO, 'data/hasty/seq/studies/girf_sensitivity_2026-11-11')
GAM = v2seq.GAM


class RCalc:
    def __init__(self, fn=None, v=None, rots=None):
        z = np.load(fn or os.path.join(OUT, 'gram_v2.npz'))
        self.n, self.w, self.isrec = z['n'], z['w'], z['isrec']
        self.G, self.Gr = z['G'], z['Gr']
        self.na, self.nc, self.sigma = int(z['na']), int(z['nc']), float(z['sigma'])
        self.N, self.N_reg = int(z['N']), int(z['N_reg'])
        self.v = v or v2seq.V2()
        self.rots = rots or v2seq.scan_rotations(self.G.shape[0], 7, 10)
        self.nr = self.G.shape[0]
        self.enc = self.rots['enc'][:, -1]
        self.Rs = self.rots['R'][:, -1]
        self.norm = 1.0 / (self.na * self.nc * self.sigma ** 2)
        # sample weights folded with the readout mean, recon samples only
        self.ww = np.where(self.isrec, self.w, 0.0)[None, :] / self.nr  # 1 x ns
        self.wnav = np.where(~self.isrec, self.w, 0.0)[None, :] / self.nr
        self.k = np.stack([self.v.k_base[self.n] @ R.T for R in self.Rs])  # nr x ns x 3 (rad/m)
        g = np.stack([np.interp(self.v.t[self.n], np.arange(len(self.v.base)) * self.v.dt, self.v.base[:, c]) for c in range(3)], 1)
        self.g = np.stack([g @ R.T for R in self.Rs])  # nominal physical gradient at the samples (T/m)
        self.ts = self.v.t[self.n]

    def q_of(self, K, P):
        """residual (nr x na_all x 3 rad/m, nr x na_all rad) -> q at the selected samples (nr x ns x 4)"""
        return np.concatenate([K[:, self.n], P[:, self.n, None]], -1)

    def quad(self, q, region=False, nav=False):
        G = self.Gr if region else self.G
        w = self.wnav if nav else self.ww
        return float((w * np.einsum('rna,rnab,rnb->rn', q, G, q)).sum() * self.norm)

    def R(self, q, region=False):
        r = np.sqrt(max(self.quad(q, region), 0.0))
        if region:
            r *= np.sqrt(self.N / self.N_reg)
        return r

    # ── nuisance / refinement directions (each nr x ns x 4)
    def dirs_affine(self):
        d = []
        for a in range(3):
            for b in range(3):
                x = np.zeros(self.k.shape[:2] + (4,)); x[..., a] = self.k[..., b]; d.append(x)
        for a in range(3):
            x = np.zeros(self.k.shape[:2] + (4,)); x[..., a] = 1.0; d.append(x)
        x = np.zeros(self.k.shape[:2] + (4,)); x[..., 3] = 1.0; d.append(x)
        return d

    def dirs_enc(self):
        d = self.dirs_affine()[:9]
        for e in range(7):
            m = (self.enc == e).astype(float)[:, None]
            for a in range(4):
                x = np.zeros(self.k.shape[:2] + (4,)); x[..., a] = m; d.append(x)
        return d

    def dirs_field(self):
        """degree-1 field error over the readout (= a constant gradient offset): dk = c t (3), dphi = w t (1)"""
        d = []
        for a in range(3):
            x = np.zeros(self.k.shape[:2] + (4,)); x[..., a] = self.ts[None, :]; d.append(x)
        x = np.zeros(self.k.shape[:2] + (4,)); x[..., 3] = self.ts[None, :]; d.append(x)
        return d

    def dirs_delay(self):
        """per-axis delay of the played gradient: dk_a = -gamma delta_a g_a(t) (rad/m per s)"""
        d = []
        for a in range(3):
            x = np.zeros(self.k.shape[:2] + (4,)); x[..., a] = -GAM * self.g[..., a]; d.append(x)
        return d

    def project(self, q, dirs, region=False, ridge=1e-12, return_theta=False, nav=False):
        """q minus its best combination of dirs in the R metric"""
        G = self.Gr if region else self.G
        w = self.wnav if nav else self.ww
        J = np.stack(dirs, -1)  # nr x ns x 4 x p
        GJ = np.einsum('rnab,rnbp->rnap', G, J)
        A = np.einsum('rn,rnap,rnaq->pq', w, J, GJ)
        b = np.einsum('rn,rnap,rna->p', w, GJ, q)
        sc = np.sqrt(np.maximum(np.diag(A), 1e-300))
        th = np.linalg.solve(A / np.outer(sc, sc) + ridge * np.eye(len(sc)), b / sc) / sc
        r = q - J @ th
        return (r, th) if return_theta else r

    def Rset(self, q, refine=None, region=True):
        """dict of R: raw / affine / enc / refined (+ local versions)"""
        out = dict(raw=self.R(q))
        out['affine'] = self.R(self.project(q, self.dirs_affine()))
        de = self.dirs_enc()
        out['enc'] = self.R(self.project(q, de))
        df = de + self.dirs_field()
        out['field'] = self.R(self.project(q, df))
        dr = df + (self.dirs_delay() if refine is None else refine)
        out['refined'] = self.R(self.project(q, dr))
        if region:
            out['loc_raw'] = self.R(q, True)
            out['loc_enc'] = self.R(self.project(q, de, True), True)
            out['loc_refined'] = self.R(self.project(q, dr, True), True)
        return out

    def fisher(self, dirs, n_tr=None, nav=False):
        """Fisher information of the direction coefficients for the whole scan (n_tr TRs; default 10 min), complex
        Gaussian noise CN(0, sigma^2) per sample and whitened coil: F = (2 / sigma^2) sum Re(dq^H D^H D dq)"""
        if n_tr is None:
            n_tr = int(600 / self.v.tr)
        J = np.stack(dirs, -1)
        G = self.G
        w = self.wnav if nav else self.ww
        A = np.einsum('rn,rnap,rnab,rnbq->pq', w, J, G, J)  # = mean over readouts of sum over samples
        return 2.0 / self.sigma ** 2 * A * n_tr
