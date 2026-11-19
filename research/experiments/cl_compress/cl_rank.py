"""Joint coil x time-segment compression of the segmented NUFFT's spatial diagonals: rank study (CPU).

Model per (virtual coil c, ADC time tau, voxel r):  E_c(r, tau) = S_c(r) g(r, tau),
  g = exp(-i psi), psi = omega(r) tau [+ sum_q wbar_q(tau) m_q(r)]  [x exp(-R2*(tau - tau0))]  [x ((1 - ff) + ff c(tau))].
Any compressed operator of the form  E_c(r, tau) ~ sum_q B_cq(tau) Phi_q(r)  (R FFT channels Phi_q x, per-sample mixing
B) is a choice of an R-dim subspace of the (c, tau) space; its weighted error is the trace of the Gram outside it.

Exact reduction (no approximation beyond a 1e-13 tail): on a time grid ts (ns points) the coil-common Gram
G_tau = sum_r w_r sum_c |S_c|^2 g g^H gives an orthonormal time basis V (ns x K) with g_r ~ V h_r, h_r = V^H g_r.
Then with z_r = vec_(c,k)(S_c(r) h_k(r)) and time weights Tw (quadrature x weight kind), Mh = (V^H Tw V)^1/2:
  weighted error energy of z -> z~  =  sum_r w_r || (I_C kron Mh)(z_r - z~_r) ||^2,   Ht = sum_r w_r z'_r z'_r^H.
All variants are evaluated from Ht (C K x C K):
  seg_svd  coil-common optimal segments (time subspace I_C kron V_L): channels C * L
  fixed    today's fixed-time diagonals, least-squares interpolators (count weights on the coil-energy fit set, as
           hasty_nufft::build_segments; and energy-weighted): channels C * L   [on the ns grid directly]
  (a)      SVD of the C L diagonals D_cl = S_c P_l of fixed segments, model sum_l a_l(tau) sum_q M_clq Phi_q
  (b)      Tucker product (HOSVD coil basis x time basis): channels Rc * Rt; and the block-diagonal core
           (coil-adaptive segment counts, own time basis per virtual coil of the region): channels sum_s L_s
  (c)      spatial blocks (2^3, 4^3 of the region's bounding box): channels sum_b R_b
  (d)      joint SVD of E over (c, tau) ("coil-aware segmentation"): channels R
Physics (--phys): ratemap = one complex rate per voxel (off-resonance + decay, truth_r2s) x the six-peak fat
spectrum folded with truth_ff, + the encode-0 mean concomitant phase; conc = off-resonance + concomitant only.
Voxel weights: 'e' = proton-density energy (truth_pd_water + truth_pd_fat)^2 on the signal support (PD > 0.01),
'u' = 1 on that support (exact ||dA||_F / ||A||_F for images supported there). Time weights per ADC sample: 'u' 1, 'd' data energy E(n),
'm' 1 + (E(n) + E_vol)/(C sigma^2) (noise-referenced, research/segment_basis.md section 2).
Hold-out: voxel parity (subspace from even voxels, error on odd).

Run (heavy: wrap in flock + systemd-run, see README in the note):
  rustlib/.venv/bin/python -I research/experiments/cl_compress/cl_rank.py --region object|box --phys ratemap|conc
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter1d

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "python/lib/hastycompute").is_dir())
sys.path.insert(0, str(ROOT / "python/lib"))
from hastycompute.store import load_any  # noqa: E402

PH = ROOT / "data/hasty/phantom/factory_arms/ph0001"
S = PH / "sim_flow_10min/scan"
GBAR = 42.577478e6
EPS = [1e-2, 1e-3, 1e-4]
TK = ["u", "d", "m"]
FAT_A = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048]) / 1.0
FAT_A = FAT_A / FAT_A.sum()
FAT_PPM = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def seg_time_grid_idx(w, ns):
    """ADC indices of hasty_nufft::seg_time_grid (a third uniform, the rest equidistributing sqrt w)"""
    na = len(w)
    nu = max(ns // 3, 2)
    idx = [int(round((na - 1) * i / (nu - 1))) for i in range(nu)]
    cum = np.cumsum(np.sqrt(np.maximum(w, 0)))
    nw = max(ns - nu, 1)
    idx += list(np.searchsorted(cum, cum[-1] * (np.arange(nw) + 0.5) / nw))
    return np.unique(np.clip(idx, 0, na - 1))


def vol_energy(E, kap):
    dV = 4 * np.pi * kap ** 2 * np.abs(np.gradient(kap))
    return (E * dV).sum() / dV.sum()


class Setup:
    def __init__(self, region, phys, nseg_nodes=range(6, 25), stride_obj=1):
        pr = np.load(HERE / "prep.npz")
        self.pr = pr
        self.tau = pr["tau"]
        na = len(self.tau)
        self.t0, self.t1 = self.tau[0], self.tau[-1]
        self.b0 = float(pr["b0"])
        dims = tuple(int(d) for d in pr["dims"])
        res = float(pr["res"])
        nc_v = 30
        sig = float(pr["sigma"])
        # time weights per ADC sample
        E = pr["tw_E"]
        self.twa = {"u": np.ones(na), "d": E / E.mean(), "m": 1.0 + (E + vol_energy(E, pr["kap"])) / (32 * sig ** 2)}
        self.mean_m = float(self.twa["m"].mean())
        # time grid: seg_time_grid (mix and data weights, 96 each) + 160 uniform ADC indices + exact fixed-node times
        ii = np.unique(np.concatenate([seg_time_grid_idx(self.twa["m"], 96), seg_time_grid_idx(E, 96), np.linspace(0, na - 1, 160).round().astype(int)]))
        ts = list(self.tau[ii])
        self.nodes = {}
        for L in nseg_nodes:
            tl = np.linspace(self.t0, self.t1, L)
            self.nodes[L] = tl
            ts += list(tl)
        ts = np.unique(np.round(np.array(ts), 12))
        self.ts = ts
        self.ns = len(ts)
        self.node_idx = {L: np.array([int(np.argmin(np.abs(ts - t))) for t in tl]) for L, tl in self.nodes.items()}
        # quadrature: every ADC sample to its nearest grid time
        nearest = np.clip(np.searchsorted((ts[1:] + ts[:-1]) / 2, self.tau), 0, self.ns - 1)
        self.tw = {k: np.bincount(nearest, weights=v, minlength=self.ns) for k, v in self.twa.items()}
        # ------------------------------------------------------------ voxels
        pw = np.load(S / "truth_pd_water.npy")
        pf = np.load(S / "truth_pd_fat.npy")
        pd = pw.astype(np.float64) + pf
        supp = pd > 0.01  # signal support: voxels without signal carry no weight
        lum = np.load(S / "truth_lumen_fraction.npy")
        cen = np.argwhere(lum > 0.5).mean(0)
        box = np.zeros(dims, bool)
        if region == "box":
            n = int(round(0.160 / res))
            lo = [int(np.clip(round(cen[a] - n / 2), 0, dims[a] - n)) for a in range(3)]
            box[lo[0]:lo[0] + n, lo[1]:lo[1] + n, lo[2]:lo[2] + n] = True
            self.box_lo = lo
        else:
            box[:] = True
            self.box_lo = [0, 0, 0]
        self.boxmask = box
        mask = supp & box
        ijk = np.argwhere(mask)
        if stride_obj > 1:
            ijk = ijk[::stride_obj]
        self.ijk = ijk
        self.nv = len(ijk)
        self.pos = (ijk - np.array(dims)[None, :] // 2) * res  # m, scanner = logical (rot identity, centre 0)
        sel = tuple(ijk.T)
        self.we = pd[sel] ** 2  # signal (proton-density) energy
        self.we /= self.we.mean()
        f = np.load(S / "field_ppm_recon.npy")
        self.om = 2 * np.pi * GBAR * self.b0 * 1e-6 * f[sel].astype(np.float64)
        r2 = np.load(S / "truth_r2s.npy")[sel].astype(np.float64)
        self.r2 = np.where(np.isfinite(r2), r2, 0.0)
        ffn = np.load(S / "truth_ff.npy")
        self.ff = np.clip(ffn[sel].astype(np.float64), 0, 1)
        x, y, z = self.pos.T
        self.mono = np.stack([z * z, x * x + y * y, x * z, y * z], 1)
        self.wbar = pr["wbar"]
        self.phys = phys
        # coils: whitened, SV-compressed to the 30 strongest eigenvectors of the (noisy) data Gram
        Wh = pr["Wh"]
        Vc = pr["cc_clean"][:, :nc_v]
        self.T = (Vc.conj().T @ Wh)  # 30 x 32: virtual = T @ raw
        sm = load_any(S / "smaps_true.hv", lazy=True)
        smr = np.asarray(sm[...])  # (nx, ny, nz, 32) c64
        self.Sv = (smr[sel].astype(np.complex128) @ self.T.T)  # nv x 30
        self.C = nc_v
        self.cE = (np.abs(self.Sv) ** 2).sum(1)
        # fit set of build_segments: coil energy > 1e-3 max over the region's grid box, every 13th voxel
        bijk = np.argwhere(box)
        cEb = np.empty(len(bijk))
        for c0 in range(0, len(bijk), 1 << 20):
            q = tuple(bijk[c0:c0 + (1 << 20)].T)
            cEb[c0:c0 + (1 << 20)] = (np.abs(smr[q].astype(np.complex128) @ self.T.T) ** 2).sum(1)
        fit = bijk[cEb > 1e-3 * cEb.max()][::13]
        fq = tuple(fit.T)
        self.fit_om = 2 * np.pi * GBAR * self.b0 * 1e-6 * f[fq].astype(np.float64)
        xf, yf, zf = ((fit - np.array(dims)[None, :] // 2) * res).T
        self.fit_mono = np.stack([zf * zf, xf * xf + yf * yf, xf * zf, yf * zf], 1)
        r2f = np.load(S / "truth_r2s.npy")[fq].astype(np.float64)
        self.fit_r2 = np.where(np.isfinite(r2f), r2f, 0.0)
        self.fit_ff = np.clip(np.nan_to_num(ffn[fq].astype(np.float64)), 0, 1)
        del smr
        # blocks: 4x4x4 of the region's bounding box
        lo, hi = ijk.min(0), ijk.max(0) + 1
        b4 = np.minimum(((ijk - lo) * 4) // (hi - lo), 3)
        self.blk4 = b4[:, 0] + 4 * (b4[:, 1] + 4 * b4[:, 2])
        self.region = region
        log(f"setup {region}/{phys}: {self.nv} voxels, ns {self.ns} grid times, fit set {len(fit)}, box lo {self.box_lo}, lumen centroid {cen.round(1)}")

    def g(self, sl, ts, fit=False):
        """time function g (nvox x len(ts)), complex128, for voxel slice sl (or the fit set)"""
        om = self.fit_om if fit else self.om[sl]
        mono = self.fit_mono if fit else self.mono[sl]
        ph = om[:, None] * ts[None, :]
        if self.phys in ("conc", "r2s", "fat", "ratemap"):
            wb = np.stack([np.interp(ts, self.tau, self.wbar[:, q]) for q in range(4)], 1)  # ns x 4
            ph = ph + mono @ wb.T
        z = np.exp(-1j * ph)
        if self.phys in ("r2s", "fat", "ratemap"):
            r2 = self.fit_r2 if fit else self.r2[sl]
            z *= np.exp(-r2[:, None] * (ts[None, :] - self.t0))
        if self.phys in ("fat", "ratemap"):
            ff = self.fit_ff if fit else self.ff[sl]
            c = (FAT_A[None, :] * np.exp(-1j * 2 * np.pi * GBAR * self.b0 * 1e-6 * FAT_PPM[None, :] * ts[:, None])).sum(1)
            z *= (1 - ff)[:, None] + ff[:, None] * c[None, :]
        return z


def chunks(n, m):
    for c0 in range(0, n, m):
        yield slice(c0, min(c0 + m, n))


def time_grams(st):
    """coil-common Grams on the grid: voxel weights e and u (x coil energy), and the fit set (count)"""
    ns = st.ns
    Ge = np.zeros((ns, ns), complex)
    Gu = np.zeros((ns, ns), complex)
    for sl in chunks(st.nv, 16384):
        g = st.g(sl, st.ts)
        a = g * np.sqrt(st.we[sl] * st.cE[sl])[:, None]
        Ge += a.T @ a.conj()
        a = g * np.sqrt(st.cE[sl])[:, None]
        Gu += a.T @ a.conj()
    gf = st.g(None, st.ts, fit=True)
    Gf = gf.T @ gf.conj()
    return Ge, Gu, Gf  # G[i, j] = sum_r w g_i conj(g_j)


def fixed_errors(st, Ge, Gfit, tk):
    """relative weighted error of fixed-time segments (uniform nodes), LS interpolators by Gfit (count) and Ge"""
    tw = st.tw[tk]
    out = {}
    for lab, Gl in [("fixed", Gfit), ("fixed_e", Ge)]:
        r = {}
        for L, N in st.node_idx.items():
            GNN = Gl[np.ix_(N, N)]
            GNN = GNN + 1e-12 * np.trace(GNN).real / L * np.eye(L)
            A = np.linalg.solve(GNN.T, Gl[:, N].T).T  # ns x L, row i = a(t_i)^T
            err = np.diag(Ge).real - 2 * np.real((A * Ge[N, :].T).sum(1)) + np.real(((A @ Ge[np.ix_(N, N)]) * A.conj()).sum(1))
            r[L] = float(np.sqrt(max((tw * err).sum(), 0) / (tw * np.diag(Ge).real).sum()))
        out[lab] = r
    return out


def interp_matrix(Gfit, N):
    L = len(N)
    GNN = Gfit[np.ix_(N, N)] + 1e-12 * np.trace(Gfit[np.ix_(N, N)]).real / L * np.eye(L)
    return np.linalg.solve(GNN.T, Gfit[:, N].T).T


def sqrtm_h(M):
    e, U = np.linalg.eigh((M + M.conj().T) / 2)
    return (U * np.sqrt(np.maximum(e, 0))) @ U.conj().T


def count_for(tail_rel, eps):
    """smallest R with sqrt(tail(R)) <= eps; tail_rel[R] = relative tail after keeping R"""
    ok = np.flatnonzero(np.sqrt(np.maximum(tail_rel, 0)) <= eps)
    return int(ok[0]) if len(ok) else None


def tails(ev, total):
    ev = np.sort(np.maximum(ev.real, 0))[::-1]
    t = total - np.concatenate([[0.0], np.cumsum(ev)])
    return t / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="box")
    ap.add_argument("--phys", default="conc")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ktol", type=float, default=1e-13)
    a = ap.parse_args()
    t_start = time.time()
    st = Setup(a.region, a.phys, stride_obj=a.stride)
    C = st.C
    tim = {}
    t_ = time.time()
    Ge, Gu, Gf = time_grams(st)
    tim["time_grams_s"] = time.time() - t_
    log("time Grams done")
    # time basis V from Ge + Gu (normalised), K by the worst residual over voxel x time weightings
    Gs = Ge / np.trace(Ge).real + Gu / np.trace(Gu).real
    ev, Vall = np.linalg.eigh(Gs)
    Vall = Vall[:, ::-1]
    resid = {}
    K = None
    for k in range(4, st.ns):
        V = Vall[:, :k]
        P = np.eye(st.ns) - V @ V.conj().T
        worst = 0.0
        for G in (Ge, Gu):
            R = P @ G @ P.conj().T
            for tk in TK:
                tw = st.tw[tk]
                worst = max(worst, (tw * np.diag(R).real).sum() / (tw * np.diag(G).real).sum())
        if worst <= a.ktol or k == min(st.ns - 1, 120):
            K = k
            resid = worst
            break
    V = Vall[:, :K]
    log(f"time basis K = {K} (worst relative residual energy {resid:.1e})")
    # reduced Grams H (C K x C K): weights e (total, even, odd, 4^3 blocks) and u (total)
    CK = C * K
    t_ = time.time()
    He_even = np.zeros((CK, CK), complex)
    He_odd = np.zeros((CK, CK), complex)
    Hu = np.zeros((CK, CK), complex)
    Hb = np.zeros((64, CK, CK), complex)
    order = np.argsort(st.blk4, kind="stable")
    bsorted = st.blk4[order]
    for b in range(64):
        idxb = order[np.searchsorted(bsorted, b, "left"):np.searchsorted(bsorted, b, "right")]
        for c0 in range(0, len(idxb), 8192):
            ii = idxb[c0:c0 + 8192]
            g = st.g(ii, st.ts)
            h = g @ V.conj()  # nvox x K  (h = V^H g)
            z = (st.Sv[ii][:, :, None] * h[:, None, :]).reshape(len(ii), CK)  # (c, k)
            ze = z * np.sqrt(st.we[ii])[:, None]
            ev_ = (ii % 2) == 0
            Ge_ = ze[ev_].T @ ze[ev_].conj()
            Go_ = ze[~ev_].T @ ze[~ev_].conj()
            He_even += Ge_
            He_odd += Go_
            Hb[b] += Ge_ + Go_
            Hu += z.T @ z.conj()
    tim["reduced_grams_s"] = time.time() - t_
    log("reduced Grams done")
    He = He_even + He_odd  # H[(c,k),(c',k')] = sum_r w z_ck conj(z_c'k') = sum_r w z z^H
    out = dict(region=a.region, phys=a.phys, weights="pd_energy", nv=st.nv, ns=st.ns, K=K, K_resid=resid, C=C, mean_m=st.mean_m, box_lo=st.box_lo, res={})
    keepQ, keepU = {}, {}
    for vk, H in [("e", He), ("u", Hu)]:
        G_tau = Ge if vk == "e" else Gu
        for tk in TK:
            tw = st.tw[tk]
            Mw = V.conj().T @ (tw[:, None] * V)
            Mh = sqrtm_h(Mw)
            IM = np.kron(np.eye(C), Mh)
            Ht = IM @ H @ IM.conj().T
            Ht = (Ht + Ht.conj().T) / 2
            total = np.trace(Ht).real
            r = {}
            en = 0.2 / 1.5 / np.sqrt(st.mean_m)  # noise rule R_pred = eps sqrt(mean w_m) <= 0.2 with the x1.5 margin
            eps_l = EPS + [3e-3, 3e-4] + ([en] if tk == "m" else [])
            r["eps_noise"] = en if tk == "m" else None
            # (d) joint
            t_e = time.time()
            evd, Qd = np.linalg.eigh(Ht)
            tim.setdefault("eigh_CK_s", time.time() - t_e)
            evd, Qd = evd[::-1], Qd[:, ::-1]
            td = tails(evd, total)
            r["d"] = {str(e): count_for(td, e) for e in eps_l}
            r["d_spec"] = (np.maximum(evd[:400], 0) / total).tolist()
            r["d_tail"] = np.sqrt(np.maximum(td, 0)).tolist()
            if vk == "e":
                Htu = IM @ Hu @ IM.conj().T
                tu = np.trace(Htu).real
                capu = np.concatenate([[0], np.cumsum(np.real(np.einsum("ij,ji->i", Qd.conj().T, Htu @ Qd)))])
                tx = np.maximum(1 - capu / tu, 0)
                r["d_cross_u"] = {str(e): count_for(tx, e) for e in eps_l}
                r["d_cross_u_tail"] = np.sqrt(tx).tolist()
            keepQ[f"{vk}{tk}"] = (Qd, IM)
            # coil-common SVD segments
            H4 = Ht.reshape(C, K, C, K)
            Htc = np.einsum("ckcl->kl", H4)
            evs = np.linalg.eigvalsh(Htc)
            ts_ = tails(evs, total)
            r["seg_svd_L"] = {str(e): count_for(ts_, e) for e in eps_l}
            r["seg_svd_spec"] = (np.sort(np.maximum(evs, 0))[::-1] / total).tolist()
            r["seg_tail"] = np.sqrt(np.maximum(ts_, 0)).tolist()
            # fixed nodes (today) on the grid
            fe = fixed_errors(st, G_tau, Gf, tk)
            r["fixed_err"] = fe["fixed"]
            r["fixed_e_err"] = fe["fixed_e"]
            r["fixed_L"] = {str(e): next((L for L in sorted(fe["fixed"]) if fe["fixed"][L] <= e), None) for e in eps_l}
            r["fixed_e_L"] = {str(e): next((L for L in sorted(fe["fixed_e"]) if fe["fixed_e"][L] <= e), None) for e in eps_l}
            # (b) Tucker HOSVD
            Gc = np.einsum("ckdk->cd", H4)
            ec, Uc = np.linalg.eigh(Gc)
            Uc = Uc[:, ::-1]
            et, Ut = np.linalg.eigh(Htc)
            Ut = Ut[:, ::-1]
            keepU[f"{vk}{tk}"] = Ut
            Wk = np.kron(Uc, Ut)
            dg = np.real(np.einsum("ij,ji->i", Wk.conj().T, Ht @ Wk)).reshape(C, K)
            cs = np.cumsum(np.cumsum(dg, 0), 1)  # captured(Rc, Rt) = cs[Rc-1, Rt-1]
            tuck = {}
            for e in eps_l:
                best = None
                for rc in range(1, C + 1):
                    for rt in range(1, K + 1):
                        if np.sqrt(max(1 - cs[rc - 1, rt - 1] / total, 0)) <= e:
                            if best is None or rc * rt < best[0]:
                                best = (rc * rt, rc, rt)
                            break
                tuck[str(e)] = best
            r["tucker"] = tuck
            # (b') block-diagonal core: own time basis per region virtual coil
            mus = []
            per = []
            for s in range(C):
                u = Uc[:, s]
                Gs_ = np.einsum("c,ckdl,d->kl", u.conj(), H4, u)
                m_ = np.sort(np.maximum(np.linalg.eigvalsh((Gs_ + Gs_.conj().T) / 2), 0))[::-1]
                mus.append(m_)
                per.append(m_)
            allm = np.sort(np.concatenate(mus))[::-1]
            tb = tails(allm, total)
            r["block_coil"] = {str(e): count_for(tb, e) for e in eps_l}
            r["bc_tail"] = np.sqrt(np.maximum(tb, 0)).tolist()
            alloc = {}
            for e in eps_l:
                R_ = count_for(tb, e)
                if R_ is None:
                    continue
                thr = allm[R_ - 1] if R_ > 0 else np.inf
                alloc[str(e)] = [int((m_ >= thr).sum()) for m_ in per]
            r["block_coil_alloc"] = alloc
            # (c) spatial blocks
            for nbk, lab in [(8, "blk8"), (64, "blk64")]:
                evs_all = []
                for b in range(nbk):
                    if nbk == 64:
                        Hb_ = Hb[b]
                    else:
                        ids = [i + 4 * (j + 4 * k) for i in range(4) for j in range(4) for k in range(4) if (i // 2) + 2 * ((j // 2) + 2 * (k // 2)) == b]
                        Hb_ = Hb[ids].sum(0)
                    if vk == "u":
                        continue
                    Htb = IM @ Hb_ @ IM.conj().T
                    evs_all.append(np.linalg.eigvalsh((Htb + Htb.conj().T) / 2))
                if evs_all:
                    tbk = tails(np.concatenate(evs_all), total)
                    r[lab] = {str(e): count_for(tbk, e) for e in eps_l}
            # (a) SVD of D_cl for fixed nodes L (LS interpolators of today's fit), optimal weighted compression.
            # D = (I x VN) z (node rows), model E~ = (I x A) D~, D~ = (I x MAh^-1) P_R (I x MAh) D, MA = A^H Tw A.
            # err^2 = total - 2 Re tr(P M2) + sum_{q<=R} eD_q,  M2 = Tz H B1^H, Tz = I x MAh VN, B1 = I x MAh^-1 A^H Tw V
            a_res = {}
            for L in (10, 12, 16, 20):
                N = st.node_idx[L]
                A = interp_matrix(Gf, N)  # ns x L
                VN = V[N, :]
                # sqrt(Tw) A = Qa Ra (QR): Ra is a square root of MA = A^H Tw A with Y = sqrt(Tw) A Ra^-1 = Qa orthonormal
                Qa, Ra = np.linalg.qr(np.sqrt(tw)[:, None] * A)
                Tz = np.kron(np.eye(C), Ra @ VN)
                B1 = np.kron(np.eye(C), Qa.conj().T @ (np.sqrt(tw)[:, None] * V))
                HD = Tz @ H @ Tz.conj().T
                eD, QD = np.linalg.eigh((HD + HD.conj().T) / 2)
                eD, QD = eD[::-1], QD[:, ::-1]
                M2 = Tz @ H @ B1.conj().T
                dm = np.real(np.einsum("ij,ji->i", QD.conj().T, M2 @ QD))
                err = np.sqrt(np.maximum(total - 2 * np.concatenate([[0], np.cumsum(dm)]) + np.concatenate([[0], np.cumsum(np.maximum(eD, 0))]), 0) / total)
                a_res[L] = {int(R_): float(err[R_]) for R_ in range(0, C * L + 1)}
                r.setdefault("a_check_full_vs_fixed", {})[str(L)] = [float(err[C * L]), r["fixed_err"][L]]
            r["a"] = {str(L): v for L, v in a_res.items()}
            r["a_R"] = {str(L): {str(e): next((R_ for R_ in sorted(v) if v[R_] <= e), None) for e in eps_l} for L, v in a_res.items()}
            r["a"] = {str(L): {str(R_): v[R_] for R_ in range(0, len(v), 5)} for L, v in a_res.items()}
            # hold-out (voxel parity), e weights only
            if vk == "e":
                Hev = IM @ He_even @ IM.conj().T
                Hod = IM @ He_odd @ IM.conj().T
                eo, Qo = np.linalg.eigh((Hev + Hev.conj().T) / 2)
                Qo = Qo[:, ::-1]
                tot_o = np.trace(Hod).real
                ho = {}
                for e in eps_l:
                    R_ = r["d"][str(e)]
                    if R_ is None:
                        continue
                    cap = np.real(np.trace(Qo[:, :R_].conj().T @ Hod @ Qo[:, :R_]))
                    ho[str(e)] = [R_, float(np.sqrt(max(1 - cap / tot_o, 0)))]
                r["holdout_d"] = ho
            out["res"][f"{vk}{tk}"] = r
            log(f"{vk}{tk}: d {r['d']} | segsvd C*L {({k: (C * v if v else None) for k, v in r['seg_svd_L'].items()})} | fixed L {r['fixed_L']} fixed_e L {r['fixed_e_L']} | "
                f"tucker {r['tucker']} | blockcoil {r['block_coil']} | blk8 {r.get('blk8')} blk64 {r.get('blk64')} | a {r['a_R']} | holdout {r.get('holdout_d')}")
    # per-voxel (point-image) relative error rho(r) with uniform time weights: basis 'eu' (image-energy) and 'uu'
    pvox = {}
    for key in ("eu", "uu"):
        Qd, IM = keepQ[key]
        Ut = keepU[key]
        rr = out["res"][key]
        tasks = []
        for e in EPS:
            Rd = rr["d"][str(e)]
            Ls = rr["seg_svd_L"][str(e)]
            if Rd is not None:
                tasks.append((f"d@{e}", Qd[:, :Rd]))
            if Ls is not None:
                tasks.append((f"seg@{e}", np.kron(np.eye(C), Ut[:, :Ls])))
        rho = {t[0]: np.empty(st.nv) for t in tasks}
        wts = np.empty(st.nv)
        for sl in chunks(st.nv, 8192):
            g = st.g(sl, st.ts)
            h = g @ V.conj()
            z = (st.Sv[sl][:, :, None] * h[:, None, :]).reshape(-1, CK) @ IM.T  # rows z'^T
            n2 = (np.abs(z) ** 2).sum(1)
            for lab, Q in tasks:
                p = z @ Q.conj()
                rho[lab][sl] = np.sqrt(np.maximum(n2 - (np.abs(p) ** 2).sum(1), 0) / n2)
        qs = [50, 90, 99, 99.9, 100]
        pvox[key] = {lab: {"pct": dict(zip([str(q) for q in qs], np.percentile(v, qs).tolist())),
                           "energy_frac_above_eps": {str(t): float(st.we[v > t].sum() / st.we.sum()) for t in (1e-4, 1e-3, 1e-2)}}
                     for lab, v in rho.items()}
        log(f"per-voxel rho ({key}): " + "; ".join(f"{lab}: p99 {d['pct']['99']:.1e} max {d['pct']['100']:.1e}" for lab, d in pvox[key].items()))
    out["pervoxel"] = pvox
    out["timing"] = tim
    out["seconds"] = time.time() - t_start
    od = HERE / "out"
    od.mkdir(exist_ok=True)
    json.dump(out, open(od / f"{a.region}_{a.phys}.json", "w"), indent=1)
    log(f"wrote out/{a.region}_{a.phys}.json ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
