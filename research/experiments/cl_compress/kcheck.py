"""Data-space check of the joint (coil x time) compression at real k-space samples (CPU).

For the box region (ratemap physics, proton-density voxel weights), build
  joint (d):   Phi_q from the weighted Gram (time weights --tk), B_cq(tau) by least squares at the actual ADC times,
  seg_svd:     coil-common optimal segments D_l, a_l(tau) by least squares at the actual ADC times,
then for selected ADC indices n and NR readouts spread over the scan compute by direct DFT at the real k_{r,n}
  e_c(r, n) = sum_v exp(-i k.v) Delta_c(v, tau_n) x(v),   s_c(r, n) = sum_v exp(-i k.v) E_c(v, tau_n) x(v),
x = truth_pd_water + truth_pd_fat (real, positive: the most coherent case at k = 0), and compare the relative error
per n with the separable (image-domain, proportional) prediction. Prints a table and writes out/kcheck_<tk>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cl_rank import Setup, S, chunks, load_any, log, sqrtm_h  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tk", default="u")
    ap.add_argument("--nr", type=int, default=48)
    ap.add_argument("--eps", default="1e-2,1e-3,1e-4")
    a = ap.parse_args()
    st = Setup("box", "ratemap")
    C = st.C
    res_json = json.load(open(HERE / "out/box_ratemap.json"))["res"][f"e{a.tk}"]
    eps = [float(e) for e in a.eps.split(",")]
    # time basis and reduced Gram (e weights) as cl_rank
    ns = st.ns
    Ge = np.zeros((ns, ns), complex)
    Gu = np.zeros((ns, ns), complex)
    for sl in chunks(st.nv, 16384):
        g = st.g(sl, st.ts)
        a1 = g * np.sqrt(st.we[sl] * st.cE[sl])[:, None]
        Ge += a1.T @ a1.conj()
        a1 = g * np.sqrt(st.cE[sl])[:, None]
        Gu += a1.T @ a1.conj()
    ev, Vall = np.linalg.eigh(Ge / np.trace(Ge).real + Gu / np.trace(Gu).real)
    K = int(json.load(open(HERE / "out/box_ratemap.json"))["K"])
    V = Vall[:, ::-1][:, :K]
    tw = st.tw[a.tk]
    Mh = sqrtm_h(V.conj().T @ (tw[:, None] * V))
    IM = np.kron(np.eye(C), Mh)
    CK = C * K
    H = np.zeros((CK, CK), complex)
    for sl in chunks(st.nv, 8192):
        h = st.g(sl, st.ts) @ V.conj()
        z = (st.Sv[sl][:, :, None] * h[:, None, :]).reshape(-1, CK) @ IM.T
        z *= np.sqrt(st.we[sl])[:, None]
        H += z.T @ z.conj()
    lam, Q = np.linalg.eigh((H + H.conj().T) / 2)
    lam, Q = lam[::-1], Q[:, ::-1]
    Hc = np.einsum("ckcl->kl", H.reshape(C, K, C, K))
    mu, Ut = np.linalg.eigh((Hc + Hc.conj().T) / 2)
    Ut = Ut[:, ::-1]
    Rs = {e: res_json["d"][str(e)] for e in eps}
    Ls = {e: res_json["seg_svd_L"][str(e)] for e in eps}
    Rmax = max(r for r in Rs.values() if r)
    Lmax = max(l for l in Ls.values() if l)
    log(f"K {K}, joint R {Rs}, seg L {Ls}")
    # spatial functions on all voxels: Phi (Rmax x nv) = Q^H z'_r ; seg diagonals D (Lmax x nv) = Ut^H Mh h_r
    Phi = np.empty((Rmax, st.nv), complex)
    D = np.empty((Lmax, st.nv), complex)
    for sl in chunks(st.nv, 8192):
        h = st.g(sl, st.ts) @ V.conj()
        z = (st.Sv[sl][:, :, None] * h[:, None, :]).reshape(-1, CK) @ IM.T
        Phi[:, sl] = (z @ Q[:, :Rmax].conj()).T
        D[:, sl] = ((h @ Mh.T) @ Ut[:, :Lmax].conj()).T
    # trajectory: NR readouts over the scan, all ADC samples; selected ADC indices
    smp = load_any(S / "samples.hv", lazy=True)
    ros = np.linspace(0, smp.shape[0] - 1, a.nr).astype(int)
    kr = np.stack([np.asarray(smp[int(r)], np.float64)[:, :3] for r in ros], 0)  # nr x na x 3 (rad/m)
    na = kr.shape[1]
    base = [0, 1, 2, 3, 5, 8, 12, 18, 26, 38, 55, 80, 120, 180, 270, 400, 650, 1000, 1500, 1995]
    nsel = np.unique(np.array(base + [na - 1 - b for b in base]))
    x = (np.load(S / "truth_pd_water.npy").astype(np.float64) + np.load(S / "truth_pd_fat.npy"))[tuple(st.ijk.T)]
    w = st.we  # = x^2 normalised
    kap = np.load(HERE / "prep.npz")["kap"]
    rows = []
    for n in nsel:
        t = st.tau[n]
        g = np.concatenate([st.g(sl, np.array([t]))[:, 0] for sl in chunks(st.nv, 1 << 18)])
        E = st.Sv.T * g[None, :]  # C x nv
        ph = np.exp(-1j * (kr[:, n, :] @ st.pos.T))  # nr x nv
        sig = (E * x[None, :]) @ ph.T  # C x nr
        es = (np.abs(sig) ** 2).sum()
        rec = dict(n=int(n), kap=float(kap[n]), sig=float(es))
        Ew = (np.abs(E) ** 2 * w[None, :]).sum()
        for e in eps:
            R = Rs[e]
            if R:
                P = Phi[:R]
                lamR = (np.abs(P) ** 2 * w[None, :]).sum(1)  # = lambda_q
                B = ((E * w[None, :]) @ P.conj().T) / lamR[None, :]  # C x R  (LS at tau_n)
                Dl = E - B @ P
                err = (Dl * x[None, :]) @ ph.T
                rec[f"d@{e}"] = float(np.sqrt((np.abs(err) ** 2).sum() / es))
                rec[f"d@{e}_sep"] = float(np.sqrt((np.abs(Dl) ** 2 * w[None, :]).sum() / Ew))
            L = Ls[e]
            if L:
                Dd = D[:L]
                wc = w * st.cE
                Gd = (Dd * wc[None, :]) @ Dd.conj().T
                al = np.linalg.solve(Gd.T, (Dd.conj() * wc[None, :]) @ g)  # a(tau_n), weighted LS
                dg = g - al @ Dd
                Dl = st.Sv.T * dg[None, :]
                err = (Dl * x[None, :]) @ ph.T
                rec[f"seg@{e}"] = float(np.sqrt((np.abs(err) ** 2).sum() / es))
                rec[f"seg@{e}_sep"] = float(np.sqrt((np.abs(Dl) ** 2 * w[None, :]).sum() / Ew))
        rows.append(rec)
        log(" ".join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    # data-energy-weighted totals over all ADC indices (log-interpolated between the selected ones)
    tot = {}
    nn = np.array([r["n"] for r in rows])
    sigs = np.array([r["sig"] for r in rows])
    s_all = np.exp(np.interp(np.arange(na), nn, np.log(sigs)))
    for key in [k for k in rows[0] if "@" in k]:
        ee = np.array([r[key] ** 2 * r["sig"] for r in rows])
        e_all = np.exp(np.interp(np.arange(na), nn, np.log(np.maximum(ee, 1e-300))))
        tot[key] = float(np.sqrt(e_all.sum() / s_all.sum()))
    log("data-weighted totals: " + " ".join(f"{k}={v:.2e}" for k, v in tot.items()))
    json.dump(dict(tk=a.tk, R=Rs, L=Ls, rows=rows, totals=tot), open(HERE / f"out/kcheck_{a.tk}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
