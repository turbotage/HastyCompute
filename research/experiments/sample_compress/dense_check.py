"""Direct-sum reference for the per-readout compression (direction 1) and the per-readout knot-time approximation
(direction 3), on real encode-0 readouts at the native (2 mm) scale.

Operator (direct sum, no NUFFT): unknowns = 16 periodic cubic cardiac knot images on a g^3 voxel grid spanning the
160 mm box (corners included), each voxel with its ratemap rate z_v (truth R2* + i field, water) and the sample's
concomitant phase:
    (A x)_j = sum_b beta_b(u_j) sum_v exp(-i k_j.r_v - z_v tau_j - i c_j.mono(r_v)) x_{b,v}
Compressed (subset S per readout, local q-tap E fitted on an independent continuous point-source sketch,
W = E^H E, b_S = E^H y):
    grad = A_S^H (W A_S x - b_S),   f(x) = (A_S x)^H W A_S x - 2 Re (A_S x)^H b_S + ||y||^2.
Reports: ||A^H A - A_S^H W A_S||_2 / ||A^H A||_2 (Lanczos), the same on a signal-weighted image (truth PD in the box),
||A^H y - A_S^H b_S|| / ||A^H y|| with y = A x_true + white noise, |f - f~| / f on random x, the noise covariance of
b_S against sigma^2 W, and the per-readout knot-time operator error ||A_pr - A_ps||_2 / ||A_ps||_2 (beta at the
readout midpoint vs per-sample).

run: flock .. systemd-run .. rustlib/.venv/bin/python -I research/experiments/sample_compress/dense_check.py [g] [nro] [os,q;os,q..]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_common as sc  # noqa: E402

OUT = os.path.join(HERE, "out")
M = sc.M_CARD


def truth_at(pos, name):
    a = np.nan_to_num(np.load(os.path.join(sc.common.SCAN, name)).astype(np.float64))
    dims = np.array(a.shape)
    idx = np.clip(np.rint(pos / sc.VOX + (dims - 1) / 2).astype(int), 0, dims - 1)
    return a[idx[:, 0], idx[:, 1], idx[:, 2]]


class Op:
    """per-readout dense phase matrices, knot weights; forward / adjoint on X [M, nv]"""

    def __init__(self, rds, pos, z, midpoint=False):
        self.Phi, self.B = [], []
        for k, tau, w, ph in rds:
            self.Phi.append(np.exp(-1j * (k @ pos.T + w @ sc.mono(pos).T) - tau[:, None] * z[None, :]))
            if midpoint:
                tm = np.full_like(tau, 0.5 * (tau[0] + tau[-1]))
                self.B.append(sc.knot_basis(ph, tm))
            else:
                self.B.append(sc.knot_basis(ph, tau))
        self.nv = pos.shape[0]

    def fwd(self, X, r):
        return ((self.Phi[r] @ X.T) * self.B[r]).sum(1)

    def adj(self, y, r):
        return (self.Phi[r].conj().T @ (y[:, None] * self.B[r])).T


def main():
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    nro = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    cfgs = [tuple(float(v) for v in c.split(",")) for c in (sys.argv[3] if len(sys.argv) > 3 else "1.35,32").split(";") if c]
    do_mid = "--mid" in sys.argv
    t0 = time.time()
    rng = np.random.default_rng(11)
    ros = np.sort(rng.choice(np.arange(0, sc.META["n_readouts"], 7), nro, replace=False))
    rds = sc.load_readouts(ros)
    half, C = 0.08, sc.BOX_CENTRE
    u = np.linspace(-1, 1, g)
    pos = C[None, :] + half * np.stack(np.meshgrid(u, u, u, indexing="ij"), -1).reshape(-1, 3)
    R = truth_at(pos, "truth_r2s.npy")
    om = 2 * np.pi * sc.F0 * truth_at(pos, "field_ppm_recon.npy")
    z = R + 1j * om
    pd = truth_at(pos, "truth_pd_water.npy") + truth_at(pos, "truth_pd_fat.npy")
    A = Op(rds, pos, z)
    nv, nx = pos.shape[0], M * pos.shape[0]
    print(f"{nro} readouts, {g}^3 voxels, {nx} unknowns ({time.time()-t0:.0f}s)", flush=True)
    wb = sc.offres_band(half, C)

    def normal_full(X):
        return sum(A.adj(A.fwd(X, r), r) for r in range(nro))

    # unknowns of knots no readout touches are a null space: restrict power iterations to the active knots
    act = np.nonzero(sum(np.abs(b).sum(0) for b in A.B) > 0)[0]
    res_act = len(act)

    def top_eig(fn, it=60, seed=0):
        """largest |eigenvalue| of a Hermitian operator on [M, nv] (power iteration on the active knots)"""
        X = np.zeros((M, nv), complex)
        X[act] = np.random.default_rng(seed).normal(size=(len(act), nv)) + 1j * np.random.default_rng(seed + 1).normal(size=(len(act), nv))
        lam_ = 0.0
        for _ in range(it):
            X /= np.linalg.norm(X)
            Y = fn(X)
            lam_ = np.vdot(X, Y).real
            X = Y
        return abs(lam_)

    lam = top_eig(normal_full)
    path = os.path.join(OUT, f"dense_check_g{g}_r{nro}.json")
    res = json.load(open(path)) if os.path.exists(path) else {"configs": []}
    res.update({"readouts": ros.tolist(), "g": g, "unknowns": nx, "normal_norm": float(lam), "active_knots": int(res_act)})
    # signal-weighted image: truth PD (static over knots) + a cardiac modulation on the vessel voxels
    lum = truth_at(pos, "truth_lumen_fraction.npy")
    ph_b = 2 * np.pi * np.arange(M) / M
    Xsig = pd[None, :] * (1 + 0.5 * lum[None, :] * np.cos(ph_b[:, None] + 2 * np.pi * rng.random(nv)[None, :]))
    Xsig = Xsig.astype(complex)
    sigma = 0.02 * np.sqrt(np.mean([np.mean(np.abs(A.fwd(Xsig, r)) ** 2) for r in range(nro)]))
    ys = [A.fwd(Xsig, r) + sigma * (rng.normal(size=len(rds[r][1])) + 1j * rng.normal(size=len(rds[r][1]))) / np.sqrt(2) for r in range(nro)]
    AHy = sum(A.adj(ys[r], r) for r in range(nro))
    NXs = normal_full(Xsig)
    for os_, q in cfgs:
        q = int(q)
        comp = []
        nS = 0
        for r, rd in enumerate(rds):
            B, dwell = sc.band(rd, "cube", half, 1.0, wb)
            S = sc.select_warped(B, dwell, os_)
            Vtr = sc.train_columns(rd, 300 + 5 * r, 1800, "cube", half, 1.0, wb)
            cols, coef = sc.local_E(Vtr, S, q)
            E = sc.E_dense(cols, coef, len(S))
            comp.append((S, E, E.conj().T @ E))
            nS += len(S)

        def normal_comp(X):
            out = 0
            for r in range(nro):
                S, E, W = comp[r]
                yS = A.fwd(X, r)[S]
                full = np.zeros(len(rds[r][1]), complex)
                full[S] = W @ yS
                out = out + A.adj(full, r)
            return out

        dl = top_eig(lambda X: normal_full(X) - normal_comp(X), it=40, seed=5)
        AHy_c = 0
        bS = []
        for r in range(nro):
            S, E, W = comp[r]
            b = E.conj().T @ ys[r]
            bS.append(b)
            full = np.zeros(len(rds[r][1]), complex)
            full[S] = b
            AHy_c = AHy_c + A.adj(full, r)
        e_sig = np.linalg.norm(NXs - normal_comp(Xsig)) / np.linalg.norm(NXs)
        e_ahy = np.linalg.norm(AHy - AHy_c) / np.linalg.norm(AHy)
        # objective: f(x) vs f~(x) on random and signal-like x
        fe = []
        yy = sum(np.vdot(y, y).real for y in ys)
        for t in range(4):
            X = Xsig * (1 + 0.3 * rng.normal(size=Xsig.shape)) if t else Xsig * 0.9
            f = sum(np.linalg.norm(A.fwd(X, r) - ys[r]) ** 2 for r in range(nro))
            ft = 0.0
            for r in range(nro):
                S, E, W = comp[r]
                a = A.fwd(X, r)[S]
                ft += np.vdot(a, W @ a).real - 2 * np.vdot(a, bS[r]).real
            ft += yy
            fe.append(abs(f - ft) / f)
        # noise covariance of b_S (first readout): E[b b^H] = sigma^2 W
        S, E, W = comp[0]
        nd = 3000
        Nn = (rng.normal(size=(E.shape[0], nd)) + 1j * rng.normal(size=(E.shape[0], nd))) / np.sqrt(2)
        Bn = E.conj().T @ Nn
        Cn = Bn @ Bn.conj().T / nd
        dW = np.diag(W).real
        ratio = np.diag(Cn).real / dW                                   # E = 1, sd = 1/sqrt(nd)
        rho_c = Cn[np.arange(len(dW) - 1), np.arange(1, len(dW))] / np.sqrt(dW[:-1] * dW[1:])
        rho_w = W[np.arange(len(dW) - 1), np.arange(1, len(dW))] / np.sqrt(dW[:-1] * dW[1:])
        cov_err = (float(ratio.mean() - 1), float(ratio.std()), float(np.abs(rho_c - rho_w).std()))
        ent = {"os": os_, "q": q, "nS": nS, "n": sum(len(rd[1]) for rd in rds), "reduction": sum(len(rd[1]) for rd in rds) / nS,
               "normal_err_spectral": float(abs(dl) / lam), "normal_err_signal_image": float(e_sig), "AHy_err": float(e_ahy),
               "objective_rel_err_max": float(max(fe)), "bS_cov_diag_ratio_mean_minus_1": cov_err[0], "bS_cov_diag_ratio_sd": cov_err[1], "bS_cov_adjacent_corr_err_sd": cov_err[2], "cov_sampling_sd_expected": float(1 / np.sqrt(nd))}
        res["configs"].append(ent)
        print(json.dumps(ent), f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(path, "w"), indent=1)
    if not do_mid:
        return
    # direction 3: knot weights at the readout midpoint instead of per sample
    Am = Op(rds, pos, z, midpoint=True)
    del Am.Phi
    Am.Phi = A.Phi

    def diff_normal(X):
        out = 0
        for r in range(nro):
            d = A.fwd(X, r) - Am.fwd(X, r)
            out = out + A.adj(d, r) - Am.adj(d, r)
        return out
    dd = top_eig(diff_normal, it=40, seed=9)
    num = sum(np.linalg.norm(A.fwd(Xsig, r) - Am.fwd(Xsig, r)) ** 2 for r in range(nro))
    den = sum(np.linalg.norm(A.fwd(Xsig, r)) ** 2 for r in range(nro))
    dB = np.concatenate([np.abs(A.B[r] - Am.B[r]).max(1) for r in range(nro)])
    res["knot_midpoint"] = {"op_err_spectral": float(np.sqrt(abs(dd) / lam)), "op_err_signal_image": float(np.sqrt(num / den)),
                            "dbeta_max": float(dB.max()), "dbeta_rms": float(np.sqrt(np.mean(dB ** 2)))}
    print("knot midpoint:", res["knot_midpoint"], flush=True)
    res["seconds"] = time.time() - t0
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
