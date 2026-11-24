"""Numerical check of research/temporal_pyramid.md (Theorems 1-4) on an exactly solvable toy.

Toy: complex dynamic n x n image, C coils, Cartesian k-space lines (one ky line per readout, all kx) at sample-level
cardiac times, periodic uniform B-splines of degree p with K knots. Normal operators are built exactly:
N = sum_lines G_line (x) E_line, G_line = sum_{r in line} b(t_r) b(t_r)^T, E_line = sum_c S_c^H F^H P_line F S_c.
Outputs: a small JSON + text summary next to this file (no arrays).
"""
import json, sys, os, time
import numpy as np

rng = np.random.default_rng(1)

# ---------------------------------------------------------------- periodic B-splines
def bspl(t, K, p):
    """values B_k(t), k = 0..K-1, of the periodic uniform B-splines of degree p on [0,1) with K knots; t: (R,)"""
    u = (np.asarray(t)[:, None] * K - np.arange(K)[None, :]) % K  # position relative to knot k, in [0, K)
    # cardinal B-spline of degree p supported on [0, p+1)
    def card(x, p):
        if p == 0:
            return ((x >= 0) & (x < 1)).astype(float)
        return (x * card(x, p - 1) + (p + 1 - x) * card(x - 1, p - 1)) / p
    return card(u, p)

def subdiv(Kc, p, m=2):
    """S (Kf x Kc): B^c_k = sum_i S_ik B^f_i for Kf = m Kc (fit by least squares on a dense grid; exact for nesting)"""
    t = (np.arange(4000) + 0.5) / 4000
    Bc, Bf = bspl(t, Kc, p), bspl(t, m * Kc, p)
    S, *_ = np.linalg.lstsq(Bf, Bc, rcond=None)
    assert np.abs(Bf @ S - Bc).max() < 1e-9, "levels not nested"
    return S

# ---------------------------------------------------------------- toy problem
def make_problem(n=10, C=4, R=400, stationary=False, dens_pow=1.5, sigma=0.0, seed=0, harmonics=6):
    r = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(n) - n / 2 + 0.5, np.arange(n) - n / 2 + 0.5, indexing="ij")
    # coil sensitivities: smooth complex Gaussians around the FOV
    ang = 2 * np.pi * np.arange(C) / C
    sens = np.stack([np.exp(-((xx - 0.6 * n * np.cos(a)) ** 2 + (yy - 0.6 * n * np.sin(a)) ** 2) / (0.8 * n) ** 2)
                     * np.exp(1j * (0.3 * xx * np.cos(a) + 0.3 * yy * np.sin(a))) for a in ang])
    if C == 1:
        sens = np.ones((1, n, n), complex)
    # truth: static background + disc whose magnitude and phase vary periodically (smooth, `harmonics` harmonics)
    disc = ((xx - 1) ** 2 + (yy + 0.5) ** 2) < (0.3 * n) ** 2
    co = r.normal(size=(harmonics, 2)) / (1 + np.arange(harmonics))[:, None] ** 1.2
    def truth(t):
        t = np.atleast_1d(t)
        h = np.arange(1, harmonics + 1)
        s = (co[:, 0][None] * np.cos(2 * np.pi * h[None] * t[:, None]) + co[:, 1][None] * np.sin(2 * np.pi * h[None] * t[:, None])).sum(1)
        mag = 1.0 + 0.3 * s
        ph = 1.2 * s
        img = 0.4 * np.exp(-(xx ** 2 + yy ** 2) / (0.5 * n) ** 2)[None] + disc[None] * (mag * np.exp(1j * ph))[:, None, None]
        return img.reshape(len(t), -1)
    # readouts: times sequential with jitter; ky line per readout
    t_r = (np.arange(R) + r.uniform(0, 1, R)) / R
    ky = np.arange(n) - n // 2
    if stationary:
        line = None  # every readout samples all of k-space
    else:
        pr = 1.0 / (1 + np.abs(ky)) ** dens_pow
        line = r.choice(n, size=R, p=pr / pr.sum())
    return dict(n=n, C=C, R=R, sens=sens.reshape(C, -1), truth=truth, t=t_r, line=line, sigma=sigma, rng=r)

def fft_mat(n):
    F1 = np.fft.fftshift(np.fft.fft(np.eye(n), axis=0, norm="ortho"), axes=0)
    return np.kron(F1, F1)  # 2-D DFT on row-major (y, x) vectors, ky-major rows

def line_normals(P):
    n, C = P["n"], P["C"]
    F = fft_mat(n)
    S = P["sens"]
    Es, Ops = [], []
    lines = [None] if P["line"] is None else list(range(n))
    for l in lines:
        rows = np.arange(n * n) if l is None else np.arange(l * n, (l + 1) * n)
        Fl = F[rows]
        op = np.concatenate([Fl * S[c][None, :] for c in range(C)], 0)  # (C*rows, npx)
        Ops.append(op)
        Es.append(op.conj().T @ op)
    return Es, Ops

def simulate(P, Ops, holdout=None):
    """per readout data y_r = op_line x*(t_r) + noise"""
    X = P["truth"](P["t"])
    ys = []
    for r_, t in enumerate(P["t"]):
        op = Ops[0] if P["line"] is None else Ops[P["line"][r_]]
        y = op @ X[r_]
        y = y + P["sigma"] * (P["rng"].normal(size=y.shape) + 1j * P["rng"].normal(size=y.shape)) / np.sqrt(2)
        ys.append(y)
    return ys

def normal_system(P, Es, Ops, ys, K, p, use):
    npx = P["n"] ** 2
    B = bspl(P["t"], K, p)  # (R, K)
    N = np.zeros((K * npx, K * npx), complex)
    b = np.zeros(K * npx, complex)
    groups = [np.flatnonzero(use)] if P["line"] is None else [np.flatnonzero(use & (P["line"] == l)) for l in range(P["n"])]
    for gi, idx in enumerate(groups):
        if len(idx) == 0:
            continue
        G = B[idx].T @ B[idx]
        N += np.kron(G, Es[gi])
        op = Ops[gi]
        for r_ in idx:
            b += np.kron(B[r_], op.conj().T @ ys[r_])
    return N, b

def eval_x(c, K, p, t, npx):
    return bspl(t, K, p) @ c.reshape(K, npx)

def main():
    out = {}
    lines = []
    def log(s):
        print(s); lines.append(s); sys.stdout.flush()

    # ------------------------------------------------------------ T1: Theorem 1, exact energy projection
    for stat in (True, False):
        for p in (0, 1, 3):
            P = make_problem(n=8, C=4, R=300, stationary=stat, sigma=0.05, seed=2)
            Es, Ops = line_normals(P)
            ys = simulate(P, Ops)
            use = np.ones(P["R"], bool)
            Kc, Kf = 6, 12
            Nc, bc = normal_system(P, Es, Ops, ys, Kc, p, use)
            Nf, bf = normal_system(P, Es, Ops, ys, Kf, p, use)
            reg = 1e-8 * np.trace(Nf).real / Nf.shape[0]  # tiny, level-consistent (S^T S ≠ I though) -> use Q_c = S^H Q_f S
            S = np.kron(subdiv(Kc, p), np.eye(P["n"] ** 2))
            Qf = reg * np.eye(Nf.shape[0]); Qc = S.conj().T @ Qf @ S
            cf = np.linalg.solve(Nf + Qf, bf)
            cc = np.linalg.solve(Nc + Qc, bc)
            Hf = Nf + Qf
            Pi_cf = S @ np.linalg.solve(S.conj().T @ Hf @ S, S.conj().T @ Hf @ cf)
            e = np.linalg.norm(S @ cc - Pi_cf) / np.linalg.norm(S @ cc)
            gal = np.linalg.norm(S.conj().T @ Nf @ S - Nc) / np.linalg.norm(Nc)
            log(f"T1 {'stationary' if stat else 'line-sampled'} p={p}: |N_c - S^H N_f S|/|N_c| = {gal:.1e}; |S c_c - Pi_H c_f|/|S c_c| = {e:.1e}")
            out[f"T1_{'st' if stat else 'ns'}_p{p}"] = e

    # ------------------------------------------------------------ T2: Corollary 2, weighted time averages
    for p in (0, 1, 3):
        P = make_problem(n=8, C=4, R=300, stationary=True, sigma=0.05, seed=3)
        Es, Ops = line_normals(P)
        ys = simulate(P, Ops)
        use = np.ones(P["R"], bool)
        Kc, Kf = 6, 12
        Nc, bc = normal_system(P, Es, Ops, ys, Kc, p, use)
        Nf, bf = normal_system(P, Es, Ops, ys, Kf, p, use)
        cc, cf = np.linalg.solve(Nc, bc), np.linalg.solve(Nf, bf)
        npx = P["n"] ** 2
        xc, xf = eval_x(cc, Kc, p, P["t"], npx), eval_x(cf, Kf, p, P["t"], npx)  # at the sample times (w = empirical)
        Bc = bspl(P["t"], Kc, p)
        mom = np.abs(Bc.T @ (xf - xc)).max() / np.abs(Bc.T @ xc).max()
        mean_gap = np.abs((xf - xc).sum(0)).max() / np.abs(xc.sum(0)).max()
        msg = f"T2 stationary p={p}: max_k |sum_r B^c_k(t_r)(x_f - x_c)| / scale = {mom:.1e}; data-weighted time mean gap {mean_gap:.1e}"
        if p == 0:
            # coarse frame = sample-count-weighted mean of its two fine frames
            Sf = bspl(P["t"], Kf, 0); w = Sf.sum(0)
            C2 = cf.reshape(Kf, npx); avg = (w[0::2, None] * C2[0::2] + w[1::2, None] * C2[1::2]) / (w[0::2] + w[1::2])[:, None]
            plain = 0.5 * (C2[0::2] + C2[1::2])
            cr = cc.reshape(Kc, npx)
            msg += f"; p=0 weighted-average identity {np.abs(avg - cr).max() / np.abs(cr).max():.1e} (plain mean {np.abs(plain - cr).max() / np.abs(cr).max():.1e})"
        log(msg)
        out[f"T2_p{p}"] = mom
    # non-stationary, single coil: per-ky identity holds, per-voxel does not
    for p in (0, 3):
        P = make_problem(n=8, C=1, R=300, stationary=False, sigma=0.05, seed=4)
        Es, Ops = line_normals(P)
        ys = simulate(P, Ops)
        use = np.ones(P["R"], bool)
        Kc, Kf = 6, 12
        Nc, bc = normal_system(P, Es, Ops, ys, Kc, p, use)
        Nf, bf = normal_system(P, Es, Ops, ys, Kf, p, use)
        reg = 1e-9 * np.trace(Nf).real / Nf.shape[0]
        S = np.kron(subdiv(Kc, p), np.eye(P["n"] ** 2))
        Qf = reg * np.eye(Nf.shape[0]); Qc = S.conj().T @ Qf @ S
        cc, cf = np.linalg.solve(Nc + Qc, bc), np.linalg.solve(Nf + Qf, bf)
        npx = P["n"] ** 2; n = P["n"]
        xc, xf = eval_x(cc, Kc, p, P["t"], npx), eval_x(cf, Kf, p, P["t"], npx)
        Bc = bspl(P["t"], Kc, p)
        vox = np.abs(Bc.T @ (xf - xc)).max() / np.abs(Bc.T @ xc).max()
        F = fft_mat(n)
        kc, kf = xc @ F.T, xf @ F.T  # k-space per time
        worst = 0.0
        for l in range(n):
            idx = np.flatnonzero(P["line"] == l)
            rows = np.arange(l * n, (l + 1) * n)
            if len(idx) == 0:
                continue
            d = Bc[idx].T @ (kf[idx][:, rows] - kc[idx][:, rows])
            worst = max(worst, np.abs(d).max() / (np.abs(Bc[idx].T @ kc[idx][:, rows]).max() + 1e-30))
        log(f"T2 line-sampled 1 coil p={p}: per-voxel moment identity {vox:.1e} (fails), per-ky moment identity with that line's own sample times {worst:.1e}")
        out[f"T2ns_p{p}"] = (vox, worst)

    # ------------------------------------------------------------ T3: arg of a weighted complex mean
    a, b_, d = 1.0, 0.6, 0.94
    rho = (a - b_) / (a + b_)
    exact = np.angle(a * np.exp(1j * d / 2) + b_ * np.exp(-1j * d / 2))
    lin = (a * d / 2 - b_ * d / 2) / (a + b_)
    log(f"T3 arg of weighted mean: exact {exact:.6f}, magnitude-weighted phase mean {lin:.6f}, cubic term predicted {rho * (1 - rho ** 2) * d ** 3 / 24:.6f}, actual gap {exact - lin:.6f}")

    # ------------------------------------------------------------ T4: refinement, divergence, regularisers
    log("T4: variable-density line sampling, 4 coils, noise; truth error vs knots and regulariser (p = 3)")
    p = 3
    P = make_problem(n=8, C=4, R=480, stationary=False, sigma=0.08, seed=5, dens_pow=1.8, harmonics=8)
    Es, Ops = line_normals(P)
    ys = simulate(P, Ops)
    npx, n = P["n"] ** 2, P["n"]
    hold = P["rng"].uniform(size=P["R"]) < 0.12
    use = ~hold
    tt = (np.arange(400) + 0.5) / 400
    Xt = P["truth"](tt)
    F = fft_mat(n)
    # per-ky coverage of the fine knot intervals
    def coverage(K):
        cov = np.zeros(n)
        for l in range(n):
            ts = P["t"][use & (P["line"] == l)]
            cov[l] = len(np.unique(np.floor(ts * K).astype(int))) / K
        return cov
    def detail_Q(K, lam_k):
        """sum over ky of lam_k[ky] * (temporal L2 detail of level K vs K/2) (x) projector on line ky"""
        Kc = K // 2
        tq = (np.arange(4000) + 0.5) / 4000
        Bf = bspl(tq, K, p); G = Bf.T @ Bf / len(tq)
        Sm = subdiv(Kc, p)
        Pc = Sm @ np.linalg.solve(Sm.T @ G @ Sm, Sm.T @ G)
        D = (np.eye(K) - Pc).T @ G @ (np.eye(K) - Pc)
        Q = np.zeros((K * npx, K * npx), complex)
        for l in range(n):
            rows = np.arange(l * n, (l + 1) * n)
            Pl = F[rows].conj().T @ F[rows]
            Q += lam_k[l] * np.kron(D, Pl)
        return Q
    def errs(c, K):
        x = eval_x(c, K, p, tt, npx)
        te = np.linalg.norm(x - Xt) / np.linalg.norm(Xt)
        ho = 0.0; hn = 0.0
        Bh = bspl(P["t"][hold], K, p)
        for i, r_ in enumerate(np.flatnonzero(hold)):
            xr = Bh[i] @ c.reshape(K, npx)
            ho += np.linalg.norm(Ops[P["line"][r_]] @ xr - ys[r_]) ** 2
            hn += np.linalg.norm(ys[r_]) ** 2
        return te, ho / hn
    res = {}
    prev = None
    scale = None
    for K in (6, 12, 24, 48):
        N, b = normal_system(P, Es, Ops, ys, K, p, use)
        if scale is None:
            scale = np.trace(N).real / N.shape[0]
        eps = 1e-6 * scale * np.eye(N.shape[0])
        row = {}
        c0 = np.linalg.solve(N + eps, b)
        row["none"] = errs(c0, K)
        if prev is not None:
            Kc = K // 2
            Sm = np.kron(subdiv(Kc, p), np.eye(npx))
            # "averages must equal the coarse solution": mu |R_L2 c - c_c|^2, R_L2 = temporal L2 restriction
            tq = (np.arange(4000) + 0.5) / 4000
            Bf = bspl(tq, K, p); G = Bf.T @ Bf / len(tq)
            Sm1 = subdiv(Kc, p)
            R1 = np.linalg.solve(Sm1.T @ G @ Sm1, Sm1.T @ G)
            R = np.kron(R1, np.eye(npx))
            for mu_rel in (1.0, 100.0):
                mu = mu_rel * scale
                ca = np.linalg.solve(N + eps + mu * R.conj().T @ R, b + mu * R.conj().T @ prev)
                row[f"avg_mu{mu_rel:g}"] = errs(ca, K)
            # detail penalty, uniform and k-adaptive (coverage of the fine knot intervals per ky line)
            cov = coverage(K)
            for lam_rel in (0.3, 3.0):
                lam = lam_rel * scale
                cu = np.linalg.solve(N + eps + detail_Q(K, np.full(n, lam)), b)
                row[f"detail_uniform{lam_rel:g}"] = errs(cu, K)
                lk = lam * (1 - cov) / np.maximum(cov, 1e-3) + 0.02 * lam
                ck = np.linalg.solve(N + eps + detail_Q(K, lk), b)
                row[f"detail_kadapt{lam_rel:g}"] = errs(ck, K)
            # energy-metric anchor (R_H = coordinates of the H-projection): inert on the unregularised problem
            H = N + eps
            RH = np.linalg.solve(Sm.conj().T @ H @ Sm, Sm.conj().T @ H)
            ch = np.linalg.solve(H + 100 * scale * RH.conj().T @ RH, b + 100 * scale * RH.conj().T @ prev)
            row["avgH_mu100"] = errs(ch, K)
            # combination: L2 anchor + uniform detail penalty
            cb = np.linalg.solve(N + eps + 1.0 * scale * R.conj().T @ R + detail_Q(K, np.full(n, 3.0 * scale)), b + 1.0 * scale * R.conj().T @ prev)
            row["avg1+detail3"] = errs(cb, K)
            # mechanism: noise variance of the detail with the coarse part free (Schur complement) vs pinned (H_dd)
            Tn = np.linalg.svd(R1)[2][Kc:].conj().T  # orthonormal basis of ker R1 (temporal), K x (K - Kc)
            Rp = np.linalg.pinv(R1)                  # coarse part, right inverse of R1
            Bm = np.kron(np.hstack([Rp, Tn]), np.eye(npx))
            Hs = Bm.conj().T @ H @ Bm
            m = Kc * npx
            Hcc, Hcd, Hdd = Hs[:m, :m], Hs[:m, m:], Hs[m:, m:]
            Sd = Hdd - Hcd.conj().T @ np.linalg.solve(Hcc, Hcd)
            row["tr_inv_schur"] = float(np.trace(np.linalg.inv(Sd)).real)
            row["tr_inv_Hdd"] = float(np.trace(np.linalg.inv(Hdd)).real)
            row["lmin_schur_rel"] = float(np.linalg.eigvalsh(Sd)[0] / scale)
            row["lmin_Hdd_rel"] = float(np.linalg.eigvalsh(Hdd)[0] / scale)
            row["coverage_per_ky"] = [round(float(v), 2) for v in cov]
        res[K] = row
        prev = c0 if prev is None else c0  # coarse anchor = the unregularised solution of the previous level
        log(f"  K={K:2d}: " + "; ".join(f"{k} truth {v[0]:.3f} heldout {v[1]:.4f}" if isinstance(v, tuple) else f"{k} {v}" for k, v in row.items()))
    out["T4"] = {str(k): {kk: vv for kk, vv in v.items()} for k, v in res.items()}
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "toy_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    with open(os.path.join(here, "toy_results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    t0 = time.time(); main(); print(f"done {time.time() - t0:.0f}s")
