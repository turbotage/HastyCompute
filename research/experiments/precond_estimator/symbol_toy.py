"""Exact 2-D toy for structured preconditioner classes (research/precond_estimator.md §7).

Operator: Q = Σ_c S_c^H A^H A S_c + μ I with A[s, x] = exp(-i (k_s·x + ω(x) t_s)): out-and-back spokes (both ends at
k = 0, density ~ k^-1.7 like the yarn), smooth coils, an off-resonance map ω with a strong-gradient region. Local
picture: a voxel with field gradient g sees the samples at k_s + t_s g, so the second centre pass (t = 1) is shifted by g.

Preconditioner classes (all SPD), condition number of P Q on the whole space and the PCG iterations to 1e-3 / 1e-6:
  none, jacobi             identity / image-space diagonal
  dcd(eps, p)              D^-1/2 F^H (λ/λmax + ε)^-p F D^-1/2, λ = nominal Chan symbol (field ignored); p = 1 is the
                           current metric, p < 1 the bounded family
  seg(L) exact             inverse of the coil-free time-segmented surrogate Q~ = D^1/2 Σ_l E_l C_l E_l^H D^1/2 + μ,
                           E_l = diag(exp(i ω t_l)), C_l = Chan circulant of the samples in time window l
                           (the best member of the "sum of modulated circulants" class; applying Q~ costs L FFT pairs)
  seg(L) + K inner its     the same, inverted by a FIXED number K of circulant-preconditioned CG steps on Q~
                           (then it is not a fixed linear operator: flexible PCG outside)
  pc(J)                    product-convolution: J clusters of the local shift g(x) = ∇ω(x), per-cluster local Chan
                           symbol of the shifted samples, P = Σ_j W_j D^-1/2 F^H (σ_j + floor)^-1 F D^-1/2 W_j (hard W_j)
Run: OMP_NUM_THREADS=1 python symbol_toy.py   (single thread, ~1 min)
"""
import os, sys, time
import numpy as np

n = 32
N = n * n
rng = np.random.default_rng(3)
xs = np.arange(n) - n // 2
X, Y = np.meshgrid(xs, xs, indexing="ij")
pos = np.stack([X.ravel(), Y.ravel()], 1).astype(float)  # (N, 2) voxel units
r = np.hypot(X, Y).ravel()
support = r <= 12.5

# trajectory: out-and-back spokes, k = π v^3, v = 1 - |2u - 1|; t = u
nsp, ns = 96, 48
u = (np.arange(ns) + 0.5) / ns
v = 1 - np.abs(2 * u - 1)
kr = np.pi * v ** 3
ga = np.pi * (3 - np.sqrt(5))
ang = np.arange(nsp) * ga
K = np.concatenate([np.stack([kr * np.cos(a), kr * np.sin(a)], 1) for a in ang])  # (M, 2)
T = np.tile(u, nsp)
M = len(K)

# coils: 4 smooth complex maps
nc = 4
S = []
for c in range(nc):
    a = 2 * np.pi * c / nc
    cx, cy = 18 * np.cos(a), 18 * np.sin(a)
    mag = np.exp(-((pos[:, 0] - cx) ** 2 + (pos[:, 1] - cy) ** 2) / (2 * 14.0 ** 2))
    ph = 0.08 * (pos[:, 0] * np.sin(a) - pos[:, 1] * np.cos(a)) + c
    S.append(mag * np.exp(1j * ph))
S = np.array(S)
D = (np.abs(S) ** 2).sum(0)

# field: smooth bump near the edge plus a mild linear part; scaled so max |∇ω| (rad/voxel per unit time) = shift_max
bump = np.exp(-((pos[:, 0] - 7) ** 2 + (pos[:, 1] - 4) ** 2) / (2 * 4.0 ** 2))
w0 = bump + 0.15 * pos[:, 0] / n
gx = (-(pos[:, 0] - 7) / 16.0) * bump + 0.15 / n
gy = (-(pos[:, 1] - 4) / 16.0) * bump
g0 = np.stack([gx, gy], 1)
gmax0 = np.abs(np.hypot(gx, gy)[support]).max()

# unitary 2-D DFT matrix, index m ↔ frequency 2π m / n (FFT order)
f1 = np.exp(-2j * np.pi * np.outer(np.arange(n), np.arange(n)) / n) / np.sqrt(n)
Fm = np.kron(f1, f1)
freqs = 2 * np.pi * np.fft.fftfreq(n) * 1.0  # rad/voxel, FFT order
mx = 2 * np.pi * np.arange(n) / n


def fejer(uu):
    s = np.sin(uu / 2)
    out = np.where(np.abs(s) < 1e-12, float(n), np.sin(n * uu / 2) ** 2 / (n * np.where(np.abs(s) < 1e-12, 1, s) ** 2))
    return out


def chan_symbol(k, w=None):
    """λ(m) = Σ_s w_s F(k_sx - 2π m_x/n) F(k_sy - 2π m_y/n), flattened like pos (x-major)."""
    w = np.ones(len(k)) if w is None else w
    Fx = fejer(k[:, 0:1] - mx[None, :])
    Fy = fejer(k[:, 1:2] - mx[None, :])
    lam = (Fx * w[:, None]).T @ Fy  # (n, n)
    return lam.ravel()


def circ(lam):
    return Fm.conj().T @ (lam[:, None] * Fm)


def operator(shift_max):
    om = w0 * (shift_max / gmax0 if gmax0 > 0 else 0.0)
    A = np.exp(-1j * (K @ pos.T + np.outer(T, om)))  # (M, N)
    N0 = A.conj().T @ A
    G = (S.conj().T @ S)  # G[x, x'] = Σ_c conj(S_c(x)) S_c(x')
    return N0 * G, om


def pcg(Q, b, apply_P, xref, tol=(1e-3, 1e-6), maxit=600, flexible=False):
    x = np.zeros_like(b); rr = b.copy(); z = apply_P(rr); p = z.copy(); rz = np.vdot(rr, z).real
    out = {}; nb = np.linalg.norm(xref)
    zold = z; rold = rr
    for it in range(1, maxit + 1):
        Qp = Q @ p
        a = rz / np.vdot(p, Qp).real
        x = x + a * p; rr = rr - a * Qp
        e = np.linalg.norm(x - xref) / nb
        for t in tol:
            if t not in out and e <= t:
                out[t] = it
        if len(out) == len(tol):
            break
        z = apply_P(rr)
        if flexible:  # Polak–Ribière form (Notay's flexible CG with one vector)
            beta = np.vdot(z, rr - rold).real / rz
        else:
            beta = np.vdot(rr, z).real / rz
        rz_new = np.vdot(rr, z).real
        if flexible:
            rold = rr.copy()
        rz = rz_new
        p = z + beta * p
    return [out.get(t, f">{maxit}") for t in tol]


def kappa(Q, P):
    w, V = np.linalg.eigh((P + P.conj().T) / 2)
    w = np.maximum(w, 0)
    Ph = (V * np.sqrt(w)) @ V.conj().T
    ev = np.linalg.eigvalsh(Ph @ Q @ Ph)
    return ev.max() / ev.min(), ev


def main():
    lines = []
    def log(s):
        print(s); lines.append(s); sys.stdout.flush()
    log(f"2-D toy n={n}, {nsp} out-and-back spokes x {ns} samples (density ~k^-1.7), {nc} coils, support r<=12.5;"
        f" kappa = cond(P Q), PCG iterations to ||x - xhat||/||xhat|| <= 1e-3 / 1e-6")
    lam0 = chan_symbol(K)
    lmax = lam0.max()
    for shift in (0.0, 0.25 * np.pi, 0.5 * np.pi):
        t0 = time.time()
        Q0, om = operator(shift)
        mu = 1e-3 * np.real(np.diag(Q0))[support].mean()
        Q = Q0 + mu * np.eye(N)
        xstar = np.where(support, 1.0 + 0.5 * np.cos(pos[:, 0] / 3) * np.sin(pos[:, 1] / 4), 0) * np.exp(0.3j * pos[:, 1] / n)
        b = Q0 @ xstar
        xhat = np.linalg.solve(Q, b)
        log(f"\n## max local shift |g| t=1: {shift / np.pi:.2f} π rad/voxel (= {shift / np.pi:.2f} k_max); ridge μ = 1e-3 diag;"
            f" |xhat - x*|/|x*| = {np.linalg.norm(xhat - xstar) / np.linalg.norm(xstar):.3f}  ({time.time() - t0:.0f}s build)")
        Dm = np.maximum(D, 0.1 * np.median(D[support]))
        dih = 1 / np.sqrt(Dm)
        cands = {}
        cands["none"] = np.eye(N)
        cands["jacobi"] = np.diag(1 / np.real(np.diag(Q)))
        for eps, p in ((1e-3, 1.0), (1e-2, 1.0), (1e-1, 1.0), (1e-2, 0.5), (1e-1, 0.5), (3e-2, 0.75)):
            cands[f"dcd eps={eps:g} p={p:g}"] = dih[:, None] * circ((lam0 / lmax + eps) ** -p) * dih[None, :]
        # time-segmented surrogate (coil-free, D-scaled), exact inverse
        for L in (2, 4, 8):
            edges = np.linspace(0, 1, L + 1)
            Qt = mu * np.eye(N, dtype=complex)
            for l in range(L):
                sel = (T >= edges[l]) & (T < edges[l + 1])
                tl = T[sel].mean()
                lam_l = chan_symbol(K[sel])
                El = np.exp(1j * om * tl)
                Qt += (np.sqrt(D) * El)[:, None] * circ(lam_l) * (np.sqrt(D) * El.conj())[None, :]
            cands[f"seg L={L} exact inverse"] = np.linalg.inv(Qt)
            if L == 4:
                Qt4 = Qt
                # same class with the coils kept (Σ_c S_c^H Σ_l E_l C_l E_l^H S_c): C·L terms
                G = S.conj().T @ S
                Qc = mu * np.eye(N, dtype=complex)
                Qtz = mu * np.eye(N, dtype=complex)
                for l in range(L):
                    sel = (T >= edges[l]) & (T < edges[l + 1])
                    tl = T[sel].mean()
                    El = np.exp(1j * om * tl)
                    Cl = (El[:, None] * circ(chan_symbol(K[sel])) * El.conj()[None, :])
                    Qc += Cl * G
                    Al = np.exp(-1j * (K[sel] @ pos.T))
                    Tl = Al.conj().T @ Al  # exact Toeplitz of window l
                    Qtz += (El[:, None] * Tl * El.conj()[None, :]) * G
                cands["seg L=4 + coils (circ) inv"] = np.linalg.inv(Qc)
                cands["seg L=4 + coils (Toeplitz) inv"] = np.linalg.inv(Qtz)
                # D-sandwich without field, exact inverse of D^1/2 C D^1/2 + μ (the 'matched floor' done right)
                Qd = mu * np.eye(N, dtype=complex) + np.sqrt(D)[:, None] * circ(lam0) * np.sqrt(D)[None, :]
                cands["(D½ C D½ + μ)⁻¹, no field"] = np.linalg.inv(Qd)
        # product-convolution on clusters of the local shift g(x)
        g = g0 * (shift / gmax0 if gmax0 > 0 else 0.0)
        for J in (4, 16):
            if shift == 0:
                lab = np.zeros(N, int); cent = np.zeros((1, 2))
            else:
                # k-means on g over the support (few iterations), air goes to the nearest centre
                cent = g[support][rng.choice(support.sum(), J, replace=False)]
                for _ in range(30):
                    lab = np.argmin(((g[:, None, :] - cent[None]) ** 2).sum(-1), 1)
                    for j in range(J):
                        if (lab[support] == j).any():
                            cent[j] = g[support & (lab == j)].mean(0)
            P = np.zeros((N, N), complex)
            for j in range(len(cent)):
                w = (lab == j).astype(float)
                if not w.any():
                    continue
                sig = chan_symbol(K + T[:, None] * cent[j][None, :])
                Cj = circ(1 / (sig + 1e-3 * sig.max()))
                P += (w * dih)[:, None] * Cj * (w * dih)[None, :]
            cands[f"pc J={J}"] = P
        # A3: one free modulation U = diag(exp(i ω t_ref)) around the current metric (re-centres one t_ref)
        base = dih[:, None] * circ((lam0 / lmax + 1e-3) ** -1.0) * dih[None, :]
        for tref in (0.5, 1.0):
            U = np.exp(1j * om * tref)
            cands[f"U dcd U^H, t_ref={tref:g}"] = U[:, None] * base * U.conj()[None, :]
        # A4: k-space two-level: exact dense block of D^-1/2 Q D^-1/2 on low |k| (radius kc cells), circulant elsewhere
        kxi = np.fft.fftfreq(n) * n
        KX, KY = np.meshgrid(kxi, kxi, indexing="ij")
        Qs = dih[:, None] * Q * dih[None, :]
        Fq = Fm @ Qs @ Fm.conj().T  # the scaled operator in k
        dsym = (lam0 / lmax + 1e-3) ** -1.0 / lmax
        for kc in (3, 6):
            low = (np.hypot(KX, KY).ravel() <= kc)
            Pk = np.diag(np.where(low, 0, dsym)).astype(complex)
            il = np.flatnonzero(low)
            Pk[np.ix_(il, il)] = np.linalg.inv(Fq[np.ix_(il, il)])
            cands[f"2-level k: dense |k|<={kc} ({low.sum()})"] = dih[:, None] * (Fm.conj().T @ Pk @ Fm) * dih[None, :]
        for name, P in cands.items():
            kap, ev = kappa(Q, P)
            its = pcg(Q, b, lambda r_: P @ r_, xhat)
            log(f"  {name:28s} kappa {kap:9.3g}   PCG its to 1e-3 / 1e-6: {its[0]!s:>5} / {its[1]!s:>5}")
        # surrogate inverted inexactly: K inner circulant-PCG steps on Q~ (L = 4), flexible PCG outside
        Pc = dih[:, None] * circ((lam0 / lmax + 1e-2) ** -1.0) * dih[None, :]
        for Kin in (5, 20):
            def apply_inner(rhs, Kin=Kin):
                y = np.zeros_like(rhs); rr = rhs.copy(); z = Pc @ rr; p = z.copy(); rz = np.vdot(rr, z).real
                for _ in range(Kin):
                    Qp = Qt4 @ p; a = rz / np.vdot(p, Qp).real
                    y = y + a * p; rr = rr - a * Qp; z = Pc @ rr; rzn = np.vdot(rr, z).real
                    p = z + (rzn / rz) * p; rz = rzn
                return y
            its = pcg(Q, b, apply_inner, xhat, flexible=True)
            log(f"  {'seg L=4, ' + str(Kin) + ' inner PCG (flex)':28s} kappa {'-':>9}   PCG its to 1e-3 / 1e-6: {its[0]!s:>5} / {its[1]!s:>5}")
        # coil surrogate inverted inexactly (K inner steps, dcd inner metric): does the κ ≈ 60 survive?
        Pin = dih[:, None] * circ((lam0 / lmax + 1e-3) ** -1.0) * dih[None, :]
        for Kin in (10, 30):
            def apply_inner_c(rhs, Kin=Kin):
                y = np.zeros_like(rhs); rr = rhs.copy(); z = Pin @ rr; p = z.copy(); rz = np.vdot(rr, z).real
                for _ in range(Kin):
                    Qp = Qc @ p; a = rz / np.vdot(p, Qp).real
                    y = y + a * p; rr = rr - a * Qp; z = Pin @ rr; rzn = np.vdot(rr, z).real
                    p = z + (rzn / rz) * p; rz = rzn
                return y
            its = pcg(Q, b, apply_inner_c, xhat, flexible=True)
            log(f"  {'coil-seg, ' + str(Kin) + ' inner (flex)':28s} kappa {'-':>9}   PCG its to 1e-3 / 1e-6: {its[0]!s:>5} / {its[1]!s:>5}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbol_toy_results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
