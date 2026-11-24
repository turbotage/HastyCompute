"""numpy port of hasty_recon::smapfit::fit_maps (same basis, penalty, scaling, CG) run on the Rust unit test's case
(fills_holes_and_removes_fine_structure), to check the test's assertions without a cargo build."""
import numpy as np


def axis(n, vox, h):
    span = int(np.floor((n - 1) * vox / h)); k = span + 4
    u = np.minimum(np.arange(n) * vox / h, span + 1 - 1e-9); j = np.minimum(np.floor(u).astype(int), span); t = u - j
    w = np.stack([(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6], 1)
    B = np.zeros((n, k))
    for d in range(4):
        B[np.arange(n), j + d] = w[:, d]
    return B


def ev(c, Bs):  # c [kx,ky,kz]
    return np.einsum("xi,yj,zk,ijk->xyz", Bs[0], Bs[1], Bs[2], c, optimize=True)


def adj(v, Bs):
    return np.einsum("xi,yj,zk,xyz->ijk", Bs[0], Bs[1], Bs[2], v, optimize=True)


def reg(c):
    out = np.zeros_like(c)
    for a in range(3):
        if c.shape[a] < 3:
            continue
        cm = np.moveaxis(c, a, 0); om = np.moveaxis(out, a, 0)
        d = cm[:-2] - 2 * cm[1:-1] + cm[2:]
        om[:-2] += d; om[1:-1] -= 2 * d; om[2:] += d
    return out


def fit(maps, w, dims, vox, h, lam, iters=400, tol=1e-7):
    Bs = [axis(dims[a], vox[a], h) for a in range(3)]
    diag = adj(w.astype(float), [B ** 2 for B in Bs])
    dbar = diag[diag > 0].mean(); l, mu = lam * dbar, 1e-6 * dbar
    N = lambda c: adj(w * ev(c, Bs), Bs) + l * reg(c) + mu * c
    out = []
    for m in maps:
        b = adj(w * m, Bs); x = np.zeros_like(b); r = b.copy(); p = r.copy(); rr = np.vdot(r, r).real; r0 = rr
        for it in range(iters):
            Ap = N(p); al = rr / np.vdot(p, Ap).real; x += al * p; r -= al * Ap; rn = np.vdot(r, r).real
            if np.sqrt(rn / r0) < tol:
                break
            p = r + rn / rr * p; rr = rn
        out.append(ev(x, Bs))
    return np.array(out), it + 1


dims = (40, 36, 24); vox = (0.005, 0.005, 0.006); nc = 6
P = np.stack(np.meshgrid(*[np.arange(n) for n in dims], indexing="ij"), -1)
U = (P - np.array([n // 2 for n in dims])) / np.array(dims)
s = []
for c in range(nc):
    th = 2 * np.pi * c / nc; cx, cy = 0.45 * np.cos(th), 0.45 * np.sin(th)
    r2 = (U[..., 0] - cx) ** 2 + (U[..., 1] - cy) ** 2 + 0.5 * U[..., 2] ** 2
    s.append(np.exp(-r2 / 0.2) * np.exp(1j * (1.2 * (U[..., 0] * np.sin(th) - U[..., 1] * np.cos(th)) + 0.4 * c)))
s = np.array(s)
tex = np.where(((P[..., 0] // 2 + P[..., 1] // 2 + P[..., 2] // 2) % 2) == 0, 1.03, 0.97)
hole = ((P[..., 0] - 14.0) ** 2 + (P[..., 1] - 18.0) ** 2 + (P[..., 2] - 12.0) ** 2) < 36
w = np.where(hole, 0.0, 1.0)
m = s * tex * (~hole)
f, its = fit(m, w, dims, vox, 0.03, 1e-3)
dis = lambda a, b: 1 - np.abs(np.sum(np.conj(a) * b, 0)) / np.sqrt(np.sum(np.abs(a) ** 2, 0) * np.sum(np.abs(b) ** 2, 0))
e = dis(f, s)
mi = np.sum(np.abs(m - s)[:, ~hole] ** 2) / np.sum(np.abs(s) ** 2); mf = np.sum(np.abs(f - s)[:, ~hole] ** 2) / np.sum(np.abs(s) ** 2)
print(f"CG its {its}; outside the hole e {e[~hole].mean():.2e} (assert < 2e-3); inside {e[hole].mean():.2e} (assert < 2e-2); "
      f"rel. sq. error input {mi:.2e}, fit {mf:.2e} (assert fit < 0.1 input: {mf < 0.1 * mi})")
