"""Per-cell condition numbers of the per-set data step (N_t + rho I) under three diagonal-in-k metrics, from the
measured per-frame cell counts n_t(k) of one (encode 0, resp bin 0) group (sampling_stats.py setup). Coils and
off-grid coupling ignored: N_t ~ diag_k n_t(k). Metrics: identity, the circulant (smoothed density mu(k) = pooled/T,
3x3x3 box) without floor rho (the current metric's form: 1/(mu + eps)), and the circulant with the floor rho."""
import numpy as np, json, os, sys
sys.argv = [sys.argv[0]]
here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "sampling_stats.py")).read().split("res = {}")[0])  # setup only
def uniform_filter(a, n, mode="wrap"):
    out = a.copy()
    for ax in range(a.ndim):
        acc = np.zeros_like(out)
        for d in range(-(n // 2), n // 2 + 1):
            acc += np.roll(out, d, axis=ax)
        out = acc / n
    return out
e, b = 0, 0
sel = np.flatnonzero((enc == e) & (rb == b))
pooled = np.zeros(ncell, np.int64); perf = np.zeros(T * ncell, np.int32)
for i0 in range(0, len(sel), 500):
    s = sel[i0:i0 + 500]
    kk = np.einsum("rab,sb->rsa", rot[s], kb)
    keep = ((kk * dx) ** 2).sum(-1) <= 0.25
    ci = np.round(kk * fov).astype(int) % dims
    lin = ((ci[..., 0] * dims[1] + ci[..., 1]) * dims[2] + ci[..., 2])[keep]
    fr = np.broadcast_to(frame[s][:, None], keep.shape)[keep]
    pooled += np.bincount(lin, minlength=ncell); perf += np.bincount(fr * ncell + lin, minlength=T * ncell).astype(np.int32)
perf = perf.reshape(T, ncell)[:, inball].astype(float)
mu = uniform_filter(np.fft.fftshift((pooled / T).reshape(dims).astype(float)), 3, mode="wrap")
mu = np.fft.ifftshift(mu).ravel()[inball]
out = []
def q(a, p): return float(np.percentile(a, p))
for rho in (0.0, 0.1, 0.3, 1.0, 3.0):
    lam_id = perf + rho
    lam_c0 = perf / (mu + 1e-3 * mu.max()) if rho == 0 else None
    lam_cr = (perf + rho) / (mu + rho)
    row = dict(rho=rho)
    # spectrum of P(N_t + rho) per (frame, cell); kappa from the 99.9th / 0.1th percentiles (min of the identity case is rho)
    row["kappa_identity"] = (q(lam_id, 99.9) / max(rho, 1e-12)) if rho > 0 else float("inf")
    row["kappa_circ_floor_rho"] = q(lam_cr, 99.9) / q(lam_cr, 0.1) if rho > 0 else float("inf")
    if lam_c0 is not None:
        nz = lam_c0[lam_c0 > 0]
        row["circ_eps1e-3_nonzero_range"] = [q(nz, 0.1), q(nz, 99.9)]
    out.append(row); print(row); sys.stdout.flush()
json.dump(out, open(os.path.join(here, "cell_kappa.json"), "w"), indent=1)

# CG on the per-cell diagonal model: system diag(n_t(k) + rho) x = b, preconditioner diag(p(k)), random b;
# iterations to ||x - x*|| / ||x*|| <= 1e-3 (A-norm-free, plain error) for each metric
rng = np.random.default_rng(0)
def cg_iters(a, p, tol=1e-3, maxit=2000):
    b = rng.standard_normal(a.shape); xs = b / a
    x = np.zeros_like(b); r = b.copy(); z = p * r; d = z.copy(); rz = (r * z).sum()
    for it in range(1, maxit + 1):
        ad = a * d; al = rz / (d * ad).sum(); x += al * d; r -= al * ad
        if np.linalg.norm(x - xs) <= tol * np.linalg.norm(xs): return it
        z = p * r; rzn = (r * z).sum(); d = z + (rzn / rz) * d; rz = rzn
    return f">{maxit}"
mub = np.broadcast_to(mu, perf.shape)
rows = []
for rho in (0.1, 0.3, 1.0):
    a = perf + rho
    metrics = {
        "identity": np.ones_like(a),
        "circ 1/(mu + 1e-3 mu_max) (current form)": 1 / (mub + 1e-3 * mu.max()),
        "circ 1/(mu + rho) (floor = ridge)": 1 / (mub + rho),
        "floor at one sample: 1/max(mu, 1)": 1 / np.maximum(mub, 1.0),
        "1/(max(mu,1) + rho)": 1 / (np.maximum(mub, 1.0) + rho),
    }
    for name, p in metrics.items():
        it = cg_iters(a, p)
        rows.append((rho, name, it)); print(f"rho {rho}: {name:42s} CG its to 1e-3: {it}"); sys.stdout.flush()
json.dump([dict(rho=r, metric=n, its=i) for r, n, i in rows], open(os.path.join(here, "cell_cg.json"), "w"), indent=1)
