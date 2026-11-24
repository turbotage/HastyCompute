"""Leave-one-out pooled averages + voxelwise unmixing (user's idea, 2026-11-22) on the exact toy of toy.py.

N frames (degree-0 bins). For each m: pooled solve over all frames except m -> x_hat_{-m}. Voxelwise scalar unmixing
with the data weights w_j assumes x_hat_S = sum_{j in S} w_j x_j / w_S. Compared with:
  exact per-frame least squares (lightly regularised), the pooled mean, and
  'mean + residual deconvolution': x_k = xbar + (wbar Nbar)^-1 A_k^H (y_k - A_k xbar)  (the first-order equivalent),
and the first-order bias prediction  b_k = -(wbar Nbar)^-1 Delta_k (x_k - xbar),  Delta_k = N_k - w_k Nbar.
Outputs a text summary next to this file.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy import make_problem, line_normals, simulate

def run(stationary, NB, sigma, seed, C=4, n=8, R=480, dens_pow=1.8, log=print):
    P = make_problem(n=n, C=C, R=R, stationary=stationary, sigma=sigma, seed=seed, dens_pow=dens_pow, harmonics=6)
    Es, Ops = line_normals(P)
    ys = simulate(P, Ops)
    npx = n * n
    bins = np.minimum((P["t"] * NB).astype(int), NB - 1)
    # truth per bin = mean of the truth over the bin's sample times (what a degree-0 frame can represent)
    X = P["truth"](P["t"])
    xt = np.stack([X[bins == k].mean(0) for k in range(NB)])
    Nk, bk = [], []
    for k in range(NB):
        idx = np.flatnonzero(bins == k)
        N = np.zeros((npx, npx), complex); b = np.zeros(npx, complex)
        for r_ in idx:
            op = Ops[0] if P["line"] is None else Ops[P["line"][r_]]
            N += op.conj().T @ op; b += op.conj().T @ ys[r_]
        Nk.append(N); bk.append(b)
    Nk, bk = np.array(Nk), np.array(bk)
    w = np.array([np.trace(N).real for N in Nk])          # data weight per frame (encoding power)
    Ntot, btot = Nk.sum(0), bk.sum(0)
    Nbar = Ntot / w.sum()                                  # mean normal operator per unit weight
    eps = 1e-6 * np.trace(Ntot).real / npx
    I = np.eye(npx)
    err = lambda xs: np.linalg.norm(xs - xt) / np.linalg.norm(xt)
    # pooled mean and exact per-frame LS
    xbar = np.linalg.solve(Ntot + eps * I, btot)
    xpf = np.stack([np.linalg.solve(Nk[k] + eps * I, bk[k]) for k in range(NB)])
    # leave-one-out pooled solves
    loo = np.stack([np.linalg.solve(Ntot - Nk[m] + eps * I, btot - bk[m]) for m in range(NB)])
    # voxelwise scalar unmixing: loo[m] = sum_{j != m} w_j x_j / (W - w_m)  ->  M x = loo, M[m, j] = w_j [j != m] / (W - w_m)
    W = w.sum()
    M = np.array([[(w[j] if j != m else 0.0) / (W - w[m]) for j in range(NB)] for m in range(NB)])
    xu = np.linalg.solve(M, loo)
    # first-order equivalent: mean + mean-operator deconvolution of the frame residual backprojection
    wbar = w.mean()
    xr = np.stack([xbar + np.linalg.solve(wbar * Nbar + eps * I, bk[k] - Nk[k] @ xbar) for k in range(NB)])
    # first-order bias prediction for the unmixed estimate (noise-free meaning), with the true frames
    # exact bias of the residual-deconvolution estimator: est - x_k = ((wbar Nbar)^-1 N_k - I)(x_k - xbar), with xbar the
    # noise-free pooled solution (the H-weighted mean); first order in Delta_k = N_k - wbar Nbar: (wbar Nbar)^-1 Delta_k (x_k - xbar)
    xbn = np.linalg.solve(Ntot + eps * I, sum(Nk[j] @ xt[j] for j in range(NB)))
    pred = np.stack([xt[k] + np.linalg.solve(wbar * Nbar + eps * I, (Nk[k] - wbar * Nbar) @ (xt[k] - xbn)) for k in range(NB)])
    # how large the relative frame-vs-mean operator deviation is (spectral norm of (wbar Nbar)^-1 N_k - I on the range)
    dev = np.median([np.linalg.norm(np.linalg.solve(wbar * Nbar + eps * I, Nk[k]) - I, 2) for k in range(NB)])
    log(f"  {'full encoding' if stationary else 'line-sampled '} N={NB:2d} sigma={sigma}: cond(M) {np.linalg.cond(M):.1f}; truth err: "
        f"per-frame LS {err(xpf):.3f}, pooled mean {err(np.broadcast_to(xbar, xt.shape)):.3f}, LOO+unmix {err(xu):.3f}, "
        f"mean+resid-deconv {err(xr):.3f}; |LOO+unmix - first-order prediction| / |x| = {np.linalg.norm(xu - pred) / np.linalg.norm(xt):.3f}"
        f" (vs |bias| {np.linalg.norm(xu - xt) / np.linalg.norm(xt):.3f}); median ||(wbar Nbar)^-1 N_k - I|| = {dev:.2f}")
    return dict(perframe=err(xpf), mean=err(np.broadcast_to(xbar, xt.shape)), loo=err(xu), resid=err(xr))

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    lines = []
    def log(s):
        print(s); lines.append(s); sys.stdout.flush()
    log("LOO pooled averages + voxelwise unmixing vs per-frame LS (8x8 image, 4 coils, truth = bin means)")
    for stat in (True, False):
        for NB in (6, 12, 24):
            for sigma in (0.0, 0.08):
                run(stat, NB, sigma, seed=7, log=log)
    with open(os.path.join(here, "loo_results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
