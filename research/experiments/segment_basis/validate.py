import sys, time, numpy as np
sys.path.insert(0, sys.argv[2])
import segeval as se
case = se.Case(sys.argv[1], conc=True, nr=48)
ns = [10, 30, 50, 80, 120, 300, 600, 1200, 1990, 2800, 3400, 3700, 3850, 3930, 3960, 3985]
for kind, L in [('fixed', 12), ('svd', 12)]:
    b = se.basis_fixed(case, L) if kind == 'fixed' else se.basis_svd(case, L)
    A = b.a(case.tau[ns]).astype(np.complex64)
    for q, n in enumerate(ns):
        t = time.time()
        d = case.g(case.tau[n]).astype(np.complex64) - A[:, q] @ b.D
        white = (case.wE * np.abs(d) ** 2).sum()
        Z = case.SX * d[None, :]  # nc x nf
        Ex = 0.0
        for r0 in range(0, 48, 8):
            k = case.kr[r0:r0 + 8, n, :]
            P = np.exp(-1j * (case.pos @ k.T)).astype(np.complex64)  # nf x 8
            F = Z @ P
            Ex += (np.abs(F) ** 2).sum()
        Ex /= 48
        # block approx
        bimg = (case.Bmat @ Z.T).T
        Fb = bimg @ case.phase(n)
        Eb = (np.abs(Fb) ** 2).sum(0).mean()
        print(f'{kind} L={L} n={n} kap={case.kap[n]:.0f}: exact {Ex:.3e}  block {Eb:.3e} ({Eb/Ex:.2f})  white {white:.3e} ({white/Ex:.2f})  [{time.time()-t:.1f}s]', flush=True)
