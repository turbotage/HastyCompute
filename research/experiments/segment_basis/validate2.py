import sys, time, numpy as np
sys.path.insert(0, sys.argv[2])
import segeval as se
case = se.Case(sys.argv[1], conc=True, nr=48)
se.setup_shells(case)
ns = [10, 30, 50, 80, 120, 300, 600, 1990, 3700, 3850, 3930, 3960, 3985]
b = se.basis_fixed(case, 12)
A = b.a(case.tau[ns]).astype(np.complex64)
for q, n in enumerate(ns):
    d = case.g(case.tau[n]).astype(np.complex64) - A[:, q] @ b.D
    P = se.shell_power(case, case.rssx * d)
    for h in [30.0]:
        case.shell_h = h
        e = se.shell_at(case, P, case.kap[n])
    # 4 exact coils FFT shell for coil check (coil-sum of 4 coils x 32/4 scaling not meaningful) -> use all 32 coils at a few n
    if n in (80, 300, 3850):
        Pc = 0
        for c in range(case.nc):
            Pc = Pc + se.shell_power(case, case.SX[c] * d)
        ec = se.shell_at(case, Pc, case.kap[n])
    else:
        ec = float('nan')
    print(f'n={n} kap={case.kap[n]:.0f}: rss-FFT shell {e:.3e}   32-coil FFT shell {ec:.3e}', flush=True)
