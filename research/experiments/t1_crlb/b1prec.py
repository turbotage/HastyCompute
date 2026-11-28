"""sigma of ln B1 (relative B1 precision, unit SNR_TR, ideal spoiling) from the T1 designs themselves, B1 free, and how much of it a smooth B1 model could average.
usage: GRID=narrow python b1prec.py"""
import numpy as np
import t1crlb_vs_lit as v
import t1crlb_wide as w

def sigma_b1(segs, prep, ts):
    cases = [(t, b) for t in w.T1_GRID for b in w.B1S]
    nc = len(cases)
    base = np.array([[1.0, t, b] for t, b in cases]).T
    TH, hs = [], []
    for i in range(3):
        h = 1e-6 * base[i]
        tp, tm = base.copy(), base.copy(); tp[i] += h; tm[i] -= h
        TH += [tp, tm]; hs.append(h)
    S, dur = w.sim(*np.concatenate(TH, axis=1)[:3], segs, prep)
    out = np.zeros(nc)
    for c in range(nc):
        D = np.stack([(S[:, 2 * k * nc + c] - S[:, (2 * k + 1) * nc + c]) / (2 * hs[k][c]) * base[k, c] for k in range(3)], axis=1)
        J = D.T @ D * ts / dur
        out[c] = np.sqrt(np.linalg.inv(J)[2, 2])
    return np.exp(np.mean(np.log(out.reshape(len(w.T1_GRID), 2)), axis=1))

lit = (w.nseg([2, 12], [7.2, 7.2], 20.0), False, 750.0)
n = int((12.0 - w.T_GAP) / 0.020) // 8
ours = ([(np.deg2rad(a), 0.020, n) for a in [13.1, 11.6, 3, 30, 3, 3, 3, 3]], True, 480.0)
print("sigma(ln B1) per unit SNR_TR, B1 free, by T1:", w.T1_GRID)
for name, d in (("literature 2/12 deg", lit), ("ours IR", ours)):
    print(f"  {name}: " + " ".join(f"{t:.1f}:{x:.4f}" for t, x in zip(w.T1_GRID, sigma_b1(*d))))
