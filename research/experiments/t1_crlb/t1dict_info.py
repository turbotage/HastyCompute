"""Information retained by a rank-K subspace (EPG dictionary SVD) for the inversion design:
sigma_R1 (1/s, per unit SNR_TR) of the CSF-Gd series vs K, with T2 free (rho,T1,B1,T2) and T2 known.
usage: python t1dict_info.py "<flips>" TR_ms Tp_s"""
import os, sys, subprocess, io
import numpy as np

BIN = "../../../rustlib/target/epg-check/release/t1ir"
GAP = 15e-3
IR = os.environ.get("IR", "ir")      # "none" for the no-prep design


def evals(rows, ir, segstr, env):
    inp = "\n".join(",".join(f"{x:.9g}" for x in r) for r in rows) + "\n"
    out = subprocess.run([BIN, "eval", ir, segstr, "csf"], input=inp, capture_output=True, text=True, check=True, env=env).stdout
    return np.loadtxt(io.StringIO(out), delimiter=",", ndmin=2)


def main():
    flips, tr, tp = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    nper = int((tp - (GAP if IR == "ir" else 0.0)) / (tr * 1e-3)) // len(flips.split(","))
    segstr = ",".join(f"{f}:{tr}:{nper}" for f in flips.split(","))
    env = dict(os.environ, INCS="117")
    # dictionary (T2 free)
    txt = subprocess.run([BIN, "dict", IR, segstr, "csf"], capture_output=True, text=True, check=True, env=env).stdout
    A = np.loadtxt(io.StringIO(txt), delimiter=",")
    G = A[:, 3:].T
    Gn = G / np.linalg.norm(G, axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Gn, full_matrices=False)
    # cases: CSF-Gd series (T10 4.5, T20 2.0, r1 3.3, r2 5), B1 1.0 and 0.8
    grid = [0.3, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 2.8, 3.6, 4.5]
    cases = []
    for t1 in grid:
        c = max(0.0, (1 / t1 - 1 / 4.5) / 3.3)
        t2 = 1.0 / (1 / 2.0 + 5.0 * c)
        for b1 in (1.0, 0.8):
            cases.append((t1, t2, b1))
    h = 1e-4
    rows = []
    for t1, t2, b1 in cases:
        rows.append((t1, t2, b1))
        for i in range(3):
            for sgn in (+1, -1):
                v = [t1, t2, b1]
                v[i] *= 1 + sgn * h
                rows.append(tuple(v))
    S_all = evals(rows, IR, segstr, env)          # (ncases*7, N)
    Ks = [4, 6, 8, 10, 12, 16, 20, 28, 40, None]
    print(f"design IR T_p {tp} s flips {flips} TR {tr} ms; sigma_R1 [1/s per unit SNR_TR], geometric mean over B1 1.0/0.8, CSF-Gd series")
    print("  K   |" + "  ".join(f"{t:5.1f}" for t in grid) + "  | L4(sigma/tol)  [T2 free]  /  [T2 known]")
    res = {}
    for free_t2 in (True, False):
        for K in Ks:
            P = (lambda M: M) if K is None else (lambda M, K=K: U[:, :K] @ (U[:, :K].T @ M))
            sig = np.zeros((len(grid), 2))
            for ci, (t1, t2, b1) in enumerate(cases):
                base = ci * 7
                s0 = S_all[base]
                # derivatives wrt ln rho, ln T1, ln B1, ln T2 ; rho derivative = signal itself ; T2 derivative from FD on T2 (index 2 below)
                d_rho = s0
                d_t1 = (S_all[base + 1] - S_all[base + 2]) / (2 * h)
                d_t2 = (S_all[base + 3] - S_all[base + 4]) / (2 * h)
                d_b1 = (S_all[base + 5] - S_all[base + 6]) / (2 * h)
                cols = [d_rho, d_t1, d_b1] + ([d_t2] if free_t2 else [])
                D = np.stack([P(c[:, None])[:, 0] for c in cols], axis=1)
                reps = 480.0 / tp
                J = D.T @ D * reps
                C = np.linalg.inv(J)
                sig[ci // 2, ci % 2] = np.sqrt(C[1, 1]) / grid[ci // 2]
            g = np.exp(np.mean(np.log(sig), axis=1))
            tol = 0.02 + 0.05 / np.array(grid)
            L4 = float(np.mean((sig / tol[:, None]) ** 4) ** 0.25)
            res[(free_t2, K)] = (g, L4)
            print(f"{'full' if K is None else K:>5} |" + "  ".join(f"{v:5.3f}" for v in g) + f"  | {L4:6.2f}  {'T2 free' if free_t2 else 'T2 known'}")
    return res


if __name__ == "__main__":
    main()
