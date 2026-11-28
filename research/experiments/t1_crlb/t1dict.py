"""Rank of the EPG dictionary g_n(T1,T2,B1) of an inversion design (what K the subspace recon needs).
usage: python t1dict.py "<flips comma>" TR_ms Tp_s [inc]
Columns are normalised (rho free); error of the best rank-K subspace per column: max and rms (relative, L2 over the readouts)."""
import os, sys, subprocess, io
import numpy as np
import t1ir

BIN = "../../../rustlib/target/epg-check/release/t1ir"


def ranks(D, label, eps=(1e-2, 3e-3, 1e-3, 1e-4)):
    Dn = D / np.linalg.norm(D, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Dn, full_matrices=False)
    out = []
    for e in eps:
        K = None
        for k in range(1, min(len(S), 60) + 1):
            R = Dn - U[:, :k] @ (U[:, :k].T @ Dn)
            err = np.linalg.norm(R, axis=0)
            if err.max() <= e:
                K = (k, err.max(), np.sqrt((err ** 2).mean()))
                break
        out.append((e, K))
    print(f"  {label}: {D.shape[1]} entries, sing.values 1..10: {np.round(S[:10] / S[0], 5)}")
    print("     K for max column error <= " + ", ".join(f"{e:g}: {K[0] if K else '>60'}" for e, K in out))
    return S


def main():
    flips = sys.argv[1]
    tr, tp = float(sys.argv[2]), float(sys.argv[3])
    inc = sys.argv[4] if len(sys.argv) > 4 else "117"
    segstr = ",".join(f"{f}:{tr}:{int((tp - 15e-3) / (tr * 1e-3)) // len(flips.split(','))}" for f in flips.split(","))
    segs = [(np.deg2rad(float(f)), float(t) * 1e-3, int(n)) for f, t, n in (b.split(":") for b in segstr.split(","))]
    env = dict(os.environ, INCS=inc)
    txt = subprocess.run([BIN, "dict", "ir", segstr, "csf"], capture_output=True, text=True, check=True, env=env).stdout
    A = np.loadtxt(io.StringIO(txt), delimiter=",")
    T1, T2, B1, G = A[:, 0], A[:, 1], A[:, 2], A[:, 3:].T            # G: (readouts, entries)
    print(f"design IR T_p {tp} s flips {flips} TR {tr} ms, increment {inc}: {G.shape[0]} readouts per period")
    ranks(G, "EPG, T1 x T2 x B1 (T2 free, T2 <= 0.9 T1)")
    # T2 tied to R1 by the Gd model of the CSF series (T10 4.5, T20 2.0, r1 3.3, r2 5) -> 1-parameter family in (T1, B1)
    sel = np.array([abs(t2 - 1.0 / (1.0 / 2.0 + 5.0 * max(0.0, (1 / t1 - 1 / 4.5) / 3.3))) / t2 < 0.35 for t1, t2 in zip(T1, T2)])
    if sel.sum() > 10:
        ranks(G[:, sel], f"EPG, T2 near the CSF-Gd relation (subset)")
    # EPG, T2 fixed at 2 s and 0.1 s (only T1 x B1)
    for t2 in (2.4, 0.1):
        m = np.isclose(T2, t2)
        if m.sum() > 10:
            ranks(G[:, m], f"EPG, T2 = {t2} s fixed (T1 x B1)")
    # ideal-spoiling model (T1, B1)
    ent = [(t1, b1) for t1 in np.unique(T1) for b1 in np.unique(B1)]
    Gi = np.array([t1ir.ideal_model(1.0, t1, b1, segs, True) for t1, b1 in ent]).T
    ranks(Gi, "ideal spoiling, T1 x B1")


if __name__ == "__main__":
    main()
