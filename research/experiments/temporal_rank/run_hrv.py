"""Run 6: heart-rate variability (research/temporal_rank.md). Beats of different length RR_j = RR (1 + s z_j) (z_j the
equal-probability normal quantiles of J beats) keep systole in absolute time (end-systole t_es = 0.33 s) and stretch
diastole; on the recon's normalised cardiac phase each beat is the truth at a warped time, so the beats differ and a
phase-only model fits their mean. Reported per s: the rank of the beat-mean signal, the floor (each beat's distance from
the mean, which no phase basis removes), and the rank of the stacked beats.
Voxels: lumen / vessel voxels plus a 1-in-4 random subset of the rest (Gram weight 4).

    rustlib/.venv/bin/python run_hrv.py [s ...]
"""
import json
import pickle
import sys
import time

import numpy as np

from common import CACHE, LEVELS, OUT, decode, encode_matrix, k_at
from experiment import eig_desc
from rank_lib import Fields, box2

T_ES = 0.33
J = 8


def main():
    sv = [float(x) for x in sys.argv[1:]] or [0.03, 0.06, 0.10]
    t0 = time.time()
    F = Fields()
    b2, cat, vmap = pickle.loads((CACHE / "context.pkl").read_bytes())
    sm = np.load(CACHE / "smooth_mag.npy")  # continuous-motion magnitude (run_regimes)
    RR = 0.8528221101760362
    ev = np.zeros(F.nv, bool)
    for per in vmap.values():
        for ci, w, t in per:
            ev[ci] = True
    keep = (b2.max(1) >= 0.25) | ev
    rng = np.random.default_rng(2)
    rest = np.flatnonzero(~keep)
    vox = np.sort(np.concatenate([np.flatnonzero(keep), rest[rng.uniform(size=len(rest)) < 0.25]]))
    wv = np.where(keep[vox], 1.0, 4.0)
    lum = (b2[vox].mean(1) >= 0.5)
    K = encode_matrix()
    from scipy.stats import norm
    z = norm.ppf((np.arange(J) + 0.5) / J)
    out = {"t_es": T_ES, "J": J, "levels": LEVELS, "runs": []}
    for s in sv:
        X = np.zeros((J, 7, len(vox), F.nf), np.complex64)
        for j in range(J):
            RRj = RR * (1 + s * z[j])
            for i in range(F.nf):
                t = i / F.nf * RRj
                tr = t if t < T_ES else T_ES + (t - T_ES) * (RR - T_ES) / (RRj - T_ES)
                if RRj < T_ES * 1.2:
                    tr = t / RRj * RR
                fpos = tr / RR * F.nf
                f0 = int(np.floor(fpos)) % F.nf
                f1 = (f0 + 1) % F.nf
                a = fpos - np.floor(fpos)
                m = (1 - a) * sm[f0].astype(np.float32) + a * sm[f1].astype(np.float32)
                v = (1 - a) * F.v[f0].astype(np.float32) + a * F.v[f1].astype(np.float32)
                for e in range(7):
                    ph = K[e, 0] * v[0] + K[e, 1] * v[1] + K[e, 2] * v[2]
                    X[j, e, :, i] = box2(m * np.exp(-1j * ph).astype(np.complex64)).reshape(-1)[vox]
            print(f"s {s} beat {j} RR {RRj:.3f} {time.time() - t0:.0f} s", flush=True)
        sw = np.sqrt(wv)[:, None]
        Xm = X.mean(0)
        G = sum((Xm[e] * sw).conj().T @ (Xm[e] * sw) for e in range(7)).astype(np.complex128)
        w, U = eig_desc(G)
        nb = F.nf
        Gs = np.zeros((J * nb, J * nb), np.complex128)
        for e in range(7):
            Y = [X[j, e] * sw for j in range(J)]
            for j in range(J):
                for k in range(j, J):
                    blk = Y[j].conj().T @ Y[k]
                    Gs[j * nb:(j + 1) * nb, k * nb:(k + 1) * nb] += blk
                    if k != j:
                        Gs[k * nb:(k + 1) * nb, j * nb:(j + 1) * nb] += blk.conj().T
        ws, _ = eig_desc(Gs)
        num = sum(((np.abs(X[j] - Xm) ** 2).sum(-1) * wv).sum() for j in range(J))
        den = J * ((np.abs(Xm) ** 2).sum(-1) * wv).sum()
        numb = sum(((np.abs(X[j][:, lum] - Xm[:, lum]) ** 2).sum(-1)).sum() for j in range(J))
        denb = J * ((np.abs(Xm[:, lum]) ** 2).sum(-1)).sum()
        vm = decode(np.moveaxis(Xm[:, lum], 0, -1))
        dv = np.concatenate([(np.linalg.norm(decode(np.moveaxis(X[j][:, lum], 0, -1)) - vm, axis=-1) * 100).ravel() for j in range(J)])
        r = {"s": s, "RR_j": (RR * (1 + s * z)).tolist(), "K_mean": k_at(w, LEVELS), "K_stacked": k_at(ws, LEVELS), "spectrum_mean": (w / w[0]).tolist(),
             "spectrum_stacked": (ws / ws[0]).tolist(), "floor_rel_all": float(np.sqrt(num / den)), "floor_rel_lumen": float(np.sqrt(numb / denb)),
             "floor_dv_mean": float(dv.mean()), "floor_dv_p99": float(np.percentile(dv, 99))}
        out["runs"].append(r)
        print({k: v for k, v in r.items() if not k.startswith("spectrum")}, flush=True)
        del X
    (OUT / "hrv.json").write_text(json.dumps(out, indent=1))
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
