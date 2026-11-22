"""Where does the temporal rank come from? Temporal Fourier energy of the truth encode images by region, and the same for
the velocity-only and magnitude-only variants (research/temporal_rank.md, run 2).

    rustlib/.venv/bin/python diag_regions.py
"""
import json
import pickle

import numpy as np

from common import CACHE, LEVELS, OUT, k_at
from experiment import eig_desc
from rank_lib import Fields, box2, build_X, tissues

H_CUT = [4, 8, 12, 16, 24]


def regions(F):
    """per 2 mm voxel the time-majority region: chamber blood, vessel blood, myocardium (+ valve), other tissue, air"""
    tn = tissues()
    groups = {"chamber blood": ["blood_LA", "blood_LV", "blood_RA", "blood_RV"], "vessel blood": ["blood_artery", "blood_vein", "blood_pulm_artery", "blood_pulm_vein"],
              "myocardium": ["myocardium", "valve"], "lung/air": ["lung", "air", "airway"]}
    lut = np.full(len(tn), 3, np.int8)  # other tissue
    names = ["chamber blood", "vessel blood", "myocardium", "other tissue", "lung/air"]
    for gi, (g, ls) in enumerate(groups.items()):
        for n in ls:
            lut[tn.index(n)] = [0, 1, 2, 4][gi]
    cnt = np.zeros((F.nv, 5), np.float32)
    for f in range(F.nf):
        r = lut[F.lab[f]]
        for k in range(5):
            cnt[:, k] += box2((r == k).astype(np.float32)).reshape(-1)
    return cnt.argmax(1), names


def hf_table(X, reg, names):
    """fraction of each region's energy (all encodes) above temporal harmonic h, and its share of the image's total
    energy above h; plus the region's share of the total energy"""
    Fx = np.fft.fft(X, axis=2)
    T = X.shape[2]
    h = np.minimum(np.arange(T), T - np.arange(T))
    E = (np.abs(Fx) ** 2).sum(0)  # [V, T]
    tot_h = {hc: E[:, h > hc].sum() for hc in H_CUT}
    out = {}
    for k, n in enumerate(names):
        m = reg == k
        e = E[m]
        out[n] = {"voxels": int(m.sum()), "energy_share": float(e.sum() / E.sum()),
                  "frac_above": {hc: float(e[:, h > hc].sum() / e.sum()) for hc in H_CUT},
                  "share_of_hf": {hc: float(e[:, h > hc].sum() / tot_h[hc]) for hc in H_CUT}}
    return out


def spectra(X, mask=None):
    Xm = X if mask is None else X[:, mask]
    G = sum(Xm[e].conj().T @ Xm[e] for e in range(X.shape[0])).astype(np.complex128)
    w, _ = eig_desc(G)
    return w


if __name__ == "__main__":
    F = Fields()
    reg, names = regions(F)
    res = {}
    mbar = F.m.astype(np.float32).mean(0)
    for tag, kw in [("truth", {}), ("velocity only (magnitude frozen at its time mean)", {"mag": lambda f: mbar}), ("magnitude only (no velocity phase)", {"vel": lambda f: np.zeros((3,) + F.fs, np.float32)})]:
        X = build_X(F, **kw)
        r = {"hf": hf_table(X, reg, names)}
        w = spectra(X)
        r["K_all"] = k_at(w, LEVELS)
        r["spectrum_all"] = (w / w[0]).tolist()
        for k, n in enumerate(names):
            wk = spectra(X, reg == k)
            r[f"K {n}"] = k_at(wk, LEVELS)
        wnc = spectra(X, reg != 0)
        r["K without chamber blood"] = k_at(wnc, LEVELS)
        r["spectrum without chamber blood"] = (wnc / wnc[0]).tolist()
        res[tag] = r
        print(tag, {k: v for k, v in r.items() if k.startswith("K")}, flush=True)
        for n, d in r["hf"].items():
            print(f"   {n:14s} vox {d['voxels']:6d} energy {d['energy_share']:.3f} above h: " + " ".join(f"{hc}:{d['frac_above'][hc]:.2e}" for hc in H_CUT) + " | share of hf: " + " ".join(f"{hc}:{d['share_of_hf'][hc]:.2f}" for hc in H_CUT), flush=True)
        del X
    (OUT / "diag_regions.json").write_text(json.dumps(res, indent=1))
