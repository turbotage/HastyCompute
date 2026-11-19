"""Direction 1 on the real trajectory: per-readout interpolative decomposition (time decimation with the exact LS
metric W = E^H E). For encode-0 readouts and each case (k scale = resolution proxy, ROI cube / ball of the box):
  - eps-rank of the readout's range (SVD of a point-source sketch; uniform sources = white image, and
    signal-weighted ratemap sources): the lower bound of any compression of this readout,
  - warped-time subset (os samples per Nyquist unit of the local band, <= 8 samples apart) with a local q-tap LS
    interpolator E trained on a uniform + band-edge sketch: relative error ||V - E V_S|| on fresh test sources
    (signal-weighted ratemap set and band-edge worst-case set; Frobenius and worst source), |S|, n / |S|,
  - normal-operator error on the test span: || V^H V - V_S^H W V_S ||_2 / || V^H V ||_2,
  - pyramid level 4 mm: the reduction among samples with |k| <= 125 cycles/m (native scale).

run (one case, readouts i0..i1 of the nro drawn; results accumulate in out/id_readouts.json):
  flock .. systemd-run .. rustlib/.venv/bin/python -I research/experiments/sample_compress/id_readouts.py nro nsvd case i0 i1
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_common as sc  # noqa: E402

OUT = os.path.join(HERE, "out")
CASES = [  # name, k scale, roi, half-width (m)
    ("box160_2mm_cube", 1.0, "cube", 0.08),
    ("box160_2mm_ball", 1.0, "ball", 0.08),
    ("box160_1mm_cube", 2.0, "cube", 0.08),
    ("box160_1mm_ball", 2.0, "ball", 0.08),
    ("box128_1mm_cube", 2.0, "cube", 0.064),
    ("box160_1mm_cube_dwell1us", 2.0, "cube", 0.08, 2),    # 1 mm with half the dwell (60 M samples per encode)
    ("box160_1mm_ball_dwell1us", 2.0, "ball", 0.08, 2),
]
OSS = [1.25, 1.35, 1.5, 2.0]
QS = [24, 32]


def save(res, cres, name, path, t0):
    """summary per (os, q): mean reduction, worst errors over the readouts; written after every readout"""
    summ = {}
    for g0 in cres["per_readout"][0]["grid"]:
        key = f"os{g0['os']}_q{g0['q']}"
        ents = [g for rr in cres["per_readout"] for g in rr["grid"] if g["os"] == g0["os"] and g["q"] == g0["q"]]
        summ[key] = {"reduction_mean": float(np.mean([e["reduction"] for e in ents])),
                     "pyr4mm_reduction_mean": float(np.mean([e["pyr4mm_reduction"] for e in ents]))}
        for f in ("err_signal_frob", "err_signal_max", "err_edge_frob", "err_edge_max", "normal_err_signal", "normal_err_edge"):
            if f in ents[0]:
                summ[key][f + "_worst"] = float(max(e[f] for e in ents))
    cres["summary"] = summ
    cres["landau_redundancy_mean"] = float(np.mean([rr["n"] / rr["landau"] for rr in cres["per_readout"]]))
    res["cases"][name] = cres
    res["seconds_" + name] = time.time() - t0
    json.dump(res, open(path, "w"), indent=1)


def main():
    nro = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    nsvd = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    i0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    i1 = int(sys.argv[5]) if len(sys.argv) > 5 else nro
    rng = np.random.default_rng(7)
    ro_all = np.arange(0, sc.META["n_readouts"], 7)
    ros = np.sort(rng.choice(ro_all, nro, replace=False))
    rds = sc.load_readouts(ros)
    path = os.path.join(OUT, "id_readouts.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.update({"readouts": ros.tolist(), "centre": sc.BOX_CENTRE.tolist(), "oss": OSS, "qs": QS})
    res.setdefault("cases", {})
    t0 = time.time()
    for case in CASES:
        name, scale, roi, half = case[:4]
        up = case[4] if len(case) > 4 else 1
        if only and name not in only:
            continue
        wb = sc.offres_band(half, sc.BOX_CENTRE)
        cres = res["cases"].get(name, {"per_readout": []})
        cres["wband_hz"] = [w / 2 / np.pi for w in wb]
        done = {rr["readout"] for rr in cres["per_readout"]}
        for i, rd in enumerate(rds):
            if not (i0 <= i < i1) or int(ros[i]) in done:
                continue
            if up > 1:
                rd = sc.upsample(rd, up)
            n = len(rd[1])
            B, dwell = sc.band(rd, roi, half, scale, wb)
            landau = float((B * dwell / (2 * np.pi)).sum())
            kc = np.linalg.norm(rd[0], axis=1) / (2 * np.pi) <= 125.0          # pyramid 4 mm level (native k)
            r = {"readout": int(ros[i]), "n": n, "landau": landau, "n_pyr4mm": int(kc.sum())}
            Vtr = sc.train_columns(rd, 100 + 7 * i, 1800, roi, half, scale, wb)
            Vsig = sc.columns(rd, np.random.default_rng(900 + i), 400, roi, half, sc.BOX_CENTRE, scale, wb, mode="signal")
            Vedg = sc.columns(rd, np.random.default_rng(950 + i), 400, roi, half, sc.BOX_CENTRE, scale, wb, mode="edge")
            if i < nsvd:
                r["eps_rank"] = {}
                for mode in ("uni", "signal"):
                    Vs = sc.columns(rd, np.random.default_rng(500 + i), int(min(1.6 * landau + 600, 6000)), roi, half,
                                    sc.BOX_CENTRE, scale, wb, mode=mode)
                    sv = np.linalg.svd(Vs, compute_uv=False)
                    tail = np.sqrt(np.cumsum((sv ** 2)[::-1])[::-1] / (sv ** 2).sum())
                    r["eps_rank"][mode] = {str(e): int(np.argmax(tail < e)) for e in (1e-3, 1e-4, 1e-5)}
                    del Vs
            grid = []
            for os_ in OSS:
                S = sc.select_warped(B, dwell, os_)
                for q in QS:
                    cols, coef = sc.local_E(Vtr, S, q)
                    fs, ms = sc.rel_err(Vsig, sc.apply_E(cols, coef, Vsig[S]))
                    fe, me = sc.rel_err(Vedg, sc.apply_E(cols, coef, Vedg[S]))
                    ent = {"os": os_, "q": q, "nS": int(len(S)), "reduction": n / len(S),
                           "pyr4mm_reduction": float(kc.sum() / max(1, kc[S].sum())),
                           "err_signal_frob": fs, "err_signal_max": ms, "err_edge_frob": fe, "err_edge_max": me}
                    if q == 32:
                        E = sc.E_dense(cols, coef, len(S))
                        Wm = E.conj().T @ E
                        G = Vsig.conj().T @ Vsig
                        Gw = Vsig[S].conj().T @ Wm @ Vsig[S]
                        ent["normal_err_signal"] = float(np.linalg.norm(G - Gw, 2) / np.linalg.norm(G, 2))
                        G = Vedg.conj().T @ Vedg
                        Gw = Vedg[S].conj().T @ Wm @ Vedg[S]
                        ent["normal_err_edge"] = float(np.linalg.norm(G - Gw, 2) / np.linalg.norm(G, 2))
                    grid.append(ent)
            r["grid"] = grid
            cres["per_readout"].append(r)
            save(res, cres, name, path, t0)
            ok = [g for g in grid if g["err_edge_max"] < 1e-4]
            oks = [g for g in grid if g["err_signal_max"] < 1e-4]
            bb = max(ok, key=lambda g: g["reduction"]) if ok else None
            bs = max(oks, key=lambda g: g["reduction"]) if oks else None
            print(f"{name} ro {ros[i]}: n/landau {n/landau:.2f} eps-rank {r.get('eps_rank')} best edge<1e-4 "
                  f"{bb and (bb['os'], bb['q'], round(bb['reduction'], 2))} signal<1e-4 {bs and (bs['os'], bs['q'], round(bs['reduction'], 2))}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
        if cres["per_readout"]:
            save(res, cres, name, path, t0)


if __name__ == "__main__":
    main()
