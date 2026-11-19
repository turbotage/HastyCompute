"""One interpolator for the whole scan? The yarn readouts are rotations of one base readout (k_r = R_r k_0, same tau
grid), so |kdot| -- and with it the band of a *ball* ROI and the subset S -- is the same for every readout. With the
data demodulated to the box centre c (y_j <- y_j e^{+i k_j.c + i c_j.mono(c)}), the remaining signal of a source at
c + rho is e^{-i k_j.rho - z tau_j - i c_j.(mono(c + rho) - mono(c))} beta(u_j): rotation enters only through the small
concomitant cross terms and the cardiac weights, both inside the design band.

Test: E fitted on readout A only (centred sketch), applied to other readouts B (their own sources, demodulated);
error on B's signal-weighted and band-edge sets, compared with B's own E.

run: rustlib/.venv/bin/python -I research/experiments/sample_compress/proto_check.py [os] [q] [roi]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_common as sc  # noqa: E402

OUT = os.path.join(HERE, "out")


def demod_cols(rd, rng, P, mode, roi, half, wb, c):
    """columns of sources around c, demodulated by the per-sample phase of the centre"""
    k, tau, w, ph = rd
    r, z = sc.draw_sources(rng, P, mode, roi, half, c, wb)
    V = sc.signatures(rd, r, z, 1.0, True, True, rng)
    dem = np.exp(1j * (k @ c + w @ sc.mono(c[None, :])[0]))
    return V * dem[:, None]


def main():
    os_ = float(sys.argv[1]) if len(sys.argv) > 1 else 1.35
    q = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    roi = sys.argv[3] if len(sys.argv) > 3 else "ball"
    half, c = 0.08, sc.BOX_CENTRE
    wb = sc.offres_band(half, c)
    rng = np.random.default_rng(21)
    ros = np.sort(rng.choice(np.arange(0, sc.META["n_readouts"], 7), 5, replace=False))
    rds = sc.load_readouts(ros)
    A = rds[0]
    B, dwell = sc.band(A, roi, half, 1.0, wb)
    S = sc.select_warped(B, dwell, os_)
    Vtr = np.concatenate([demod_cols(A, np.random.default_rng(1), 900, "uni", roi, half, wb, c),
                          demod_cols(A, np.random.default_rng(2), 900, "edge", roi, half, wb, c)], 1)
    cols, coef = sc.local_E(Vtr, S, q)
    out = {"os": os_, "q": q, "roi": roi, "nS": int(len(S)), "reduction": len(A[1]) / len(S), "prototype": int(ros[0]), "tests": []}
    for i, rd in enumerate(rds[1:]):
        Bb, _ = sc.band(rd, roi, half, 1.0, wb)
        Sb = sc.select_warped(Bb, dwell, os_)
        same_S = bool(len(Sb) == len(S) and np.all(Sb == S))
        ent = {"readout": int(ros[i + 1]), "same_subset": same_S}
        for mode in ("signal", "edge"):
            V = demod_cols(rd, np.random.default_rng(40 + i), 400, mode, roi, half, wb, c)
            ent[f"proto_{mode}"] = sc.rel_err(V, sc.apply_E(cols, coef, V[S]))
            Vo = np.concatenate([demod_cols(rd, np.random.default_rng(60 + i), 900, "uni", roi, half, wb, c),
                                 demod_cols(rd, np.random.default_rng(61 + i), 900, "edge", roi, half, wb, c)], 1) if mode == "signal" else None
            if Vo is not None:
                co, ce = sc.local_E(Vo, Sb, q)
                ent["own_signal"] = sc.rel_err(V, sc.apply_E(co, ce, V[Sb]))
        out["tests"].append(ent)
        print(json.dumps(ent), flush=True)
    path = os.path.join(OUT, "proto_check.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res[f"{roi}_os{os_}_q{q}"] = out
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
