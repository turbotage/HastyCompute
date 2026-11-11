"""SAFE PNS (pypulseq example hardware, as hasty-seq pns.rs; NOT a GE model) of the thin-slice test inputs, played on
each axis alone: max % of threshold per input and axis -> <OUT>/pns_inputs.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import thinslice as ts

HW = [dict(tau=[0.20, 0.03, 3.00], a=[0.40, 0.10, 0.50], lim=30.0, gs=0.35), dict(tau=[1.50, 2.50, 0.15], a=[0.55, 0.15, 0.30], lim=15.0, gs=0.31),
      dict(tau=[2.00, 0.12, 1.00], a=[0.42, 0.40, 0.18], lim=25.0, gs=0.25)]


def lp(x, tau_ms, dt_ms):
    al = dt_ms / (tau_ms + dt_ms)
    y = np.zeros_like(x); p = 0.0
    for i, v in enumerate(x):
        p = al * v + (1 - al) * p; y[i] = p
    return y


def safe(g, dt, ax):
    h = HW[ax]
    s = np.diff(g) / dt
    s = np.r_[np.zeros(int(3.0 / (dt * 1e3)) * 4), s, np.zeros(int(3.0 * 4 / (dt * 1e3)))]
    d = dt * 1e3
    st = h['a'][0] * np.abs(lp(s, h['tau'][0], d)) + h['a'][1] * lp(np.abs(s), h['tau'][1], d) + h['a'][2] * np.abs(lp(s, h['tau'][2], d))
    return float((st / h['lim'] * h['gs'] * 100).max())


if __name__ == '__main__':
    out = {}
    for s_chirp in (0.5, 0.8):
        p = ts.Protocol(s_chirp=s_chirp)
        for n, g in p.inputs():
            if n != 'chirp' and s_chirp != 0.5:
                continue
            out[f'{n}_s{s_chirp}'] = [safe(g, p.dt, a) for a in range(3)]
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ts.__file__.rsplit('/research/', 1)[0], 'data/hasty/seq/studies/girf_sensitivity_2026-11-11/pns_inputs.json'), 'w'), indent=1)
