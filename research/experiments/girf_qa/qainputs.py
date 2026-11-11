"""Python port of the pass-2 input generators of hasty-seq girfqa.rs (narrow_chirp, tone_burst, pns_fit with the
SAFE example hardware and GE's hrmw PNS model), so the Monte Carlo can build each draw's targeted inputs; checked
against the Rust generator by check_vs_rust() (girfqa --targets)."""
import json, os, subprocess, sys
import numpy as np

GAMMA = 2.6752218744e8
DT = 4e-6
SMAX = 200.0
G_CAP = 45e-3
KLIM = lambda thk: 0.4 * 2 * np.pi / thk
SAFE = [dict(tau=[0.20, 0.03, 3.00], a=[0.40, 0.10, 0.50], lim=30.0, gs=0.35), dict(tau=[1.50, 2.50, 0.15], a=[0.55, 0.15, 0.30], lim=15.0, gs=0.31),
        dict(tau=[2.00, 0.12, 1.00], a=[0.42, 0.40, 0.18], lim=25.0, gs=0.25)]
GE = dict(chronaxie=642.4e-6, rheobase=17.9, alpha=0.310)


def _lp(x, tau_ms, dt_ms):
    al = dt_ms / (tau_ms + dt_ms)
    y = np.zeros_like(x)
    p = 0.0
    for i, v in enumerate(x):
        p = al * v + (1 - al) * p
        y[i] = p
    return y


def safe(g, ax, dt=DT):
    """hasty-seq pns.rs (SAFE, pypulseq example hw), single axis, % of threshold"""
    h = SAFE[ax]
    zpt = max(max(x['tau']) for x in SAFE) * 4 / 1000
    pad1, pad2 = int(round(zpt / 4 / dt)), int(round(zpt / dt))
    gw = np.r_[np.zeros(pad1), g, np.zeros(pad2)]
    s = np.diff(gw) / dt
    d = dt * 1e3
    st = h['a'][0] * np.abs(_lp(s, h['tau'][0], d)) + h['a'][1] * _lp(np.abs(s), h['tau'][1], d) + h['a'][2] * np.abs(_lp(s, h['tau'][2], d))
    return float((st / h['lim'] * h['gs'] * 100).max())


def pns_ge(g, dt=DT, pad=1250):
    """hasty-seq pnsge.rs (TOPPE pns.m), single axis, % of threshold"""
    c = GE['chronaxie']
    srmin = GE['rheobase'] / GE['alpha']
    x = np.r_[g, np.zeros(pad)]
    sr = np.r_[0.0, np.diff(x) / dt]
    n = len(sr)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    ker = 1.0 / (c + (np.arange(n) + 0.5) * dt) ** 2
    p = np.fft.irfft(np.fft.rfft(sr, nfft) * np.fft.rfft(ker, nfft), nfft)[:n] * 100 * dt * c / srmin
    return float(np.abs(p).max())


def worst(g):
    return max(max(safe(g, a), pns_ge(g)) for a in range(3))


def narrow_chirp(fc, zeta, t_sweep=30e-3, s_frac=0.5, thk=1.5e-3, t_on=1e-3):
    kl = 0.8 * KLIM(thk)
    half = max(6 * zeta * fc, 15.0)
    fa, fb = fc - half, fc + half
    a = min(G_CAP, s_frac * SMAX / (2 * np.pi * fb), 2 * np.pi * fa * kl / GAMMA)
    n = int(round((t_on + t_sweep + 0.5e-3) / DT)) + 1
    t = np.arange(n) * DT - t_on
    on = (t >= 0) & (t <= t_sweep)
    ph = 2 * np.pi * (fa * t + 0.5 * (fb - fa) / t_sweep * t * t)
    return np.where(on, a * np.sin(np.pi * t / t_sweep) ** 2 * np.sin(ph), 0.0)


def tone_burst(f, t_b=20e-3, s_frac=0.5, thk=1.5e-3, t_on=1e-3):
    kl = 0.8 * KLIM(thk)
    a = min(G_CAP, s_frac * SMAX / (2 * np.pi * f), 2 * np.pi * f * kl / GAMMA)
    n = int(round((t_on + t_b + 0.5e-3) / DT)) + 1
    t = np.arange(n) * DT - t_on
    on = (t >= 0) & (t <= t_b)
    return np.where(on, a * np.sin(np.pi * t / t_b) ** 2 * np.sin(2 * np.pi * f * t), 0.0)


def pns_fit(make, s0=0.5, limit=80.0):
    s = s0
    while True:
        g = make(s)
        if worst(g) <= limit or s < 0.05:
            return g, s
        s *= 0.93


def targeted(f0, zeta):
    g1, s1 = pns_fit(lambda s: narrow_chirp(f0, float(np.clip(zeta, 0.002, 0.05)), s_frac=s))
    g2, s2 = pns_fit(lambda s: tone_burst(f0, s_frac=s))
    return (g1, s1), (g2, s2)


def check_vs_rust(binpath, out, f0=712.3, zeta=0.006):
    """run girfqa --only p2 --targets and compare its inputs with this port"""
    os.makedirs(out, exist_ok=True)
    tj = os.path.join(out, 'targets.json')
    json.dump(dict(targets=[dict(axis=1, f0=f0, zeta=zeta, gain=0.04)]), open(tj, 'w'))
    subprocess.run([binpath, '--out', out, '--targets', tj, '--only', 'p2', '--no-seq'], check=True, capture_output=True)
    p = json.load(open(os.path.join(out, 'protocol.json')))
    X = np.load(os.path.join(out, 'inputs.npy'))
    names = [i['name'] for i in p['inputs']]
    (g1, s1), (g2, s2) = targeted(f0, zeta)
    i1, i2 = names.index(f'nchirp@{f0:.2f}'), names.index(f'tone@{f0:.2f}')
    r1 = X[i1, :len(g1)]
    r2 = X[i2, :len(g2)]
    return dict(nchirp_max_diff=float(np.abs(r1 - g1).max()), tone_max_diff=float(np.abs(r2 - g2).max()), s=[s1, s2], rust_s=[p['inputs'][i1]['slew_fraction'], p['inputs'][i2]['slew_fraction']],
                pns=[worst(g1), worst(g2)])


if __name__ == '__main__':
    print(json.dumps(check_vs_rust(sys.argv[1], sys.argv[2]), indent=1))
