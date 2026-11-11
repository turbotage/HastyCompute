"""Writer of the measured-GIRF file (hasty-girf-measured v1, read by hasty-seq girfmeas.rs) and of the resonance list
for the sequence designer, from the QA estimator's outputs; plus a Python reader that builds the same chain (for
checks against the Rust loader)."""
import json
import numpy as np
import qasim as qs
import qaest as qe

gm = qs.gm


def _tab(t):
    f = np.asarray(t.f, float)
    h = np.asarray(t.h, complex)
    keep = np.r_[True, np.diff(f) > 0]
    return dict(f=f[keep].tolist(), re=h[keep].real.tolist(), im=h[keep].imag.tolist())


def write(path, est_axes, targets, own, p2, therm, x_ref, xterms, rx=None, beta=None, scanner='simulated', date='', p_flow=None):
    """est_axes: per axis dict with 'tab' (continuous Table at state x_ref) and 'tab0' (B0 term); targets / own: the
    resonances and the table's own (f0, ζ, g) for each; p2: pass-2 fits (ζ, gain); therm: fit_thermal output
    (FA per target: f0 = F + A x); xterms: {(a, l): MultiHP}"""
    axes = []
    for a in range(3):
        tab = est_axes[a]['tab']
        f = np.asarray(tab.f, float)
        h = np.asarray(tab.h, complex)
        res = []
        for j, t in enumerate(targets):
            if t[0] != a:
                continue
            f0o, zo, go = own[j]
            h = h / (1 + go * qe.bp(f, f0o, zo))
            F_, A_ = therm['FA'][j] if therm else (t[1], 0.0)
            res.append(dict(f0=float(F_ + A_ * x_ref), zeta=float(p2[j]['zeta']), gain=float(p2[j]['gain']), df0_dx=float(A_), dgain_dx=0.0, sd_f0=float(p2[j]['sd'][0])))
        keep = np.r_[True, np.diff(f) > 0]
        axes.append(dict(f=f[keep].tolist(), re=h[keep].real.tolist(), im=h[keep].imag.tolist(), resonances=res))
    cross, spatial = [], []
    for (a, l), r in xterms.items():
        hp = [[float(k), float(t)] for k, t in r.terms]
        if l in (1, 2, 3) and l - 1 != a:
            cross.append({'out': l - 1, 'in': a, 'hp': hp})
        elif l >= 4:
            spatial.append({'in': a, 'basis': qs.BASIS[l], 'hp': hp})
    doc = {'format': 'hasty-girf-measured', 'version': 1, 'scanner': scanner, 'date': date,
           'thermal': {'model': 'power-2c', 'tau_s': [therm['tau1'], therm['tau2']] if therm else [], 'w': therm['w'] if therm else 1.0, 'p_flow_T2m2': p_flow, 'x_ref': x_ref},
           'axes': axes, 'cross': cross, 'b0': [_tab(est_axes[a]['tab0']) for a in range(3)], 'spatial': spatial,
           'receiver': {'delay_ns_vs_4us': {str(k): v for k, v in (rx or {}).items()}}, 'nonlinearity': {'beta': list(beta) if beta is not None else [0, 0, 0], 'g_ref': 0.07}}
    json.dump(doc, open(path, 'w'))
    return doc


def resonance_list(doc, x):
    """designer list at state x: (axis, f0, ζ, gain, sd_f0)"""
    out = []
    for a, ax in enumerate(doc['axes']):
        for r in ax['resonances']:
            out.append(dict(axis=a, f0=r['f0'] + r['df0_dx'] * (x - doc['thermal']['x_ref']), zeta=r['zeta'], gain=r['gain'], sd_f0=r['sd_f0']))
    return out


def read_chain(doc, x):
    """the same chain as girfmeas.rs MeasuredGirf::girf(x)"""
    axis = []
    for a, ax in enumerate(doc['axes']):
        r = gm.Table(ax['f'], np.array(ax['re']) + 1j * np.array(ax['im']))
        res = [(rr['f0'] + rr['df0_dx'] * (x - doc['thermal']['x_ref']), rr['zeta'], rr['gain']) for rr in ax['resonances']]
        axis.append(gm.MultiRes(r, res) if res else r)
    cross = []
    for c in doc['cross']:
        for k, t in c['hp']:
            cross.append((c['out'], c['in'], gm.HighPass(t, k)))
    b0 = [None if b is None else gm.Table(b['f'], np.array(b['re']) + 1j * np.array(b['im'])) for b in doc['b0']]
    return gm.Girf(axis, cross=cross, b0=b0, tail=0.15, name='measured')
