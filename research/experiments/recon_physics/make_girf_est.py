"""Estimated GIRF for the reconstruction: one simulated WP2 thin-slice measurement of a truth chain (default the
girf.rs `resonance` variant, the chain of the next v2 resimulation), estimated as in research/girf_sensitivity.md
(mc.estimate: nonparametric continuous H below f_trust + parametric extension, B0 terms), written as a
`hasty-girf-1` table file (hasty_seq Girf::from_measurement; format in research/recon_physics_wiring.md).

    make_girf_est.py [--truth resonance] [--tadc 0.2] [--navg 25] [--snr 1000] [--draw 0] [--out <file.json>]

Imports research/experiments/girf_sensitivity/{gmodel,thinslice,mc}.py read-only.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GS = os.path.join(HERE, '..', 'girf_sensitivity')
sys.path.insert(0, GS)
import gmodel as gm, thinslice as ts, mc  # noqa: E402

REPO = os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute')
OUT = os.path.join(REPO, 'data/hasty/phantom/factory_arms/ph0001/recon_physics_2026-11-11')


def tab_json(r):
    """a gm response as {f, h} (Table: its own grid; else a dense grid to 5 MHz)"""
    if isinstance(r, gm.Table):
        f, h = np.asarray(r.f, float), np.asarray(r.h, complex)
    else:
        f = np.r_[np.arange(0, 20e3, 2.0), np.geomspace(20e3, 5e6, 600)]
        h = r.at(f)
    return {'f': f.tolist(), 'h': {'re': h.real.tolist(), 'im': h.imag.tolist()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', default='resonance')
    ap.add_argument('--tadc', type=float, default=0.2)
    ap.add_argument('--navg', type=int, default=25)
    ap.add_argument('--snr', type=float, default=1000.0)
    ap.add_argument('--draw', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(1000 + a.draw)
    prot = ts.Protocol(snr0=a.snr, n_avg=a.navg, thk=1.5e-3, t_adc=a.tadc, t_frame=a.tadc + 0.02)
    truth = gm.variant(a.truth)
    nu = ts.Nuisance(rng, prot)
    Y, sig = ts.simulate(truth, prot, nu, rng)
    RK, RP = mc.ref_predictions(prot, gm.clinical_example())
    est_par, est_np, diag = mc.estimate(Y, sig, prot, RK, RP)
    res = []
    for ax in range(3):
        e = est_par.axis[ax]
        for (f0, z, g) in (e.res if isinstance(e, gm.MultiRes) else []):
            res.append({'axis': ax, 'f0': float(f0), 'zeta': float(z), 'gain': float(g)})
    band = mc.band_errors(truth, est_np, diag)
    j = {
        'format': 'hasty-girf-1', 'convention': 'continuous',
        'f': [0.0, 1.0],  # unused: every term carries its own grid
        'self': [tab_json(est_np.axis[ax]) for ax in range(3)],
        'cross': [],
        'b0': [tab_json(est_np.b0[ax]) if est_np.b0[ax] is not None else None for ax in range(3)],
        'tail': 0.05,
        'resonances': res,
        'note': (f'estimated (simulated WP2 thin-slice measurement, research/girf_sensitivity.md protocol) of the girf.rs '
                 f'{a.truth!r} chain: ADC {a.tadc * 1e3:.0f} ms, {a.navg} averages ({prot.minutes():.1f} min), SNR0 {a.snr:.0f}, '
                 f'nominal session nuisances, draw {a.draw}; nonparametric below f_trust, parametric above; cross terms not '
                 f'measured (0); resonance list from the parametric fit'),
        'diag': {'f_trust': [diag[ax]['f_trust'] for ax in range(3)], 'f0_trust': [diag[ax]['f0_trust'] for ax in range(3)],
                 'gd_err_ns': [band[f'gd{ax}_ns'] for ax in range(3)], 'band': band},
    }
    out = a.out or os.path.join(OUT, f'girf_est_{a.truth}_T{int(round(a.tadc * 1e3))}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(j, open(out, 'w'))
    print(f'{out}: resonances {res}; group-delay error per axis {[round(band[f"gd{ax}_ns"], 2) for ax in range(3)]} ns; '
          f'f_trust {[round(diag[ax]["f_trust"]) for ax in range(3)]} Hz ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
