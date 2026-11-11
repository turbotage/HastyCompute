"""Truth gradient chains for the study.

- 'resonance' / 'clinical' / other girf.rs variants (gmodel.variants): illustrative, the inverse-crime case for the
  parametric fit (same family).
- draw(rng): a random 'realistic' 3 T chain, deliberately richer than the fitted family. Ranges are assumptions
  informed by the literature (delays of a few us and a ~10-20 kHz bandwidth, eddy residuals after vendor
  pre-emphasis ~0.1 %, mechanical resonances 0.5-4 kHz of a few %, B0 cross terms ~1e-5 m; Vannesjo 2013, Rahmer
  2019, Stich 2020, Loecher 2025, Bacon 2026, partly from memory) [CHECK each against a measured GE GIRF]:
    per axis: delay U[1.5, 4] us, 1st-order low-pass tau U[4, 10] us, an extra 2nd-order roll-off (fc U[25, 60] kHz,
    Q U[0.5, 0.8]); eddy terms (a U[0.05, 0.3] %, tau logU[20, 150] us), (a U[0.05, 0.3] %, tau logU[0.5, 5] ms),
    long residual (a U[0.01, 0.08] %, tau logU[10, 60] ms); 2-4 resonances f0 U[0.5, 4] kHz, zeta logU[0.004,
    0.05], peak U[1, 8] %; a smooth complex ripple (5 Gaussian bumps 0.3-15 kHz, width 2 kHz, |c| ~ 0.2 %);
    all six cross terms high-pass (k N(0, 3e-4), tau logU[0.5, 3] ms); B0 terms two high-pass terms (k N(0, 2e-5) m,
    tau logU[1, 5] ms; k N(0, 1e-5) m, tau logU[0.1, 1] ms); odd amplitude nonlinearity g + beta g|g|/70 mT/m,
    beta U[0, 3e-4].
- thermal(g, dT, coef): the chain dT (deg C) warmer: delay + c_d dT, low-pass tau x (1 + c_lp dT), eddy amplitudes
  x (1 + c_e dT), resonance f0 x (1 + c_f dT), resonance peak x (1 + c_g dT), B0 terms unchanged (Stich 2020: the
  B0 cross term is not linear in temperature; left out). Default coefficients are assumptions [CHECK]: Nussbaum 2022
  report resonance shifts "in the same range as the linewidths" for heating to 32-65 deg C and rising magnitudes.
"""
import numpy as np
import gmodel as gm

THERMAL = dict(c_d=0.5e-9, c_lp=1e-3, c_e=3e-3, c_f=-1e-3, c_g=5e-3)


def logu(rng, a, b):
    return float(np.exp(rng.uniform(np.log(a), np.log(b))))


def draw(rng, name='rand'):
    axis = []
    for a in range(3):
        par = gm.Param(rng.uniform(1.5e-6, 4e-6), rng.uniform(4e-6, 10e-6),
                       [(rng.uniform(5e-4, 3e-3), logu(rng, 20e-6, 150e-6)), (rng.uniform(5e-4, 3e-3), logu(rng, 0.5e-3, 5e-3)),
                        (rng.uniform(1e-4, 8e-4), logu(rng, 10e-3, 60e-3))])
        nres = int(rng.integers(2, 5))
        res = [(rng.uniform(500, 4000), logu(rng, 0.004, 0.05), rng.uniform(0.01, 0.08)) for _ in range(nres)]
        rip = gm.Ripple(rng.uniform(300, 15e3, 5), 2e3, (rng.normal(0, 2e-3, 5) + 1j * rng.normal(0, 2e-3, 5)))
        axis.append(gm.Mul(gm.MultiRes(par, res), gm.LP2(rng.uniform(25e3, 60e3), rng.uniform(0.5, 0.8)), rip))
    cross = [(o, i, gm.HighPass(logu(rng, 0.5e-3, 3e-3), rng.normal(0, 3e-4))) for o in range(3) for i in range(3) if o != i]
    b0 = [gm.MultiHP([(rng.normal(0, 2e-5), logu(rng, 1e-3, 5e-3)), (rng.normal(0, 1e-5), logu(rng, 0.1e-3, 1e-3))]) for _ in range(3)]
    return gm.Girf(axis, cross=cross, b0=b0, tail=0.15, name=name, nl=rng.uniform(0, 3e-4))


def _therm(r, dT, c):
    if isinstance(r, gm.Param):
        return gm.Param(r.delay + c['c_d'] * dT, r.tau_lp * (1 + c['c_lp'] * dT), [(a * (1 + c['c_e'] * dT), t) for a, t in r.eddy])
    if isinstance(r, gm.Resonant):
        return gm.Resonant(_therm(r.base, dT, c), r.f0 * (1 + c['c_f'] * dT), r.zeta, r.gain * (1 + c['c_g'] * dT))
    if isinstance(r, gm.MultiRes):
        return gm.MultiRes(_therm(r.base, dT, c), [(f0 * (1 + c['c_f'] * dT), z, g * (1 + c['c_g'] * dT)) for f0, z, g in r.res])
    if isinstance(r, gm.Mul):
        return gm.Mul(*[_therm(p, dT, c) for p in r.parts])
    return r


def thermal(g, dT, coef=None, only=None):
    """the chain dT warmer; only = one coefficient name (the others zero) for attribution"""
    c = dict(THERMAL if coef is None else coef)
    if only is not None:
        c = {k: (v if k == only else 0.0) for k, v in c.items()}
    return g.copy_with(axis=[_therm(r, dT, c) for r in g.axis], name=f'{g.name}@{dT:+g}C')
