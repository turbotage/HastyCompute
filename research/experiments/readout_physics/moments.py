"""Readout moments about the echo for a seqdesign run: M0 (k), M1, M2 split into the part parallel to M0
(pure displacement t1 = M1.M0/|M0|^2, t2 = M2.M0/|M0|^2) and the perpendicular remainder.

  moments.py <design_dir> [out.npz]
Times from the echo (the first k = 0 of the readout), raster dt of the design (10 us).
"""
import sys, json, numpy as np
G = 2.6752218744e8


def readout_moments(run):
    b = json.load(open(run + '/summary.json'))['best']
    dt = json.load(open(run + '/summary.json'))['problem']['dt']
    g = np.load(run + '/tr_grad.npy')[0]
    i0 = b['i_ro'] + b.get('i_echo', 0)
    i1 = b['i_ro'] + b.get('recon', [[0, b['i_adc']]])[0][1]
    t = (np.arange(len(g)) - i0) * dt
    gg = g[i0:i1 + 1]; tt = t[i0:i1 + 1]
    # midpoint (trapezoid-consistent) cumulative moments, gradient raster value held over [t, t+dt)
    tm = tt + 0.5 * dt
    m0 = np.cumsum(gg, 0) * dt
    m1 = np.cumsum(gg * tm[:, None], 0) * dt
    m2 = np.cumsum(gg * tm[:, None] ** 2, 0) * dt
    tend = tt + dt
    n0 = (m0 ** 2).sum(1)
    ok = n0 > 1e-30 * n0.max()
    t1 = np.where(ok, (m1 * m0).sum(1) / np.where(ok, n0, 1), 0)
    t2 = np.where(ok, (m2 * m0).sum(1) / np.where(ok, n0, 1), 0)
    p1 = m1 - t1[:, None] * m0; p2 = m2 - t2[:, None] * m0
    return dict(t=tend, k=G * m0, kabs=G * np.sqrt(n0), m1=G * m1, m2h=0.5 * G * m2, t1=t1, t2=t2,
                perp1=G * np.linalg.norm(p1, axis=1), perp2=0.5 * G * np.linalg.norm(p2, axis=1),
                m1perp=G * p1, m2hperp=0.5 * G * p2, te=b['te'], dt=dt, kmax=2 * np.pi * json.load(open(run + '/summary.json'))['problem']['kmax'])


if __name__ == '__main__':
    r = readout_moments(sys.argv[1])
    kq = r['kabs'] / r['kmax']
    print(f"{sys.argv[1]}: window {r['t'][-1]*1e3:.2f} ms after the echo, max |k|/kmax {kq.max():.3f}")
    first = lambda q: np.argmax(kq >= q)
    for q in [0.1, 0.25, 0.5, 0.75, 0.99]:
        i = first(q)
        print(f"  first |k| = {q:.2f} kmax at {r['t'][i]*1e3:.2f} ms: t1 {r['t1'][i]*1e3:.2f} ms (disp {r['t1'][i]*1e3:.2f} mm per m/s), "
              f"t2 {r['t2'][i]*1e6:.1f} us^2 (0.5 a t2 at 20 m/s2: {0.5*20*r['t2'][i]*1e3:.3f} mm), |gamma M1perp| {r['perp1'][i]:.3f} rad/(m/s), |gamma M2perp/2| {r['perp2'][i]:.2e} rad/(m/s2)")
    print(f"  max |gamma M1perp| {r['perp1'].max():.3f} at {r['t'][r['perp1'].argmax()]*1e3:.2f} ms; max |M2perp/2| {r['perp2'].max():.2e}; max t1 {r['t1'].max()*1e3:.2f} ms")
    if len(sys.argv) > 2:
        np.savez(sys.argv[2], **{k: v for k, v in r.items()})
