"""Undersampling artifacts vs encoding design at a fixed total number of readouts (2-D toy, 2026-11-17).

Question: in the undersampling-dominated regime (R ~ 50 per encode), how does the velocity error depend on
(i) the sensitivity (VENC), (ii) the number of encodes K at a fixed total sample count, (iii) whether all encodes
share one k-space sampling pattern (PC-VIPR style: one rotation played for every encode) or use independent ones?

Toy: 128^2 complex images built from a 4x finer grid (partial volume and intravoxel dephasing are in the data),
static body + fat ring + an "aorta" (1.5 m/s peak through-plane + swirl) + a descending aorta + six coronary-like
vessels (radius 2.5 px, 0.15-0.35 m/s, random directions). Single coil, Cartesian variable-density random masks
(p(k) ~ 1/|k|, the 2-D section of 3-D radial density), complex Gaussian k-space noise. Reconstruction per encode:
data-consistent TV (FISTA, 150 iterations) -- the regularised regime -- and the zero-filled image (min-norm LS).
Velocity: enc_lib.GridDecoder (approximate ML over |v| <= 1.15 x 3 m/s) and an oracle-wrap LS decode (continuous error only).
Scores: rms velocity error over the core voxels of the coronaries, the aorta, and static tissue (where v = 0).

    nice -n 10 rustlib/.venv/bin/python research/experiments/encoding/enc_artifact.py [out.json] [R_e at K=7, default 12] [TV lambda, default 0.01]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from enc_lib import GridDecoder, designs  # noqa: E402

N, UP = 128, 4
VEXT = 3.0  # m/s protected speed


def phantom(rng):
    n = N * UP
    y, x = np.mgrid[0:n, 0:n] / UP
    mag = np.zeros((n, n))
    vel = np.zeros((n, n, 3))
    body = ((x - 64) / 58) ** 2 + ((y - 64) / 50) ** 2 < 1
    mag[body] = 0.5 + 0.1 * np.sin(x[body] / 9) * np.cos(y[body] / 7)
    ring = body & (((x - 64) / 54) ** 2 + ((y - 64) / 46) ** 2 > 1)
    mag[ring] = 1.0  # fat-like bright rim
    lab = np.zeros((n, n), int)

    def vessel(cx, cy, r, vpk, dirn, label, swirl=0.0):
        rr = np.hypot(x - cx, y - cy)
        inside = rr < r
        prof = vpk * (1 - (rr[inside] / r) ** 2)
        mag[inside] = 0.9
        vel[inside] = prof[:, None] * np.asarray(dirn)[None]
        if swirl:
            t = np.arctan2(y[inside] - cy, x[inside] - cx)
            vel[inside, 0] += -swirl * np.sin(t) * rr[inside] / r
            vel[inside, 1] += swirl * np.cos(t) * rr[inside] / r
        lab[inside] = label

    vessel(52, 60, 11, 1.5, (0, 0, 1), 1, swirl=0.3)
    vessel(88, 84, 8, 1.0, (0, 0, -1), 2)
    for j in range(6):
        a = 2 * np.pi * j / 6 + 0.3
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        vessel(52 + 22 * np.cos(a), 60 + 22 * np.sin(a), 2.5, rng.uniform(0.15, 0.35), d, 3)
    return mag, vel, lab


def downsample(a):
    s = a.shape
    return a.reshape(N, UP, N, UP, *s[2:]).mean((1, 3))


def mask(frac, rng):
    ky, kx = np.meshgrid(np.fft.fftfreq(N) * N, np.fft.fftfreq(N) * N, indexing="ij")
    r = np.hypot(kx, ky)
    p = 1.0 / np.maximum(r, 1.0)
    p *= frac * N * N / p.sum()
    for _ in range(20):  # cap at 1 and renormalise
        p = np.minimum(p, 1.0)
        p *= frac * N * N / p.sum()
    return rng.uniform(size=(N, N)) < np.minimum(p, 1.0)


def grad(u):
    return np.stack([np.roll(u, -1, 0) - u, np.roll(u, -1, 1) - u])


def div(p):
    return (p[0] - np.roll(p[0], 1, 0)) + (p[1] - np.roll(p[1], 1, 1))


def tv_prox(z, lam, it=12):
    """argmin_u 1/2|u - z|^2 + lam TV(u), complex isotropic TV, Chambolle projection"""
    p = np.zeros((2,) + z.shape, complex)
    for _ in range(it):
        g = grad(div(p) - z / lam)
        p = (p + 0.125 * g) / (1 + 0.125 * np.sqrt((np.abs(g) ** 2).sum(0)))
    return z - lam * div(p)


def jtv_prox(Z, lam, it=12):
    """argmin 1/2 sum_e |u_e - z_e|^2 + lam sum_x sqrt(sum_e |grad u_e|^2): joint (group) TV across encodes"""
    p = np.zeros((len(Z), 2) + Z.shape[1:], complex)
    for _ in range(it):
        g = np.array([grad(div(p[e]) - Z[e] / lam) for e in range(len(Z))])
        nrm = np.sqrt((np.abs(g) ** 2).sum((0, 1)))
        p = (p + 0.125 * g) / (1 + 0.125 * nrm)
    return np.array([Z[e] - lam * div(p[e]) for e in range(len(Z))])


def recon_jtv(Y, Ms, lam, iters=150):
    F = lambda u: np.fft.fft2(u, norm="ortho")
    Fh = lambda k: np.fft.ifft2(k, norm="ortho")
    X = np.array([Fh(y * M) for y, M in zip(Y, Ms)])
    Z, t = X.copy(), 1.0
    for _ in range(iters):
        G = np.array([Fh(M * (F(z) - y)) for z, y, M in zip(Z, Y, Ms)])
        Xn = jtv_prox(Z - G, lam * np.sqrt(len(Y)))
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        Z = Xn + (t - 1) / tn * (Xn - X)
        X, t = Xn, tn
    return X


def recon_tv(y, M, lam, iters=150):
    F = lambda u: np.fft.fft2(u, norm="ortho")
    Fh = lambda k: np.fft.ifft2(k, norm="ortho")
    x = Fh(y * M)
    z, t = x.copy(), 1.0
    for _ in range(iters):  # step 1 (A^H A has norm 1)
        xn = tv_prox(z - Fh(M * (F(z) - y)), lam)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = xn + (t - 1) / tn * (xn - x)
        x, t = xn, tn
    return x


def run_case(k_phys, frac_tot, shared, sigma, rng, mag_f, vel_f, lam):
    K = len(k_phys)
    frac = frac_tot / K
    masks = [mask(frac, rng)] * K if shared else [mask(frac, rng) for _ in range(K)]
    bg = np.exp(1j * 2.0 * np.sin(np.mgrid[0:N * UP, 0:N * UP][0] / (N * UP) * 3))  # common background phase
    imgs, zf, Ys = [], [], []
    for e in range(K):
        x_f = mag_f * bg * np.exp(1j * (vel_f @ k_phys[e]))
        x = downsample(x_f)
        y = np.fft.fft2(x, norm="ortho")
        y = y + sigma * (rng.normal(size=y.shape) + 1j * rng.normal(size=y.shape)) / np.sqrt(2)
        M = masks[e]
        y = y * M
        zf.append(np.fft.ifft2(y, norm="ortho") / frac)
        imgs.append(recon_tv(y, M, lam))
        Ys.append(y)
    jt = recon_jtv(Ys, masks, lam)
    return np.array(imgs), np.array(zf), jt


def decode(imgs, k_phys, dec, truth_v, sel):
    ph = np.angle(imgs.reshape(len(k_phys), -1).T)[sel]  # (n, K)
    truth_v = truth_v[sel]
    v, _ = dec.decode(ph / 1.0)  # the decoder works in units of VEXT (k scaled)
    v = v * VEXT
    # oracle wraps: unwrap every encode against the truth prediction, then LS
    A = np.hstack([np.ones((len(k_phys), 1)), k_phys])
    Ap = np.linalg.pinv(A)
    th = np.angle(np.exp(1j * (ph - truth_v @ k_phys.T)).mean(1))
    pred = th[:, None] + truth_v @ k_phys.T
    phu = pred + np.angle(np.exp(1j * (ph - pred)))
    vo = phu @ Ap[1:].T
    return v, vo


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    rng = np.random.default_rng(3)
    mag_f, vel_f, lab_f = phantom(rng)
    mag = np.abs(downsample(mag_f))
    # core voxels: all 16 sub-voxels share the label; truth = magnitude-weighted mean velocity
    labs = lab_f.reshape(N, UP, N, UP).transpose(0, 2, 1, 3).reshape(N, N, -1)
    core = np.where(np.all(labs == labs[..., :1], -1), labs[..., 0], -1)
    vt = downsample(mag_f[..., None] * vel_f) / np.maximum(downsample(mag_f)[..., None], 1e-9)
    truth_v = vt.reshape(-1, 3)
    static = ((core == 0) & (mag > 0.3)).reshape(-1)
    keep = np.zeros(N * N, bool)
    keep[rng.choice(np.nonzero(static)[0], 1500, replace=False)] = True
    static &= keep
    groups_full = {"coronary": core.reshape(-1) == 3, "aorta": np.isin(core.reshape(-1), [1, 2]), "static": static}
    sel = groups_full["coronary"] | groups_full["aorta"] | static
    groups = {g: m[sel] for g, m in groups_full.items()}
    tv_sel = truth_v[sel]
    d = designs()
    from enc_lib import icosa6
    from enc_search import lam1_dual
    import math

    def integer(z, f=0.92):
        z = np.asarray(z, float)
        return z, f * math.pi * lam1_dual(z[1:] - z[0])

    setups = {
        "ref4 (f 0.9)": (d["ref4"], np.pi * 0.9),
        "coprime7 p=2": integer(d["coprime7 p=2"]),
        "coprime7 p=3": integer(d["coprime7 p=3"]),
        "coprime8H p=2 (f 1.2)": (d["coprime8H p=2"], np.pi * 1.2),
        "search6 @10.9": integer([[0, 0, 0], [2, -2, 0], [-1, -2, -1], [0, 0, -2], [2, 1, -3], [2, -1, -3]]),
        "search5 @10.2": integer([[0, 0, 0], [-3, 3, 0], [1, 3, -1], [-2, 2, -3], [-1, 2, -1]]),
    }
    sigma = 0.02  # k-space noise per sample (fully sampled single-encode image SNR ~ 0.9/0.02 = 45)
    frac_tot = 7 / float(sys.argv[2]) if len(sys.argv) > 2 else 7 / 12.0  # K = 7 at R_e = argv[2] per encode
    lam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    res = []
    for name, (shape, kap) in setups.items():
        k_unit = kap * np.asarray(shape, float)  # rad per (VEXT) -> per m/s: divide by VEXT
        k_phys = k_unit / VEXT
        dec = GridDecoder(k_unit)
        for shared in (True, False):
            imgs, zf, jt = run_case(k_phys, frac_tot, shared, sigma, rng, mag_f, vel_f, lam)
            row = dict(design=name, K=len(shape), shared=shared)
            for tag, im in (("tv", imgs), ("zf", zf), ("jtv", jt)):
                v, vo = decode(im, k_phys, dec, truth_v, sel)
                for g, gs in groups.items():
                    e = np.linalg.norm(v[gs] - tv_sel[gs], axis=1)
                    eo = np.linalg.norm(vo[gs] - tv_sel[gs], axis=1)
                    row[f"{tag}_{g}_rms"] = float(np.sqrt(np.mean(eo ** 2)))  # oracle wraps
                    row[f"{tag}_{g}_wrapfail"] = float(np.mean(e > 0.3))  # ML decode error > 0.3 m/s
            res.append(row)
            print(json.dumps(row), flush=True)
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
