"""Operator frames for the parameter-spline velocity model (research/velocity_model_sr.md §F): how many operator frames,
and which interpolation between them, make the knot operator reproduce the model signal at every sample time.

Model (parameters in the 6 × 16 respiratory × cardiac basis, the target of temporal_operator.md §21):
    x_e(r, t) = m(r, t) exp(i (ψ(r, t) + φ_e(r) − k_e·v(r, t))),  m, ψ, v = Σ_b B_b(t) (m_b, ψ_b, v_b)
The operator only interpolates complex images between its frames. Options per cardiac period (at one breathing state):
  lin F      model values at F uniform frames, linear interpolation (operator cardiac degree 1; 2 cardiac weights/sample)
  cub F      model values at F frames, periodic cubic-spline interpolation: coefficients = (1/6, 2/3, 1/6)-circulant⁻¹
             applied to the frame values per voxel (a pointwise-in-space linear map), operator cardiac degree 3 (4 weights)
  herm F     value + analytic time derivative at F frames, cubic Hermite per sample (2F images, 4 weights; needs new
             operator kernels)
  cubw F     cub F on a global cardiac-time warp s(τ) (frames denser where the model's phase moves fast; the warp comes
             from the model itself — in the method from θ, never from truth): operator uniform in s, only the samples'
             cardiac coordinates change
Respiratory direction: operator frames = the parameter knots (complex-linear between them) or with midpoints.

Truth is used for EVALUATION ONLY (it builds a realistic parameter field): ph0001 truth velocities (truth_vel_logical, 20
frames, Fourier-interpolated, as interp_caveat.py), lumen fraction > 0.9, magnitude |c_0| per frame (truth_pc_signal),
ψ = 0; the parameter knots are the least-squares spline fit of that field. Errors are of the operator signal against
the model signal (the operator approximation error), normalised
  rel_lumen   over the lumen voxels' model signal energy
  rel_global  over the signal energy of all voxels (the 1e-4 segment rule's normalisation: static tissue is exact)
  vel         RMS of the decoded encode-difference velocity (low-VENC encodes) of the operator signal against the
              model's own velocity, relative to the dynamic velocity RMS (the physics/basis error of 16 cubic knots is
              0.10 on that scale, interp_caveat.py)
Cost per encode pass, 2 mm full FOV, 30 virtual coils, L 10, 30 M samples (temporal_operator.md §22.6: 0.28 s per
operator image, 99 s sample side at 8 weights per sample = 12.4 s per weight): T = 0.28 · n_images + 12.4 · weights;
--keep 8: the sample side / 8.

    python -I frames_caveat.py <scan dir> [--sub 12000] [--out result.json]
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/lib"))
from hastycompute.store import load_any

ap = argparse.ArgumentParser()
ap.add_argument("scan")
ap.add_argument("--knots", type=int, default=16)
ap.add_argument("--kr", type=int, default=6)
ap.add_argument("--nt", type=int, default=960)
ap.add_argument("--sub", type=int, default=6000)
ap.add_argument("--frac", type=float, default=0.9)
ap.add_argument("--out", default=None)
ap.add_argument("--frames", default="16,24,32,48,64,96,128", help="operator frame counts to evaluate")
a = ap.parse_args()
S = Path(a.scan)
rng = np.random.default_rng(7)
vel = np.asarray(load_any(str(S / "truth_vel_logical.hv")), dtype=np.float32)  # (X, Y, Z, 3, F)
lum = np.asarray(load_any(str(S / "truth_lumen_fraction.npy")))
sig = np.load(S / "truth_pc_signal.npz")
c0abs = sig["c0_abs"]  # (X, Y, Z, 20)
k = np.load(S / "truth_pc_frames.npz")["k"].astype(np.float64)  # (E, 3)
E = k.shape[0]
F20 = vel.shape[-1]
idx_all = np.flatnonzero(lum.ravel(order="F") > a.frac)
idx = np.sort(rng.choice(idx_all, size=min(a.sub, len(idx_all)), replace=False))
n = len(idx)
v20 = vel.reshape(-1, 3, F20, order="F")[idx].astype(np.float64)
m20 = c0abs.reshape(-1, F20, order="F")[idx].astype(np.float64)
# global signal energy per voxel-time (all voxels, all encodes ≈ 7 |c0|²) and the lumen sample's share scaled to all
# lumen voxels: rel_global = rel_lumen · sqrt(E_lumen / E_all)
e_all = float((c0abs.astype(np.float64) ** 2).sum()) * E
e_lum = float((c0abs.reshape(-1, F20, order="F")[idx_all].astype(np.float64) ** 2).sum()) * E
glob = np.sqrt(e_lum / e_all)
del vel, c0abs
nt = a.nt
tau = (np.arange(nt) + 0.5) / nt
def fourier(x20):
    sp = np.fft.rfft(x20, axis=-1)
    h = np.arange(sp.shape[-1])
    ph = np.exp(2j * np.pi * np.outer(h, tau - 0.5 / F20))
    w = np.where((h == 0) | ((F20 % 2 == 0) & (h == F20 // 2)), 1.0, 2.0)
    return np.einsum("...h,ht->...t", sp * w, ph).real / F20
vt = fourier(v20)  # (n, 3, nt)
mt = np.maximum(fourier(m20), 0.0)
M = a.knots
def card_w(t, b, m, deg=3, der=0):
    d = (t - b) % m
    d = np.where(d > 0.5 * m, d - m, d)
    x = np.abs(d)
    s = np.sign(d)
    if deg == 1:
        return np.maximum(1 - x, 0.0) if der == 0 else np.where(x < 1, -s, 0.0)
    if der == 0:
        return np.where(x <= 1, 2 / 3 - x * x + 0.5 * x ** 3, np.where(x <= 2, (2 - x) ** 3 / 6, 0.0))
    return s * np.where(x <= 1, -2 * x + 1.5 * x * x, np.where(x <= 2, -0.5 * (2 - x) ** 2, 0.0))
def basis(tq, m, deg=3, der=0):
    t = (tq * m - 0.5) % m
    B = np.stack([card_w(t, b, m, deg, der) for b in range(m)], axis=1)
    return B * (m ** der)  # d/dτ
Bt = basis(tau, M)
P = np.linalg.pinv(Bt)
vb = np.einsum("bt,nat->nab", P, vt)
mb = np.einsum("bt,nt->nb", P, mt)
# the model and its analytic derivative at arbitrary cardiac times
def model(tq, der=False):
    B = basis(tq, M)
    v = np.einsum("tb,nab->nat", B, vb)
    m = np.einsum("tb,nb->nt", B, mb)
    ph = -np.einsum("ea,nat->net", k, v)
    x = m[:, None, :] * np.exp(1j * ph)
    if not der:
        return x, v
    Bd = basis(tq, M, der=1)
    vd = np.einsum("tb,nab->nat", Bd, vb)
    md = np.einsum("tb,nb->nt", Bd, mb)
    xd = (md[:, None, :] - 1j * m[:, None, :] * np.einsum("ea,nat->net", k, vd)) * np.exp(1j * ph)
    return x, v, xd
xm, vm = model(tau)
ref_e = float((np.abs(xm) ** 2).sum())
dks = {e: np.linalg.norm(k[e] - k[0]) for e in range(1, E)}
lo = [e for e in dks if dks[e] > 0.999 * max(dks.values())]
okv = np.abs(xm[:, 0]).min(-1) > 0.05 * np.median(np.abs(xm[:, 0]))  # voxels with signal at every time (the truth has
# zero magnitude in ≈ 15 % of the lumen-fraction voxels: outside the simulated coverage)
def vel_err(x):
    x = x[okv]
    out = []
    for e in lo:
        dk = k[e] - k[0]
        nk_ = np.linalg.norm(dk)
        u = dk / nk_
        vtrue = np.einsum("a,nat->nt", u, vm[okv])
        vd = vtrue - vtrue.mean(-1, keepdims=True)
        ref = -nk_ * vtrue
        p = np.angle(x[:, e] * np.conj(x[:, 0]))
        p = ref + np.angle(np.exp(1j * (p - ref)))
        err = -p / nk_ - vtrue
        out.append((np.sqrt((err ** 2).mean() / (vd ** 2).mean()), np.sqrt((err ** 2).mean())))
    return float(np.mean([o[0] for o in out])), float(np.mean([o[1] for o in out]))
def score(xo, n_img, w_card, name):
    e = xo - xm
    rl = float(np.sqrt((np.abs(e) ** 2).sum() / ref_e))
    per = np.abs(e) / np.maximum(np.abs(xm), 1e-3 * np.abs(xm).max())
    vr, va = vel_err(xo)
    w = 2 * w_card  # respiratory linear × cardiac
    return {"name": name, "images_per_resp": n_img, "weights": w, "rel_lumen": rl, "rel_global": rl * glob, "p999_rel": float(np.percentile(per, 99.9)),
            "vel_rel": vr, "vel_rms_ms": va, "cost_full_s": 0.28 * n_img * a.kr + 12.4 * w, "cost_keep8_s": 0.28 * n_img * a.kr + 12.4 * w / 8}
def circ_inv(F):
    c = np.zeros(F); c[0] = 2 / 3; c[1] = 1 / 6; c[-1] = 1 / 6
    return np.real(np.fft.ifft(1 / np.fft.fft(c)))
def cub(F, tf=None, ts=None):
    """model values at frame times tf (cardiac phase), cubic spline uniform in s with samples at ts = s(τ)"""
    tf = (np.arange(F) + 0.5) / F if tf is None else tf
    ts = tau if ts is None else ts
    xf, _ = model(tf)
    ci = circ_inv(F)
    C = np.stack([np.roll(ci, f) for f in range(F)], axis=0)  # C[f, g] = ci[(g - f) mod F]: symmetric circulant
    cf = np.einsum("fg,neg->nef", C, xf)
    return np.einsum("tf,nef->net", basis(ts, F), cf)
def lin(F):
    xf, _ = model((np.arange(F) + 0.5) / F)
    return np.einsum("tf,nef->net", basis(tau, F, deg=1), xf)
def herm(F):
    tf = (np.arange(F) + 0.5) / F
    xf, _, xdf = model(tf, der=True)
    h = 1.0 / F
    t = (tau * F - 0.5) % F
    i0 = np.floor(t).astype(int) % F
    i1 = (i0 + 1) % F
    u = t - np.floor(t)
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return xf[:, :, i0] * h00 + h * xdf[:, :, i0] * h10 + xf[:, :, i1] * h01 + h * xdf[:, :, i1] * h11
res = {"voxels": n, "voxels_with_signal": int(okv.sum()), "lumen_voxels": int(len(idx_all)), "knots": M, "nt": nt, "glob_factor": glob, "low_venc": lo, "cardiac": []}
# phase rate of the model (population RMS over voxels and encodes): the warp density (θ-derived in the method)
_, _, xd = model(tau, der=True)
rate = np.sqrt((np.abs(xd) ** 2).sum((0, 1)) / np.maximum((np.abs(xm) ** 2).sum((0, 1)), 1e-30))  # (nt,) rad per cycle
res["rate_rms"] = {"mean": float(rate.mean()), "max": float(rate.max()), "min": float(rate.min())}
def cubls(F):
    """L2 projection of the model onto periodic cubic splines with F knots (least squares over the fine times; in the
    method: quadrature points per frame interval, a banded Gram)"""
    Bf = basis(tau, F)
    Pf = np.linalg.pinv(Bf)
    return np.einsum("tf,nef->net", Bf, np.einsum("ft,net->nef", Pf, xm))
# physics budget (cardiac): the 16-knot model of the truth field against the truth signal itself
xtrue = mt[:, None, :] * np.exp(-1j * np.einsum("ea,nat->net", k, vt))
et = xm - xtrue
res["physics_cardiac"] = {"rel_lumen": float(np.sqrt((np.abs(et) ** 2).sum() / (np.abs(xtrue) ** 2).sum())), "rel_global": float(np.sqrt((np.abs(et) ** 2).sum() / (np.abs(xtrue) ** 2).sum()) * glob)}
del xtrue, et
print("physics (16-knot model vs truth signal):", res["physics_cardiac"], flush=True)
for F in [int(x) for x in a.frames.split(",")]:
    res["cardiac"].append(score(cubls(F), F, 4, f"cubls{F}"))
    res["cardiac"].append(score(lin(F), F, 2, f"lin{F}"))
    res["cardiac"].append(score(cub(F), F, 4, f"cub{F}"))
    if F <= 64:
        res["cardiac"].append(score(herm(F), 2 * F, 4, f"herm{F}"))
    for beta, alpha in ((0.5, 1.0), (1.0, 0.5), (1.0, 0.8)):
        dens = (1 - beta) + beta * (rate / rate.mean()) ** alpha
        dens = np.convolve(np.r_[dens[-nt // 32:], dens, dens[:nt // 32]], np.ones(nt // 16) / (nt // 16), mode="same")[nt // 32: nt // 32 + nt]
        s = (np.cumsum(dens) - 0.5 * dens) / dens.sum()  # s(τ_j): monotone, [0, 1)
        # frame times: τ_f = s⁻¹((f + ½)/F)
        tf = np.interp((np.arange(F) + 0.5) / F, s, tau)
        res["cardiac"].append(score(cub(F, tf, s), F, 4, f"cubw{F}_b{beta}_a{alpha}"))
    print(json.dumps([r for r in res["cardiac"] if r["images_per_resp"] in (F, 2 * F) and r["name"].endswith(str(F)) or r["name"].startswith(f"cubw{F}_")], indent=None), flush=True)
# respiratory direction: truth per-encode signals at the states 0.181 and 0.493 (one knot spacing of the bellows-quantile
# knots in the expiration range: knots at state ≈ 0.09, 0.33, 0.63, …), common lumen voxels
A = np.load(S / "truth_pc_frames_r0.181.npz"); Bz = np.load(S / "truth_pc_frames.npz")
ia, ib = A["idx"], Bz["idx"]
com, pa, pb = np.intersect1d(ia, ib, return_indices=True)
keep = np.isin(com, idx_all)
ca, cb = A["c"][pa[keep]].astype(np.complex128), Bz["c"][pb[keep]].astype(np.complex128)
ok = (A["cov"][pa[keep]].min(1) >= 0.99) & (Bz["cov"][pb[keep]].min(1) >= 0.99)
ca, cb = ca[ok], cb[ok]
dphi = np.angle(cb * np.conj(ca))
ma, mbm = np.abs(ca), np.abs(cb)
# param-linear model between the two states vs complex-linear, at u ∈ [0, 1] (the wrapped step: a lower bound)
us = np.linspace(0, 1, 33)
rows = {}
for nsub in (1, 2, 3):
    num = den = 0.0
    for u in us:
        # nsub operator frames per knot interval: complex-linear between model values at j / nsub
        j = min(int(u * nsub), nsub - 1)
        u0, u1 = j / nsub, (j + 1) / nsub
        w = (u - u0) * nsub
        mod = lambda q: ((1 - q) * ma + q * mbm) * np.exp(1j * (np.angle(ca) + q * dphi))
        xo = (1 - w) * mod(u0) + w * mod(u1)
        xm_ = mod(u)
        num += (np.abs(xo - xm_) ** 2).sum()
        den += (np.abs(xm_) ** 2).sum()
    rows[f"op_frames_per_interval_{nsub}"] = {"rel_lumen": float(np.sqrt(num / den)), "rel_global": float(np.sqrt(num / den) * glob)}
ad = np.abs(dphi).ravel()
res["respiratory"] = {"voxels": int(ok.sum()), "state_step": 0.493 - 0.181, "phase_step": {"p50": float(np.percentile(ad, 50)), "p99": float(np.percentile(ad, 99)), "max": float(ad.max()), "frac_gt_pi4": float((ad > np.pi / 4).mean())}, "interp": rows}
# physics budget (respiratory): predict the truth at state 0.181 from the states 0.07 and 0.493 (u = 0.262), complex-
# linear (what the operator does between respiratory frames) vs parameter-linear (magnitude and phase linear, the wrapped
# step), against the truth signal at 0.181
Z = np.load(S / "truth_pc_frames_r0.07.npz")
i3, p0, p1, p2 = np.intersect1d(np.intersect1d(Z["idx"], A["idx"]), Bz["idx"]), None, None, None
def take(z, ids):
    _, pa_, pb_ = np.intersect1d(z["idx"], ids, return_indices=True)
    o = np.argsort(pb_)
    return z["c"][pa_[o]].astype(np.complex128), z["cov"][pa_[o]]
i3 = i3[np.isin(i3, idx_all)]
c07, v07 = take(Z, i3); c18, v18 = take(A, i3); c49, v49 = take(Bz, i3)
okr = (v07.min(1) >= 0.99) & (v18.min(1) >= 0.99) & (v49.min(1) >= 0.99)
c07, c18, c49 = c07[okr], c18[okr], c49[okr]
u = (0.181 - 0.07) / (0.493 - 0.07)
xc = (1 - u) * c07 + u * c49
dp = np.angle(c49 * np.conj(c07))
xp = ((1 - u) * np.abs(c07) + u * np.abs(c49)) * np.exp(1j * (np.angle(c07) + u * dp))
den = (np.abs(c18) ** 2).sum()
res["physics_respiratory"] = {"voxels": int(okr.sum()), "u": u, "complex_linear_rel_lumen": float(np.sqrt((np.abs(xc - c18) ** 2).sum() / den)), "param_linear_rel_lumen": float(np.sqrt((np.abs(xp - c18) ** 2).sum() / den)), "op_vs_param_rel_lumen": float(np.sqrt((np.abs(xc - xp) ** 2).sum() / den))}
for kk in ("complex_linear", "param_linear", "op_vs_param"):
    res["physics_respiratory"][kk + "_rel_global"] = res["physics_respiratory"][kk + "_rel_lumen"] * glob
print(json.dumps(res["respiratory"], indent=1))
print(json.dumps(res["physics_respiratory"], indent=1))
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1))
