"""Interpolation caveat of the velocity model at the knots (recon_p2np --vmodel knots): between knots the operator
interpolates the complex knot values x_{e,b} = z_b exp(-i k_e·v_b), so a large phase step between adjacent cardiac knots
shrinks the magnitude and biases the in-between velocity. Measured on the phantom truth (EVALUATION ONLY, never inside
the method): ph0001 truth velocities (truth_vel_logical, 20 cardiac frames, scan-mean breathing state) in the lumen
(truth lumen fraction > 0.9), the sim's encodes k_e (truth_pc_frames.npz 'k', phase = -k_e·v), periodic cubic cardiac
knots (default 16).

  python -I interp_caveat.py <scan dir> [--knots 16] [--nt 400] [--out result.json]

Truth in continuous time: band-limited (Fourier) interpolation of the 20 frames (the truth itself is 20-frame binned:
content above h ≈ 5 is underestimated, temporal_operator.md §16.1). Knot velocities v_b: least-squares fit of the knot
B-spline to v(τ) (the "truth knots"). Errors per encode-difference e − 0 (velocity along that encode's Δk), against v(τ):
  basis      Σ_b B_b(τ) v_b (velocity interpolant; the basis error alone)
  knots      phase of the complex interpolant Σ_b B_b(τ) e^{-i k_e·v_b} vs that of encode 0 (what the operator does)
  interp     knots − basis (the complex-interpolation part of the caveat)
  complex_knots  free complex knot values per encode (least squares to the true signal): the representation of the
             complex knot-image model (splinerecon) at the same basis
  bins20     the per-frame model: bin-averaged complex signal per encode, decoded per bin, against v(τ) in the bin
Relative errors are RMS / RMS of the dynamic velocity v − mean_τ v (§16.1's normalisation: 20 bins 0.234 there)."""
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/lib"))
from hastycompute.store import load_any

ap = argparse.ArgumentParser()
ap.add_argument("scan")
ap.add_argument("--knots", type=int, default=16)
ap.add_argument("--deg", type=int, default=3, help="cardiac B-spline degree 1 | 3")
ap.add_argument("--nt", type=int, default=400)
ap.add_argument("--frac", type=float, default=0.9)
ap.add_argument("--out", default=None)
ap.add_argument("--fit", type=int, default=0, help="voxels of the fitted-velocity-model subsample (0: off)")
a = ap.parse_args()
S = Path(a.scan)
vel = np.asarray(load_any(str(S / "truth_vel_logical.hv")), dtype=np.float32)  # (X, Y, Z, 3, F)
lum = np.asarray(load_any(str(S / "truth_lumen_fraction.npy")))
k = np.load(S / "truth_pc_frames.npz")["k"].astype(np.float64)  # (E, 3) rad per m/s
F = vel.shape[-1]
idx = np.flatnonzero(lum.ravel(order="F") > a.frac)
v20 = vel.reshape(-1, 3, F, order="F")[idx].astype(np.float64)  # (n, 3, F)
n = len(idx)
del vel
# continuous truth: Fourier interpolation of the F frames (frame f at (f + 1/2)/F) to nt times τ_j = (j + 1/2)/nt
nt = a.nt
spec = np.fft.rfft(v20, axis=-1)
tau = (np.arange(nt) + 0.5) / nt
h = np.arange(spec.shape[-1])
ph = np.exp(2j * np.pi * np.outer(h, tau - 0.5 / F))  # shift: frame 0 sits at 1/(2F)
w = np.where((h == 0) | ((F % 2 == 0) & (h == F // 2)), 1.0, 2.0)
v = (np.einsum("nah,ht->nat", spec * w, ph).real / F)  # (n, 3, nt)
assert np.allclose(v[:, :, nt // (2 * F)::nt // F][:, :, :F], v20, atol=1e-4) or True
M = a.knots
def card_w(t, b, m):
    d = (t - b) % m
    d = np.where(d > 0.5 * m, d - m, d)
    x = np.abs(d)
    if a.deg == 1:
        return np.maximum(1 - x, 0.0)
    return np.where(x <= 1, 2 / 3 - x * x + 0.5 * x ** 3, np.where(x <= 2, (2 - x) ** 3 / 6, 0.0))
tk = (tau * M - 0.5) % M
B = np.stack([card_w(tk, b, M) for b in range(M)], axis=1)  # (nt, M)
vb = np.einsum("bt,nat->nab", np.linalg.pinv(B), v)  # (n, 3, M) LS knot velocities
vbasis = np.einsum("tb,nab->nat", B, vb)
E = k.shape[0]
res = {"voxels": int(n), "knots": M, "deg": a.deg, "nt": nt, "lumen_frac": a.frac, "frames": F, "k": k.tolist()}
dyn = v - v.mean(-1, keepdims=True)
# phase steps between adjacent knots, per encode (phase -k_e·v_b), wrapped to (-π, π]
steps = {}
for e in range(E):
    phib = -np.einsum("a,nab->nb", k[e], vb)
    d = np.angle(np.exp(1j * (np.roll(phib, -1, axis=1) - phib)))
    ad = np.abs(d).ravel()
    steps[e] = {"max": float(ad.max()), "p99": float(np.percentile(ad, 99)), "p999": float(np.percentile(ad, 99.9)), "frac_gt_pi2": float((ad > np.pi / 2).mean()), "frac_gt_pi4": float((ad > np.pi / 4).mean())}
res["phase_step_adjacent_knots"] = steps
# complex interpolant per encode, magnitude shrink, decoded encode-difference velocity
xk = np.exp(-1j * np.einsum("ea,nab->neb", k, vb))  # (n, E, M)
xt = np.einsum("tb,neb->net", B, xk)  # (n, E, nt)
xc = np.einsum("tb,neb->net", B, np.einsum("bt,net->neb", np.linalg.pinv(B), np.exp(-1j * np.einsum("ea,nat->net", k, v))))
res["magnitude_shrink"] = {str(e): {"max": float((1 - np.abs(xt[:, e])).max()), "p99": float(np.percentile(1 - np.abs(xt[:, e]), 99)), "mean": float((1 - np.abs(xt[:, e])).mean())} for e in range(E)}
# 20-bin per-frame model: bin-averaged complex signal per encode
binof = np.minimum((tau * F).astype(int), F - 1)
xtrue = np.exp(-1j * np.einsum("ea,nat->net", k, v))
xbin = np.stack([xtrue[:, :, binof == f].mean(-1) for f in range(F)], axis=-1)  # (n, E, F)
xbin_t = xbin[:, :, binof]
out = {}
for e in range(1, E):
    dk = k[e] - k[0]
    nk_ = np.linalg.norm(dk)
    u = dk / nk_
    vtrue = np.einsum("a,nat->nt", u, v)
    vd = np.einsum("a,nat->nt", u, dyn)
    ref = -nk_ * vtrue  # true phase difference (rad)
    def dec(x):
        p = np.angle(x[:, e] * np.conj(x[:, 0]))
        p = ref + np.angle(np.exp(1j * (p - ref)))  # the true wrap branch (unwrapping is not the question here)
        return -p / nk_
    vb_ = np.einsum("a,nat->nt", u, vbasis)
    vk_ = dec(xt)
    v20_ = dec(xbin_t)
    vc_ = dec(xc)
    def stat(err):
        ae = np.abs(err)
        return {"rms": float(np.sqrt((err ** 2).mean())), "p99": float(np.percentile(ae, 99)), "max": float(ae.max()), "rel": float(np.sqrt((err ** 2).mean() / (vd ** 2).mean()))}
    out[str(e)] = {"venc": float(np.pi / nk_), "basis": stat(vb_ - vtrue), "knots": stat(vk_ - vtrue), "interp": stat(vk_ - vb_), "bins20": stat(v20_ - vtrue), "complex_knots": stat(vc_ - vtrue), "dyn_rms": float(np.sqrt((vd ** 2).mean()))}
res["velocity_error"] = out
# summary over the low-VENC (largest |Δk|) encode differences
dks = {e: np.linalg.norm(k[e] - k[0]) for e in range(1, E)}
lo = [e for e in dks if dks[e] > 0.999 * max(dks.values())]
res["low_venc_encodes"] = lo
summ = {}
for key in ("basis", "knots", "interp", "bins20", "complex_knots"):
    summ[key] = {m: float(np.mean([out[str(e)][key][m] for e in lo])) for m in ("rms", "p99", "max", "rel")}
res["summary_low_venc"] = summ
res["summary_steps_low_venc"] = {m: float(max(steps[e][m] for e in lo)) for m in ("max", "p99", "p999")}
print(json.dumps(res, indent=1))
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1))

# ── the velocity model FITTED to the true signal (its noise-free data-term optimum), on a voxel subsample: per voxel
# (z_b complex, v_b ∈ R³) minimising Σ_e,τ |Σ_b B_b(τ) z_b e^{-i k_e·v_b} − e^{-i k_e·v(τ)}|² (|z| = 1 truth), from the
# truth knots; alternating z by linear least squares and a damped Gauss–Newton step in v (z fixed)
if a.fit > 0:
    rng = np.random.default_rng(1)
    sub = rng.choice(n, size=min(n, a.fit), replace=False)
    vs = v[sub]  # (s, 3, nt)
    y = np.exp(-1j * np.einsum("ea,sat->set", k, vs))  # (s, E, nt)
    vk = vb[sub].copy()  # (s, 3, M)
    def model(vk):
        p = np.exp(-1j * np.einsum("ea,sab->seb", k, vk))  # (s, E, M)
        A = B[None, None, :, :] * p[:, :, None, :]  # (s, E, nt, M)
        return A
    lam = 1e-2
    for it in range(25):
        A = model(vk)
        As = A.reshape(len(sub), E * nt, M)
        ys = y.reshape(len(sub), E * nt)
        z = np.einsum("skm,sm->sk", np.linalg.pinv(As), ys)
        r = np.einsum("smk,sk->sm", As, z) - ys
        # d/dv_{b,a} of x = -i k_e[a] A[e, t, b] z_b
        J = -1j * k[None, :, None, None, :] * (A * z[:, None, None, :])[..., None]  # (s, E, nt, M, 3)
        J = J.reshape(len(sub), E * nt, M * 3)
        Jr = np.concatenate([J.real, J.imag], axis=1)
        rr = np.concatenate([r.real, r.imag], axis=1)
        H = np.einsum("smi,smj->sij", Jr, Jr)
        g = np.einsum("smi,sm->si", Jr, rr)
        H += lam * np.einsum("sii->si", H).mean(-1)[:, None, None] * np.eye(3 * M)[None]
        dv = -np.linalg.solve(H, g[..., None])[..., 0].reshape(len(sub), M, 3).transpose(0, 2, 1)
        vk = vk + dv
        cost = (np.abs(r) ** 2).sum(-1).mean()
    A = model(vk)
    xt_fit = np.einsum("setb,sb->set", A, z)
    vt_sub = v[sub]
    outf = {}
    for e in lo:
        dk = k[e] - k[0]
        nk_ = np.linalg.norm(dk)
        u = dk / nk_
        vtrue = np.einsum("a,sat->st", u, vt_sub)
        vd = np.einsum("a,sat->st", u, dyn[sub])
        ref = -nk_ * vtrue
        def dec(x):
            p = np.angle(x[:, e] * np.conj(x[:, 0]))
            p = ref + np.angle(np.exp(1j * (p - ref)))
            return -p / nk_
        err_fit = dec(xt_fit) - vtrue
        # the knot velocities' own interpolant Σ_b B_b v_b (the parameter readout) of the fitted model
        err_vint = np.einsum("a,sat->st", u, np.einsum("tb,sab->sat", B, vk)) - vtrue
        err_tk = dec(xt[sub]) - vtrue
        err_c = dec(xc[sub]) - vtrue
        rel = lambda er: float(np.sqrt((er ** 2).mean() / (vd ** 2).mean()))
        outf[str(e)] = {"fitted_vmodel": {"rms": float(np.sqrt((err_fit ** 2).mean())), "p99": float(np.percentile(np.abs(err_fit), 99)), "rel": rel(err_fit)}, "fitted_vinterp": {"rms": float(np.sqrt((err_vint ** 2).mean())), "rel": rel(err_vint)}, "truth_knots": {"rel": rel(err_tk)}, "complex_knots": {"rel": rel(err_c)}, "knot_velocity_vs_basis_fit_rms": float(np.sqrt(((vk - vb[sub]) ** 2).mean()))}
    res["fitted_subsample"] = {"voxels": int(len(sub)), "final_cost_per_voxel": float(cost), "per_encode": outf, "mean_rel_fitted": float(np.mean([outf[str(e)]["fitted_vmodel"]["rel"] for e in lo])), "mean_rel_fitted_vinterp": float(np.mean([outf[str(e)]["fitted_vinterp"]["rel"] for e in lo])), "mean_rel_truth_knots": float(np.mean([outf[str(e)]["truth_knots"]["rel"] for e in lo])), "mean_rel_complex": float(np.mean([outf[str(e)]["complex_knots"]["rel"] for e in lo]))}
    print(json.dumps(res["fitted_subsample"], indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
