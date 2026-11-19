"""Coil eigen-spectra of the phantom's true maps (whitened) for research/coil_cost_reduction.md §2 and §6.

    rustlib/.venv/bin/python -I research/experiments/coil_sketch/spectrum.py [scan_dir] [box_scan_dir]

- Grams over the body (map energy), signal-weighted (Σ ρ² s sᴴ ≈ the calibration Gram), heart box, out-of-box.
- Global energy-ordered virtual coils (signal-weighted Gram, = recon_p2np --coil-gram calib up to the low-res ρ).
- What the box sees of the global coils, box-PCA, ROVir (Kim et al. MRM 2021: generalized eig G_box u = μ G_out u).
- Saves out/spectrum.json and out/slice.npz (one axial slice of the 30 global virtual coils + truth) for toy.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute.store import load_any  # noqa: E402

SC = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan"
BX = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min_box1mm_b/scan"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
NV = 30  # production virtual coil count (99.9 % on this phantom)
ZS = 57  # axial slice for the toy (box centre z)

meta = json.loads((SC / "meta.json").read_text())
bmeta = json.loads((BX / "meta.json").read_text())
dims = np.array(meta["recon_dims"])
res = np.array(meta["res"])
c0 = np.array(meta["presc_center_subject"]) - (dims // 2) * res


def box_slices(center, half):
    lo = np.ceil((center - half - c0) / res).astype(int)
    hi = np.floor((center + half - c0) / res).astype(int) + 1
    lo, hi = np.maximum(lo, 0), np.minimum(hi, dims)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


bc, bf = np.array(bmeta["presc_center_subject"]), np.array(bmeta["fov"])
BOXES = {
    "fov": box_slices(bc, bf / 2),             # the box scan's FOV (heart + 15 mm + 15 mm margin)
    "box": box_slices(bc, bf / 2 - 0.015),     # heart + great vessels bbox dilated 15 mm (--keep-box heart:15)
    "heart": box_slices(bc, bf / 2 - 0.030),   # the bbox itself
}

S = load_any(SC / "smaps_true.hv")  # (x, y, z, 32) complex64
psi = np.array(meta["psi"])
L = np.linalg.cholesky(psi)
W = np.linalg.inv(L).T.astype(np.complex64)  # row vectors: s_wᵀ = sᵀ L⁻ᵀ
nc = S.shape[-1]
S = S.reshape(-1, nc)
for i in range(0, S.shape[0], 1 << 20):
    S[i:i + (1 << 20)] = S[i:i + (1 << 20)] @ W
S = S.reshape(*dims, nc)
rho = (np.load(SC / "truth_pd_water.npy") + np.load(SC / "truth_pd_fat.npy")).astype(np.float32)
body = rho > 0.05 * rho.max()


def gram(mask, wt=None):
    idx = np.flatnonzero(mask)
    G = np.zeros((nc, nc), np.complex128)
    Sf = S.reshape(-1, nc)
    for i in range(0, idx.size, 1 << 19):
        j = idx[i:i + (1 << 19)]
        a = Sf[j]
        b = a if wt is None else a * wt.reshape(-1)[j, None]
        G += (b.conj().T @ a).astype(np.complex128)
    return G.conj()  # Σ s sᴴ with s a column


def counts(ev, levels=(0.99, 0.995, 0.999, 0.9995, 0.9999)):
    c = np.cumsum(ev) / ev.sum()
    return {f"{l:g}": int(np.searchsorted(c, l) + 1) for l in levels}


def eigdesc(G):
    ev, U = np.linalg.eigh(G)
    return ev[::-1].clip(0), U[:, ::-1]


R = {}
UB = {}
inbox = {k: np.zeros(dims, bool) for k in BOXES}
for k, sl in BOXES.items():
    inbox[k][sl] = True
    R[f"{k}_slices"] = [[s.start, s.stop] for s in sl]
w2 = rho ** 2
G_map = gram(body)
G_sig = gram(body, w2)
ev_map, _ = eigdesc(G_map)
ev_sig, U = eigdesc(G_sig)
R["map_frac"] = (ev_map / ev_map.sum()).tolist()
R["sig_frac"] = (ev_sig / ev_sig.sum()).tolist()
R["map_counts"], R["sig_counts"] = counts(ev_map), counts(ev_sig)
Uv = U[:, :NV]  # global energy-ordered virtual coils (columns)

for k in BOXES:
    Gb = gram(body & inbox[k], w2)
    Go = gram(body & ~inbox[k], w2)
    evb, Ub = eigdesc(Gb)
    R[f"{k}_pca_frac"] = (evb / evb.sum()).tolist()
    R[f"{k}_pca_counts"] = counts(evb)
    # global virtual coils: box energy captured by the first n
    Gb_v = Uv.conj().T @ Gb @ Uv
    d = np.real(np.diag(Gb_v))
    R[f"{k}_global_cum"] = (np.cumsum(d) / np.real(np.trace(Gb))).tolist()
    # box-PCA with n coils: in-box kept, out-of-box carried (relative to in-box energy) -> must be modelled
    tb, to = np.real(np.trace(Gb)), np.real(np.trace(Go))
    rows = []
    for n in (4, 6, 8, 10, 12, 16, 20, 24, 30):
        P = Ub[:, :n]
        kin = np.real(np.trace(P.conj().T @ Gb @ P)) / tb
        kout = np.real(np.trace(P.conj().T @ Go @ P))
        # ROVir: generalized eigvecs of (Gb, Go + δ I)
        mu, V = sla.eigh(Gb, Go + 1e-6 * np.real(np.trace(Go)) / nc * np.eye(nc))
        V = V[:, ::-1][:, :n]
        Q, _ = np.linalg.qr(V)  # span (the beamformed coils are re-orthonormalised; data stay white)
        rin = np.real(np.trace(Q.conj().T @ Gb @ Q)) / tb
        rout = np.real(np.trace(Q.conj().T @ Go @ Q))
        rows.append(dict(n=n, pca_in=kin, pca_out_over_in=kout / (kin * tb), pca_out_frac=kout / to,
                         rovir_in=rin, rovir_out_over_in=rout / (rin * tb), rovir_out_frac=rout / to))
    R[f"{k}_table"] = rows
    R[f"{k}_in_over_out_energy"] = tb / to
    # local loss of the global first-n compression inside the box (shading risk): 1 − ‖P_n s‖²/‖s‖² percentiles
    m = body & inbox[k]
    Sb = S[m]
    tot = np.sum(np.abs(Sb) ** 2, 1)
    proj = Sb @ Uv.conj()  # coefficients on the global virtual coils, in energy order
    cum = np.cumsum(np.abs(proj) ** 2, 1) / tot[:, None]
    R[f"{k}_local_loss_p50_p99_max"] = {n: [float(np.percentile(1 - cum[:, n - 1], q)) for q in (50, 99, 100)]
                                       for n in (8, 12, 16, 20, 24, 30)}
    projb = Sb @ Ub.conj()  # box-PCA coefficients
    cumb = np.cumsum(np.abs(projb) ** 2, 1) / tot[:, None]
    R[f"{k}_boxpca_local_loss_p50_p99_max"] = {n: [float(np.percentile(1 - cumb[:, n - 1], q)) for q in (50, 99, 100)]
                                              for n in (8, 11, 14, 18, 22)}
    UB[k] = Ub
    del Sb, proj, cum, projb, cumb

# whole-body local loss too
Sb = S[body]
tot = np.sum(np.abs(Sb) ** 2, 1)
cum = np.cumsum(np.abs(Sb @ Uv.conj()) ** 2, 1) / tot[:, None]
R["body_local_loss_p50_p99_max"] = {n: [float(np.percentile(1 - cum[:, n - 1], q)) for q in (50, 99, 100)]
                                    for n in (8, 12, 16, 20, 24, 30)}
del Sb, cum

# calibration Gram as recon_p2np --coil-gram calib builds it (Parseval: Σ over the low-res coil images), noise-free:
# central CAL³ k-space cube of ρ s_c (ρ = magnitude proton density, so this is a lower bound on the concentration)
import scipy.fft as sfft  # noqa: E402
CAL = 24
Y = np.zeros((nc, CAL, CAL, CAL), np.complex64)
cs = [slice(d // 2 - CAL // 2, d // 2 + CAL // 2) for d in dims]
for c in range(nc):
    k = sfft.fftshift(sfft.fftn(S[..., c] * rho, workers=8))
    Y[c] = k[cs[0], cs[1], cs[2]]
Yf = Y.reshape(nc, -1)
ev_cal, U_cal = eigdesc(Yf @ Yf.conj().T)
R["calib_frac"] = (ev_cal / ev_cal.sum()).tolist()
R["calib_counts"] = counts(ev_cal)
# energy of the true maps (signal-weighted) captured by the calib ordering
d = np.real(np.diag(U_cal.conj().T @ G_sig @ U_cal))
R["calib_order_sig_cum"] = (np.cumsum(d) / np.real(np.trace(G_sig))).tolist()

(OUT / "spectrum.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

# slice for the toy: the 30 global virtual coils (whitened), truth, masks
Ssl = S[:, :, ZS, :] @ Uv.conj()  # (x, y, 30): virtual map v = u_vᴴ s
lum = np.load(SC / "truth_lumen_fraction.npy")[:, :, ZS]
vel = load_any(SC / "truth_vel_logical.hv", lazy=True)[:, :, ZS, :, :]  # (x, y, 3, frames)
np.savez_compressed(OUT / "slice.npz", maps=Ssl.astype(np.complex64), raw=S[:, :, ZS, :].astype(np.complex64), U_sig=U, U_cal=U_cal,
                    U_heart=UB["heart"], U_box=UB["box"], rho=rho[:, :, ZS], lumen=lum, vel=np.asarray(vel, np.float32),
                    body=body[:, :, ZS], box=inbox["box"][:, :, ZS], heart=inbox["heart"][:, :, ZS], venc=meta["venc"])
print(json.dumps({k: R[k] for k in ("map_counts", "sig_counts", "box_pca_counts", "heart_pca_counts", "fov_pca_counts")}, indent=0))
print("calib", R["calib_counts"], np.round(R["calib_frac"], 4).tolist())
print("calib order: sig energy cum@8,12,16,20,24,30", [round(R["calib_order_sig_cum"][n - 1], 4) for n in (8, 12, 16, 20, 24, 30)])
print("sig_frac", np.round(R["sig_frac"][:20], 4).tolist())
print("map_frac", np.round(R["map_frac"][:20], 4).tolist())
for k in BOXES:
    print(k, "in/out energy", round(R[f"{k}_in_over_out_energy"], 4), "global cum@8,12,16,20,24,30",
          [round(R[f"{k}_global_cum"][n - 1], 5) for n in (8, 12, 16, 20, 24, 30)])
    for r in R[f"{k}_table"]:
        print("  ", {a: (round(b, 5) if isinstance(b, float) else b) for a, b in r.items()})
    print("  local loss (global first n)", R[f"{k}_local_loss_p50_p99_max"])
    print("  local loss (box-PCA first n)", R[f"{k}_boxpca_local_loss_p50_p99_max"])
print("body local loss", R["body_local_loss_p50_p99_max"])
