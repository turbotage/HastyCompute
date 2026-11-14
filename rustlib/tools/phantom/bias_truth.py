"""Three velocity truths of a `simphantom --beats` scan on its reconstruction grid (research/velocity_bias.md §3):

    rustlib/.venv/bin/python -I rustlib/tools/phantom/bias_truth.py <phantom_dir> <sim_dir> [--resp mean|0|R]
        [--signal] [--signal-frames [--frames-out NAME.npz]]

- u_vol: volume-mean blood velocity (Σ w u / Σ w over the LBM cells in the voxel, the scan's heartbeats, the frame's
  phase window) — the gated ensemble mean;
- u_sig: signal-weighted mean κ1 = Σ w s u / Σ w s (s = M_z / M0 from the library's spin history; = u_vol without);
- u_pc:  the ideal phase-contrast measurement: per encode e the noise-free signal C_e = Σ w s e^{i k_e·u} over the six
  moment-matched sub-sources of every cell (as simphantom's bsig), averaged like the data (over heartbeats and the
  phase window), phase differences to encode 0, decoded by least squares on Δk_e with the wraps chosen against u_vol.
Encodes k_e = γ M1_e from the scan's meta (the played first moments at TE). The LBM runs in the cardiac frame: the
reconstruction voxels are mapped there through the phantom's respiratory field at the scan's mean breathing state
(like sim_truth). Only voxels the LBM covers (coverage ≥ 0.5) are defined (NaN elsewhere).
Writes <sim_dir>/truth_bias.npz: u_vol, u_sig, u_pc [X, Y, Z, 3, F] (logical axes, m/s), coverage [X, Y, Z, F],
plus figures in <sim_dir>/figs_bias/.
`--signal`: also <sim_dir>/truth_pc_signal.npz: c_mean [X, Y, Z, E] complex64, the frame-mean of the ideal signal C_e
(equal frame weights: what a reconstruction pooling every frame sees, spin history and intravoxel velocity spread
included; the sim's sign, e^{−i k·u}), and c0_abs [X, Y, Z, F] float32 = |C_0| per frame. Voxel signal = Σ over the
fine sub-voxels (the 2×2×2 box sum, coverage-weighted); 0 where the LBM does not cover the voxel.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
sys.path.insert(0, str(HERE))
from hastycompute import value_format as vf  # noqa: E402
from lbm_beats import BeatLibrary, cov_mat  # noqa: E402

A = sys.argv[1:]
PH, SIM = Path(A[0]), Path(A[1])
resp = A[A.index("--resp") + 1] if "--resp" in A else "mean"
want_signal = "--signal" in A
# --signal-frames: also the per-frame ideal signal C_e of every voxel the LBM covers in some frame (sparse):
# <sim_dir>/truth_pc_frames.npz with idx (flat F-order indices on the reconstruction grid), c [n, E, F] complex64
# (the sim's sign e^{-i k.u}; Σ over the 2×2×2 fine sub-voxels, coverage-weighted like c_mean), cov [n, F]
want_frames = "--signal-frames" in A
# --resp R (a number): the truth at breathing state R (e.g. a respiratory bin's mean bellows amplitude) instead of the
# scan mean; --frames-out NAME.npz writes the per-frame signal there instead of truth_pc_frames.npz
frames_out = A[A.index("--frames-out") + 1] if "--frames-out" in A else "truth_pc_frames.npz"
meta = json.loads((SIM / "meta.json").read_text())
assert meta.get("beats"), "not a --beats scan"
br = Path(meta["beats"]["run"])
L = BeatLibrary(br if br.is_absolute() else REPO / br)
GAMMA = 2 * np.pi * 42.577478e6
enc = meta["sequence"]["encoding"]["at_te"]
K = GAMMA * np.array([e["m1"] for e in enc])  # [E, 3] rad / (m/s), lab axes
E = len(K)
dK = K[1:] - K[0]
F = int(meta["n_frames"])
assert np.allclose(np.array(meta["rot_presc"]), np.eye(3)), "oblique prescriptions not handled"

# heartbeat → library beat: the recorded sequence, or the simulator's shuffle regenerated
seq = meta["beats"].get("sequence")
if seq is None:
    n_lib = min(len(L.beats), 30)
    n_hb = int(np.ceil(meta["scan_time"] / meta["mean_rr"])) + 1
    # hasty_bench::beats::beat_sequence (xorshift*; same constants)
    M64 = (1 << 64) - 1
    s = ((meta["beats"].get("seed", 1) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) & M64) | 1

    def nxt():
        global s
        s ^= s >> 12
        s ^= (s << 25) & M64
        s ^= s >> 27
        return ((s * 0x2545F4914F6CDD1D) & M64) >> 11

    seq = []
    while len(seq) < n_hb:
        p = list(range(n_lib))
        for i in range(n_lib - 1, 0, -1):
            j = nxt() % (i + 1)
            p[i], p[j] = p[j], p[i]
        if n_lib > 1 and seq and seq[-1] == p[0]:
            p[0], p[-1] = p[-1], p[0]
        seq += p
    seq = seq[:n_hb]
counts = np.bincount(seq, minlength=len(L.beats)).astype(float)
print(f"{SIM}: {len(seq)} heartbeats over {int((counts > 0).sum())} library beats of {L.run}, {E} encodes, {F} frames")

# frame windows: bin b covers phase [b, b+1)/nb, frame f [f, f+1)/F
nb = L.bins
W = np.zeros((F, nb))
for f in range(F):
    for b in range(nb):
        lo, hi = max(b / nb, f / F), min((b + 1) / nb, (f + 1) / F)
        W[f, b] = max(hi - lo, 0) * nb

# per frame, per cell (dense cell grid) the extensive sums: w, w u (3), w s, w s u (3), w C_e (E complex)
o_c, h_c, dims_c = L.grid()
X_, Y_, Z_ = (int(d) for d in dims_c)
ncell = X_ * Y_ * Z_
acc = {"w": np.zeros((F, ncell), np.float32), "wu": np.zeros((F, 3, ncell), np.float32), "ws": np.zeros((F, ncell), np.float32),
       "wsu": np.zeros((F, 3, ncell), np.float32), "wc": np.zeros((F, E, ncell), np.complex64)}


def cell_signals(v):
    """records [n, nf] → w, mean, s̄, E[s u], and C_e = Σ_q p_q s_q e^{i k_e·u_q} over the six sub-sources"""
    w = v[:, 0]
    mu = v[:, 1:4]
    C = cov_mat(v[:, 4:10].T).transpose(2, 0, 1)
    lam, Ev = np.linalg.eigh(C)
    lam = np.maximum(lam, 0.0)
    from lbm_beats import m3_tensor
    T3 = m3_tensor(v[:, 10:20].T).transpose(3, 0, 1, 2)
    sk = np.einsum("nijk,nia,nja,nka->na", T3, Ev, Ev, Ev)
    sd = np.sqrt(lam)
    g = np.clip(np.where(sd > 1e-6, sk / np.maximum(sd, 1e-12) ** 3, 0.0), -3, 3)
    rr = np.sqrt(g * g / 4 + 3)
    xp, xm = sd * (g / 2 + rr), sd * (g / 2 - rr)
    wp = np.where(sd > 1e-6, lam / np.maximum(xp * (xp - xm), 1e-30), 1 / 6)
    wm = np.where(sd > 1e-6, -wp * xp / np.minimum(xm, -1e-30), 1 / 6)
    sm = v[:, 23] if v.shape[1] > 23 else np.ones(len(v))
    csu = v[:, 25:28] if v.shape[1] > 25 else np.zeros((len(v), 3))
    Cs = np.zeros((len(v), E), np.complex128)
    for a in range(3):
        ea = Ev[:, :, a]
        gr = np.where(lam[:, a] > 1e-8, (csu * ea).sum(1) / np.maximum(lam[:, a], 1e-8), 0.0)
        for x_, w_ in ((xp[:, a], wp[:, a]), (xm[:, a], wm[:, a])):
            u_ = mu + x_[:, None] * ea
            s_ = np.clip(sm + gr * x_, 0, 1)
            Cs += (w_ * s_)[:, None] * np.exp(1j * (u_ @ K.T))
    esu = csu + sm[:, None] * mu
    return w, mu, sm, esu, Cs


for k in np.flatnonzero(counts):
    data = L.read(L.beats[k])
    for b, (idx, v, _) in data.items():
        wf = W[:, b] * counts[k]
        fs = np.flatnonzero(wf)
        if not len(fs):
            continue
        cells = ((L.gz[idx] - L.lo[2]) * Y_ + (L.gy[idx] - L.lo[1])) * X_ + (L.gx[idx] - L.lo[0])
        w, mu, sm, esu, Cs = cell_signals(v)
        for f in fs:
            c = wf[f]
            acc["w"][f, cells] += c * w
            acc["wu"][f][:, cells] += (c * w)[None] * mu.T
            acc["ws"][f, cells] += c * w * sm
            acc["wsu"][f][:, cells] += (c * w)[None] * esu.T
            acc["wc"][f][:, cells] += ((c * w)[:, None] * Cs).T
    print(f"  beat {L.beats[k]} (× {int(counts[k])}) done")
norm = (W.sum(1) * counts.sum())[:, None]
for key in acc:
    acc[key] = acc[key] / (norm if acc[key].ndim == 2 else norm[:, :, None])

# reconstruction grid (fine ×2) → cardiac frame
center, res, dims = np.array(meta["presc_center_subject"]), float(meta["res"][0]), np.array(meta["recon_dims"])
fine = res / 2
lo = center - (dims // 2) * res - res / 2
# crop to the LBM cell box (+ one voxel)
cb_lo, cb_hi = o_c - h_c, o_c + dims_c * h_c
i0 = np.maximum(np.floor((cb_lo - lo) / res).astype(int) - 1, 0)
i1 = np.minimum(np.ceil((cb_hi - lo) / res).astype(int) + 1, dims)
nf_ = (i1 - i0) * 2
gx, gy, gz = (lo[a] + i0[a] * res + (np.arange(nf_[a]) + 0.5) * fine for a in range(3))
Zf, Yf, Xf = np.meshgrid(gz, gy, gx, indexing="ij")
P = np.stack([Xf, Yf, Zf], -1)
r_state = 0.0
if resp == "mean":
    r_state = (meta.get("truth") or {}).get("resp_state")
    if r_state is None:
        SI = vf.read(SIM.parent / "sim_input.hv")
        r_state = float(np.asarray(SI["breathing"]["r"])[: int(meta["scan_time"] / 1e-3)].mean())
elif resp != "0":
    r_state = float(resp)
if r_state:
    rm = vf.read(PH / "motion" / "resp_motion.hv")
    Bf = np.asarray(rm["B"]).sum(0)  # [3, Z, Y, X]
    ro, rh = np.asarray(rm["origin_m"], float), float(rm["voxel_m"])

    def S(Q):
        idx = [(Q[..., a] - ro[a]) / rh - 0.5 for a in (2, 1, 0)]
        return np.stack([ndimage.map_coordinates(Bf[c], idx, order=1, mode="nearest") for c in range(3)], -1)

    Xc = P.copy()
    for _ in range(6):
        Xc = P - r_state * S(Xc)
    P = Xc
cidx = [(P[..., a] - o_c[a]) / h_c for a in (2, 1, 0)]


def sample(field):
    """dense cell field [ncell] (complex or real) at the fine points, then the 2×2×2 box sum → recon voxels"""
    fz = field.reshape(Z_, Y_, X_)
    if np.iscomplexobj(fz):
        out = ndimage.map_coordinates(fz.real, cidx, order=1, mode="constant") + 1j * ndimage.map_coordinates(fz.imag, cidx, order=1, mode="constant")
    else:
        out = ndimage.map_coordinates(fz, cidx, order=1, mode="constant")
    z, y, x = out.shape
    return out.reshape(z // 2, 2, y // 2, 2, x // 2, 2).sum((1, 3, 5))


shp = tuple(dims) + (3, F)
u_vol, u_sig, u_pc = (np.full(shp, np.nan, np.float32) for _ in range(3))
cov = np.zeros(tuple(dims) + (F,), np.float32)
sl = tuple(slice(i0[a], i1[a]) for a in range(3))
Kinv = np.linalg.pinv(dK)
if want_frames:
    box = tuple(i1 - i0)
    c_box = np.zeros(box + (E, F), np.complex64)
    w_box = np.zeros(box + (F,), np.float32)
if want_signal:
    c_mean = np.zeros(tuple(dims) + (E,), np.complex64)
    c0_abs = np.zeros(tuple(dims) + (F,), np.float32)
for f in range(F):
    w = sample(acc["w"][f])
    ok = w / 8 >= 0.5
    wu = np.stack([sample(acc["wu"][f, c]) for c in range(3)], -1)
    ws = sample(acc["ws"][f])
    wsu = np.stack([sample(acc["wsu"][f, c]) for c in range(3)], -1)
    wc = np.stack([sample(acc["wc"][f, e]) for e in range(E)], -1)
    uv = wu / np.maximum(w, 1e-12)[..., None]
    us = wsu / np.maximum(ws, 1e-12)[..., None]
    dphi = np.angle(wc[..., 1:] * np.conj(wc[..., :1]))  # [z, y, x, E-1]
    n = np.round(((uv @ dK.T) - dphi) / (2 * np.pi))
    up = (dphi + 2 * np.pi * n) @ Kinv.T
    for arr, val in ((u_vol, uv), (u_sig, us), (u_pc, up)):
        tgt = np.where(ok[..., None], val, np.nan).transpose(2, 1, 0, 3)  # [x, y, z, 3]
        arr[sl + (slice(None), f)] = tgt
    cov[sl + (f,)] = (w / 8).transpose(2, 1, 0)
    if want_frames:
        c_box[..., f] = np.conj(wc).transpose(2, 1, 0, 3).astype(np.complex64)
        w_box[..., f] = (w / 8).transpose(2, 1, 0)
    if want_signal:
        # the sim's convention e^{−i k·u}: conjugate of C_e here
        c_mean[sl] += (np.conj(wc) / F).transpose(2, 1, 0, 3).astype(np.complex64)
        c0_abs[sl + (f,)] = np.abs(wc[..., 0]).transpose(2, 1, 0)
if not ((want_signal or want_frames) and (SIM / "truth_bias.npz").exists()):  # --signal on a sim that has its truths: keep the file
    np.savez_compressed(SIM / "truth_bias.npz", u_vol=u_vol, u_sig=u_sig, u_pc=u_pc, coverage=cov, k=K, sequence=np.asarray(seq), resp_state=r_state)
if want_frames:
    anyc = w_box.max(-1) > 0
    loc = np.nonzero(anyc)
    gidx = [loc[a] + i0[a] for a in range(3)]
    flat = gidx[0] + dims[0] * (gidx[1] + dims[1] * gidx[2])
    np.savez_compressed(SIM / frames_out, idx=flat.astype(np.int64), c=c_box[loc], cov=w_box[loc], k=K, dims=dims, resp_state=r_state)
    print(f"wrote {SIM / frames_out}: {flat.size} voxels x {E} encodes x {F} frames (breathing state {r_state})")
if want_signal:
    np.savez_compressed(SIM / "truth_pc_signal.npz", c_mean=c_mean, c0_abs=c0_abs, k=K, resp_state=r_state)
    print(f"wrote {SIM / 'truth_pc_signal.npz'}")

if frames_out != "truth_pc_frames.npz":  # an extra truth (other breathing state): keep the scan's figures
    sys.exit(0)

# summary + figure
m = np.isfinite(u_vol[..., 0, :])
d_sig = np.linalg.norm(u_sig - u_vol, axis=3)[m]
d_pc = np.linalg.norm(u_pc - u_sig, axis=3)[m]
sp = np.linalg.norm(u_vol, axis=3)[m]
print(f"covered voxel-frames {int(m.sum())}: |u_vol| median {np.median(sp):.3f} m/s; signal weighting |κ1 − ū| median {np.median(d_sig) * 100:.2f} cm/s "
      f"(p95 {np.percentile(d_sig, 95) * 100:.2f}); PC vs κ1 |v̂ − κ1| median {np.median(d_pc) * 100:.2f} cm/s (p95 {np.percentile(d_pc, 95) * 100:.2f})")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

(SIM / "figs_bias").mkdir(exist_ok=True)
fpk = int(np.nanargmax([np.nanmean(np.linalg.norm(u_vol[..., f], axis=3)) for f in range(F)]))
zc = int(np.nanargmax(np.isfinite(u_vol[..., 0, fpk]).sum((0, 1))))
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for ax, (img, t, cm, lim) in zip(axs, [(np.linalg.norm(u_vol[:, :, zc, :, fpk], axis=2), "|ū| (m/s)", "magma", (0, 1.5)),
                                       (np.linalg.norm(u_sig[:, :, zc, :, fpk] - u_vol[:, :, zc, :, fpk], axis=2) * 100, "|κ1 − ū| (cm/s): signal weighting", "viridis", (0, 5)),
                                       (np.linalg.norm(u_pc[:, :, zc, :, fpk] - u_sig[:, :, zc, :, fpk], axis=2) * 100, "|v̂_PC − κ1| (cm/s): skew + mixing", "viridis", (0, 5)),
                                       (cov[:, :, zc, fpk], "LBM coverage", "gray", (0, 1))]):
    im = ax.imshow(img.T, origin="lower", cmap=cm, vmin=lim[0], vmax=lim[1])
    ax.set_title(f"{t}, frame {fpk}, z {zc}", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.75)
fig.savefig(SIM / "figs_bias" / "bias_truths.png", dpi=100, bbox_inches="tight")
print(f"wrote {SIM / 'truth_bias.npz'}, {SIM / 'figs_bias'}")
