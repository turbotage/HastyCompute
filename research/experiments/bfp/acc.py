"""BFP storage accuracy along the segmented-NUFFT gradient chain (research/bfp_storage.md §3).

Real phantom ph0001 (sim_flow_10min/scan): image 254 x 144 x 114 (x, y, z), fine grid 320 x 180 x 144, true coil maps
(32 coils) compressed to 10 virtual coils, ratemap = recon field map + R2* truth, L = 10 segments. One coil batch of
5 virtual coils x 10 segments = 50 channels, grid layout [z][y][x][channel] (channel la = c * L + l fastest), as the
engine. BFP blocks = 16 consecutive complex values (32 reals) of each buffer's flat index (the codec of hasty-bfp,
emulated in f64: same exponent rule, ties-to-even rounding, saturation; the f32 input rounding is not applied, it is
the 'f32' row).

Compute in f64. Every buffer's quantisation error is measured on the f64 reference chain (the transforms after a stored
buffer are unitary up to scale, so the relative error injected at a stage is the relative error it contributes to the
output); a fully chained run (every buffer quantised, errors propagated) checks that the stage errors add in quadrature.

Phases (argv): image | forward | adjoint | kspace | e2e | all. Writes acc_<phase>.json next to this file.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.fft as sf

sys.path.insert(0, "/home/turbotage/Documents/GitHub/HastyCompute/python/lib")
import hastycompute.store as hs  # noqa: E402

D = "/home/turbotage/Documents/GitHub/HastyCompute/data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan/"
OUT = os.path.dirname(os.path.abspath(__file__))
BITS = [8, 10, 12, 14, 16, 18, 20, 22, 24]
L = 10          # segments
NVC = 10        # virtual coils
CB = 5          # coil batch
GAMMA_HZ_PPM = 42.577478 * 3.0  # Hz per ppm at 3 T
T0, T1 = 0.5e-3, 8.5e-3          # segment time window (s): ADC 3990 x 2 us after excitation
WK = 2


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# codec emulation

def q_bfp(a, b):
    """BFP round trip of a complex128 / float64 array (flat blocks of 32 reals), in chunks; returns a new array"""
    out = np.empty_like(a)
    fv, ov = a.reshape(-1).view(np.float64), out.reshape(-1).view(np.float64)
    n = fv.size
    nb = n // 32
    lim = 2.0 ** (b - 1) - 1
    CH = 1 << 22
    for s in range(0, nb, CH):
        v = fv[s * 32:min(nb, s + CH) * 32].reshape(-1, 32)
        m = np.abs(v).max(1)
        _, e = np.frexp(m)                    # m < 2^e (m > 0); frexp(0) = 0
        e = np.maximum(e, b - 127)
        sc = np.ldexp(1.0, b - 1 - e)
        bump = (np.rint(m * sc) > lim) & (e < 128)
        e = e + bump
        sc = np.ldexp(1.0, b - 1 - e)[:, None]
        ov[s * 32:min(nb, s + CH) * 32] = (np.clip(np.rint(v * sc), -lim, lim) / sc).reshape(-1)
    if n % 32:
        t = fv[nb * 32:]
        pad = np.zeros(32)
        pad[:t.size] = t
        ov[nb * 32:] = q_bfp(pad, b)[:t.size]
    return out


def q_f32(a):
    return a.astype(np.complex64).astype(np.complex128) if np.iscomplexobj(a) else a.astype(np.float32).astype(np.float64)


def q_mant(a, bits):
    """per-value float with `bits` mantissa bits (8-bit exponent): bf16 = 7, fp24 = 15"""
    def r(v):
        f = v.astype(np.float32).view(np.uint32).astype(np.uint64)
        s = 23 - bits
        f = ((f + (1 << (s - 1))) >> s) << s
        return f.astype(np.uint32).view(np.float32).astype(np.float64)
    return r(a.real) + 1j * r(a.imag) if np.iscomplexobj(a) else r(a)


FORMATS = [("f32", q_f32), ("fp24", lambda a: q_mant(a, 15)), ("bf16", lambda a: q_mant(a, 7))] + [(f"bfp{b}", (lambda b: lambda a: q_bfp(a, b))(b)) for b in BITS]


def rel(err2, ref2):
    return float(np.sqrt(err2 / ref2))


def stage_err(a, fmts=FORMATS):
    """relative L2 error of storing `a` in each format: {name: rel}; chunked over the leading axis"""
    ref2 = float(np.vdot(a, a).real)
    res = {}
    for name, q in fmts:
        e2 = 0.0
        for s in range(0, a.shape[0], 8):
            c = a[s:s + 8]
            d = q(c) - c
            e2 += float(np.vdot(d, d).real)
        res[name] = rel(e2, ref2)
    return res


# ---------------------------------------------------------------------------------------------------------------------
# inputs

def load_inputs(seg_basis="fixed"):
    pdw = np.load(D + "truth_pd_water.npy").astype(np.float64).T  # (z, y, x)
    pdf = np.load(D + "truth_pd_fat.npy").astype(np.float64).T
    fld = np.nan_to_num(np.load(D + "field_ppm_recon.npy").astype(np.float64).T) * GAMMA_HZ_PPM
    r2 = np.load(D + "truth_r2s.npy").astype(np.float64).T
    r2 = np.clip(np.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1e4)  # s^-1 (NaN outside the body; up to 6250 in signal voxels)
    n = pdw.shape
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, k) for k in n], indexing="ij")
    # complex image: water + fat (fat at a 3 ms-like phase), smooth background phase
    img = (pdw + pdf * np.exp(1j * 2.1)) * np.exp(1j * (0.8 * x + 0.5 * y - 0.3 * z + 0.6 * x * y))
    mask = (pdw + pdf) > 0.05 * (pdw + pdf).max()
    log("image", n, "signal voxels", int(mask.sum()), "field Hz p1/p50/p99 in signal", np.percentile(fld[mask], [1, 50, 99]).round(1))
    # coil maps: 32 true coils -> 10 virtual by the signal-region Gram
    sm = hs.load_any(D + "smaps_true.hv")  # (x, y, z, 32)
    sm = np.ascontiguousarray(np.transpose(sm, (3, 2, 1, 0)))  # (32, z, y, x) complex64
    sv = sm.reshape(32, -1)[:, mask.reshape(-1)]
    G = (sv @ sv.conj().T).astype(np.complex128)
    w, U = np.linalg.eigh(G)
    U = U[:, ::-1][:, :NVC]
    w = w[::-1]
    log("virtual coils: energy kept", float(w[:NVC].sum() / w.sum()), "eig ratio last/first", float(w[NVC - 1] / w[0]))
    S = np.einsum("ck,cv->kv", U.conj(), sm.reshape(32, -1).astype(np.complex128)).reshape((NVC,) + n)
    del sm, sv
    # segment diagonals
    zr = r2 + 2j * np.pi * fld
    if seg_basis == "fixed":
        tl = np.linspace(T0, T1, L)
        P = np.exp(-zr[None] * tl[:, None, None, None])
    else:
        # rank-L SVD basis of g(v, t) = e^{-z t} on 96 times (fit on every 7th signal voxel), diagonals D = g V
        ts = np.linspace(T0, T1, 96)
        fv = zr[mask][::7]
        g = np.exp(-fv[:, None] * ts[None])
        _, sv_, vh = np.linalg.svd(g, full_matrices=False)
        V = vh[:L].conj().T  # (96, L)
        P = np.empty((L,) + n, np.complex128)
        zf = zr.reshape(-1)
        Pf = P.reshape(L, -1)
        for s in range(0, zf.size, 1 << 18):
            gg = np.exp(-zf[s:s + (1 << 18), None] * ts[None])
            Pf[:, s:s + (1 << 18)] = (gg @ V).T
        log("svd basis singular values / first", (sv_[:L + 2] / sv_[0]).round(6).tolist())
    return img, S, P, mask, fld, r2


def grid_shape(n):
    return tuple(int(np.ceil(k * 1.25 / 4) * 4) for k in n)


def hk_place(nimg, ngrid):
    """grid index of image index a on each axis: (a - N/2) mod nf (the engine's pad placement)"""
    return [(np.arange(N) - N // 2) % nf for N, nf in zip(nimg, ngrid)]


def pad_grid(img, S, P, coils, G):
    """grid[z, y, x, c*L + l] = S_c P_l x on the image block (zeros elsewhere), complex128"""
    n = img.shape
    nch = len(coils) * L
    g = np.zeros(G + (nch,), np.complex128)
    iz, iy, ix = hk_place(n, G)
    for a2 in range(n[0]):
        blk = np.empty((n[1], n[2], nch), np.complex128)
        for ci, c in enumerate(coils):
            for l in range(L):
                blk[:, :, ci * L + l] = S[c, a2] * P[l, a2] * img[a2]
        g[iz[a2]][np.ix_(iy, ix)] = blk
    return g


def crop_grid(g, S, P, coils, n):
    """x = sum_la conj(S_c P_l) grid_la on the image block"""
    G = g.shape[:3]
    iz, iy, ix = hk_place(n, G)
    out = np.zeros(n, np.complex128)
    for a2 in range(n[0]):
        blk = g[iz[a2]][np.ix_(iy, ix)]
        for ci, c in enumerate(coils):
            for l in range(L):
                out[a2] += np.conj(S[c, a2] * P[l, a2]) * blk[:, :, ci * L + l]
    return out


def fft_pass(g, axis, inverse=False, q=None):
    """in place FFT along grid axis (2 = x, 1 = y, 0 = z), optional storage round trip q of the result"""
    f = sf.ifft if inverse else sf.fft
    if axis in (1, 2):
        for s in range(0, g.shape[0], 8):
            c = f(g[s:s + 8], axis=axis, workers=WK)
            g[s:s + 8] = q(c) if q else c
    else:
        for s in range(0, g.shape[1], 8):
            c = f(g[:, s:s + 8], axis=0, workers=WK)
            g[:, s:s + 8] = q(c) if q else c


# ---------------------------------------------------------------------------------------------------------------------

def phase_image():
    """image-side buffers: x, coil maps, segment diagonals (fixed and SVD basis), and the phase alternatives"""
    res = {}
    for basis in ["fixed", "svd"]:
        img, S, P, mask, fld, r2 = load_inputs(basis)
        n = img.shape
        coils = list(range(CB))
        # pad output (50 channels) and crop of the exact pad output (Gram proxy crop(pad(x))), streamed per channel
        cr_ref = sum(np.abs(S[c] * P[l]) ** 2 * img for c in coils for l in range(L))
        pad_ref2 = float(sum((np.abs(S[c] * P[l] * img) ** 2).sum() for c in coils for l in range(L)))
        def pad_err(xi, Si, Pi):
            return rel(float(sum((np.abs(Si[c] * Pi[l] * xi - S[c] * P[l] * img) ** 2).sum() for c in coils for l in range(L))), pad_ref2)
        def crop_err(Si, Pi):
            cq = sum(np.conj(Si[c] * Pi[l]) * (S[c] * P[l] * img) for c in coils for l in range(L))
            return rel(float((np.abs(cq - cr_ref) ** 2).sum()), float((np.abs(cr_ref) ** 2).sum()))
        r = {}
        for name, q in FORMATS:
            xq, Sq, Pq = q(img), np.stack([q(s) for s in S]), np.stack([q(p) for p in P])
            row = {}
            for lab, (xi, Si, Pi) in {"x": (xq, S, P), "S": (img, Sq, P), "P": (img, S, Pq), "all": (xq, Sq, Pq)}.items():
                if basis == "svd" and lab not in ("P",):
                    continue
                row["pad_" + lab] = pad_err(xi, Si, Pi)
                if lab != "x":
                    row["crop_" + lab] = crop_err(Si, Pi)
            # the stored buffer's own relative error (signal-weighted over the image for S, P)
            row["buf_x"] = rel(float(np.vdot(xq - img, xq - img).real), float(np.vdot(img, img).real))
            w = np.abs(img) ** 2
            row["buf_S_sigw"] = rel(float((w * np.abs(Sq - S) ** 2).sum()), float((w * np.abs(S) ** 2).sum()))
            row["buf_P_sigw"] = rel(float((w * np.abs(Pq - P) ** 2).sum()), float((w * np.abs(P) ** 2).sum()))
            r[name] = row
            log(basis, name, {k: f"{v:.2e}" for k, v in row.items()})
        res[basis] = r
        if basis == "fixed":
            # alternatives for unit-modulus-times-decay diagonals: (a) b-bit fixed-point phase angle + sincos, magnitude
            # from R2* exactly; (b) on the fly in f32 from (f, R2*): e^{-(R + 2 pi i f) t_l}; (c) on the fly from
            # fixed-point turns: phase0 = frac(f t_0), dphase = frac(f dt) as u32, phase_l = phase0 + l dphase (wraps)
            alt = {}
            w = np.abs(img) ** 2
            den = float((w * np.abs(P) ** 2).sum())
            def perr(Pq):
                return rel(float((w * np.abs(Pq - P) ** 2).sum()), den)
            mag = np.abs(P)
            ang = np.angle(P) / (2 * np.pi)  # turns in (-0.5, 0.5]
            for b in BITS:
                qa = np.round(ang * 2.0 ** b) / 2.0 ** b
                alt[f"angle{b}"] = perr(mag * np.exp(2j * np.pi * qa))
            tl = np.linspace(T0, T1, L)
            f32, r32 = fld.astype(np.float32), r2.astype(np.float32)
            P32 = np.empty_like(P)
            for l in range(L):
                t = np.float32(tl[l])
                ph = (np.float32(-2 * np.pi) * f32 * t).astype(np.float32)
                P32[l] = (np.exp(-r32 * t).astype(np.float32) * (np.cos(ph).astype(np.float32) + 1j * np.sin(ph).astype(np.float32)))
            alt["onthefly_f32_radians"] = perr(P32)
            # fixed-point turns (exact wrap), then f32 sincospi of the reduced angle
            ph0 = np.mod(-fld * tl[0], 1.0)
            dph = np.mod(-fld * (tl[1] - tl[0]), 1.0)
            u0 = np.round(ph0 * 2.0 ** 32).astype(np.uint64) & 0xFFFFFFFF
            du = np.round(dph * 2.0 ** 32).astype(np.uint64) & 0xFFFFFFFF
            Pt = np.empty_like(P)
            for l in range(L):
                u = (u0 + np.uint64(l) * du) & 0xFFFFFFFF
                s = (u.astype(np.int64) - ((u >= 2 ** 31) * 2 ** 32)).astype(np.float64) / 2.0 ** 32  # turns in [-0.5, 0.5)
                a32 = (np.float32(2 * np.pi) * s.astype(np.float32)).astype(np.float32)
                t = np.float32(tl[l])
                Pt[l] = np.exp(-r32 * t).astype(np.float32) * (np.cos(a32).astype(np.float32) + 1j * np.sin(a32).astype(np.float32))
            alt["onthefly_u32_turns"] = perr(Pt)
            alt["bytes_per_voxel"] = {"stored_c32": 8 * L, "bfp_b": f"(4b+1)/16 * L", "angle_b": "b/8 * L (+ R2* 4)", "onthefly": "12 (f or phase0, dphase, R2*: 3 x 4 B) for all L"}
            res["P_alternatives"] = alt
            log("P alternatives", {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in alt.items()})
        del img, S, P
    json.dump(res, open(os.path.join(OUT, "acc_image.json"), "w"), indent=1)


def snr_sigma_k(img, S, coils):
    """per-channel k-space noise sigma (unnormalised FFT) for image SNR 20 of the coil-combined image"""
    comb = np.abs(img) * np.sqrt(sum(np.abs(S[c]) ** 2 for c in range(NVC)))
    sig = comb[comb > 0.05 * comb.max()].mean()
    return sig / 20.0 * np.sqrt(img.size)


def phase_forward():
    img, S, P, mask, fld, r2 = load_inputs("fixed")
    n = img.shape
    G = grid_shape(n)
    coils = list(range(CB))
    log("grid", G, "channels", CB * L)
    sk = snr_sigma_k(img, S, coils)
    g = pad_grid(img, S, P, coils, G)
    res = {"grid": G, "image": n, "channels": CB * L, "sigma_k": sk}
    stages = {}
    for name, ax in [("xy", 2), ("yz", 1), ("grid", 0)]:
        fft_pass(g, ax)
        t = time.time()
        stages[name] = stage_err(g)
        log("stage", name, {k: f"{v:.2e}" for k, v in stages[name].items()}, f"{time.time() - t:.0f} s")
    res["stage"] = stages
    # rms of the grid over the k-support ball (|k| <= image Nyquist on every axis), for the noise ratio
    kk = [np.abs(np.fft.fftfreq(k) * k / N) for k, N in zip(G, n)]  # in units of the image Nyquist (0.5 = edge)
    ball = (kk[0][:, None, None] ** 2 + kk[1][None, :, None] ** 2 + kk[2][None, None, :] ** 2) <= 0.25
    gb2 = float(sum(np.vdot(g[z][ball[z]], g[z][ball[z]]).real for z in range(G[0])))
    nb = int(ball.sum()) * CB * L
    res["grid_rms_ball"] = float(np.sqrt(gb2 / nb))
    res["grid_rms_all"] = float(np.sqrt(float(np.vdot(g, g).real) / g.size))
    log("grid rms in ball / sigma_k", res["grid_rms_ball"] / sk)
    # chained check: every forward buffer quantised in the chain (pad output exact), propagated
    chained = {}
    for b in [16, 20, 24]:
        q = lambda a: q_bfp(a, b)
        gq = np.zeros(G + (CB * L,), np.complex64)  # BFP values are exact in f32
        iz, iy, ix = hk_place(n, G)
        zimg = {int(z): a2 for a2, z in enumerate(iz)}
        for z in range(G[0]):  # pad (exact, f64) fused into the x pass, as the engine
            if z in zimg:
                a2 = zimg[z]
                pl = np.zeros(G[1:] + (CB * L,), np.complex128)
                blk = np.empty((n[1], n[2], CB * L), np.complex128)
                for ci, c in enumerate(coils):
                    for l in range(L):
                        blk[:, :, ci * L + l] = S[c, a2] * P[l, a2] * img[a2]
                pl[np.ix_(iy, ix)] = blk
                gq[z] = q(sf.fft(pl, axis=1, workers=WK))
        for ax in (1, 0):
            if ax == 1:
                for s0 in range(0, G[0], 8):
                    gq[s0:s0 + 8] = q(sf.fft(gq[s0:s0 + 8].astype(np.complex128), axis=1, workers=WK))
            else:
                for s0 in range(0, G[1], 8):
                    gq[:, s0:s0 + 8] = q(sf.fft(gq[:, s0:s0 + 8].astype(np.complex128), axis=0, workers=WK))
        d2 = 0.0
        for s0 in range(0, G[0], 8):
            d = gq[s0:s0 + 8].astype(np.complex128) - g[s0:s0 + 8]
            d2 += float(np.vdot(d, d).real)
        del gq
        r = rel(d2, float(np.vdot(g, g).real))
        pred = float(np.sqrt(sum(stages[s][f"bfp{b}"] ** 2 for s in stages)))
        chained[f"bfp{b}"] = {"chained": r, "quadrature_of_stages": pred, "rms_err_over_sigma_k": r * res["grid_rms_all"] / sk}
        log("chained", b, chained[f"bfp{b}"])
    res["chained"] = chained
    json.dump(res, open(os.path.join(OUT, "acc_forward.json"), "w"), indent=1)


def phase_adjoint():
    """adjoint: input = (a) the forward grid (residual ~ data, early iterations), (b) white noise on the k-support
    ball (residual ~ noise, converged); stages: scatter output (grid), z -> y, y -> x; crop with exact maps"""
    img, S, P, mask, fld, r2 = load_inputs("fixed")
    n = img.shape
    G = grid_shape(n)
    coils = list(range(CB))
    kk = [np.abs(np.fft.fftfreq(k) * k / N) for k, N in zip(G, n)]
    ball = (kk[0][:, None, None] ** 2 + kk[1][None, :, None] ** 2 + kk[2][None, None, :] ** 2) <= 0.25
    res = {}

    def make_input(kind):
        if kind == "data":
            g = pad_grid(img, S, P, coils, G)
            for ax in (2, 1, 0):
                fft_pass(g, ax)
            return g
        rng = np.random.default_rng(7)
        g = np.zeros(G + (CB * L,), np.complex128)
        for z in range(G[0]):
            m = ball[z]
            k = int(m.sum())
            g[z][m] = rng.standard_normal((k, CB * L)) + 1j * rng.standard_normal((k, CB * L))
        return g

    for kind in ["data", "noise"]:
        g = make_input(kind)
        stages = {"scatter": stage_err(g)}
        log(kind, "stage scatter", {k: f"{v:.2e}" for k, v in stages["scatter"].items()})
        for name, ax in [("zy", 0), ("yx", 1)]:
            fft_pass(g, ax, inverse=True)
            stages[name] = stage_err(g)
            log(kind, "stage", name, {k: f"{v:.2e}" for k, v in stages[name].items()})
        fft_pass(g, 2, inverse=True)
        xref = crop_grid(g, S, P, coils, n)
        del g
        chained = {}
        for b in [16, 20, 24]:
            q = lambda a: q_bfp(a, b)
            g = make_input(kind)
            for s in range(0, G[0], 8):
                g[s:s + 8] = q(g[s:s + 8])
            fft_pass(g, 0, inverse=True, q=q)
            fft_pass(g, 1, inverse=True, q=q)
            fft_pass(g, 2, inverse=True)
            xq = crop_grid(g, S, P, coils, n)
            del g
            d = xq - xref
            r = rel(float(np.vdot(d, d).real), float(np.vdot(xref, xref).real))
            pred = float(np.sqrt(sum(stages[s][f"bfp{b}"] ** 2 for s in stages)))
            chained[f"bfp{b}"] = {"crop_output_rel": r, "quadrature_of_grid_stages": pred}
            log(kind, "chained", b, chained[f"bfp{b}"])
        res[kind] = {"stage": stages, "chained": chained}
    json.dump(res, open(os.path.join(OUT, "acc_adjoint.json"), "w"), indent=1)


def phase_kspace():
    """k-space data: 256 readouts x 3990 samples of the noise-free sim + complex Gaussian noise sigma per real
    component (meta noise_sigma), coil-compressed 32 -> 10 with the coil-map Gram's basis; BFP blocks along samples
    of one coil (device layout [coil][sample]) or along coils of one sample ([sample][coil])"""
    meta = json.load(open(D + "meta.json"))
    sigma = float(meta["noise_sigma"])
    ka = hs.load_any(D + "data_clean.hv", lazy=True)  # (readouts, samples, 32)
    r0 = ka.shape[0] // 2
    y = np.asarray(ka[r0:r0 + 256]).astype(np.complex128)  # (256, 3990, 32)
    rng = np.random.default_rng(3)
    y += sigma * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
    # coil compression of the data (whitened coils: same unitary as the maps' Gram basis)
    img, S, P, mask, fld, r2 = load_inputs("fixed")
    del img, S, P
    sm = hs.load_any(D + "smaps_true.hv")
    sv = np.transpose(sm, (3, 2, 1, 0)).reshape(32, -1)[:, mask.reshape(-1)]
    Gm = (sv @ sv.conj().T).astype(np.complex128)
    w, U = np.linalg.eigh(Gm)
    U = U[:, ::-1][:, :NVC]
    del sm, sv
    yv = np.einsum("rsc,ck->rsk", y, U.conj())  # noise stays sigma per virtual coil (U orthonormal columns)
    res = {"sigma": sigma, "readouts": 256, "peak_over_sigma": float(np.abs(yv).max() / sigma), "rms_over_sigma": float(np.sqrt(np.mean(np.abs(yv) ** 2)) / sigma)}
    layouts = {"samples_per_coil": np.ascontiguousarray(np.transpose(yv, (2, 0, 1))), "coils_per_sample": np.ascontiguousarray(yv)}
    for lay, a in layouts.items():
        r = {}
        for b in BITS:
            d = q_bfp(a, b) - a
            r[f"bfp{b}"] = {"rms_err_over_sigma": float(np.sqrt(np.mean(np.abs(d) ** 2) / 2) / sigma), "max_err_over_sigma": float(np.abs(d).max() / sigma)}
        r["f32"] = {"rms_err_over_sigma": float(np.sqrt(np.mean(np.abs(q_f32(a) - a) ** 2) / 2) / sigma)}
        res[lay] = r
        log("kspace", lay, {k: f"{v['rms_err_over_sigma']:.2e}" for k, v in r.items()})
    # uniform quantiser tied to sigma (KStore +qREL): step q sigma -> rms q / sqrt(12)
    res["uniform_step_over_sigma"] = {str(qs): qs / np.sqrt(12) for qs in [0.25, 0.1, 0.05]}
    json.dump(res, open(os.path.join(OUT, "acc_kspace.json"), "w"), indent=1)


def phase_e2e():
    """End to end: the Gram application g = A^H A x (pad, forward FFT, adjoint FFT, crop; fully sampled grid, so the
    exact result is sum_la |S_c P_l|^2 x) with the buffers rounded per configuration, in one complex64 buffer (BFP
    values and f32 values are exact in it; every pass computes in f64)."""
    img, S, P, mask, fld, r2 = load_inputs("fixed")
    n = img.shape
    G = grid_shape(n)
    coils = list(range(CB))
    nch = CB * L
    xref = sum(np.abs(S[c] * P[l]) ** 2 for c in coils for l in range(L)) * img
    ref2 = float(np.vdot(xref, xref).real)
    iz, iy, ix = hk_place(n, G)
    zimg = {int(z): a2 for a2, z in enumerate(iz)}
    f32 = q_f32
    res = {}
    configs = []
    for b in [16, 20, 22, 24]:
        qb = (lambda b: lambda a: q_bfp(a, b))(b)
        configs.append((f"proposed_b{b}", {"x": qb, "S": lambda a: q_bfp(a, 20), "P": f32, "xy": f32, "yz": qb, "grid": qb, "scatter": f32, "zy": qb, "yx": f32}))
    for b in [16, 20, 24]:
        qb = (lambda b: lambda a: q_bfp(a, b))(b)
        configs.append((f"all_b{b}", {k: qb for k in ["x", "S", "P", "xy", "yz", "grid", "scatter", "zy", "yx"]}))
    configs.append(("all_f32", {k: f32 for k in ["x", "S", "P", "xy", "yz", "grid", "scatter", "zy", "yx"]}))
    for name, q in configs:
        t = time.time()
        xq, Sq, Pq = q["x"](img), np.stack([q["S"](s_) for s_ in S]), np.stack([q["P"](p_) for p_ in P])
        g = np.zeros(G + (nch,), np.complex64)
        for z in range(G[0]):
            if z in zimg:
                a2 = zimg[z]
                pl = np.zeros(G[1:] + (nch,), np.complex128)
                blk = np.empty((n[1], n[2], nch), np.complex128)
                for ci, c in enumerate(coils):
                    for l in range(L):
                        blk[:, :, ci * L + l] = Sq[c, a2] * Pq[l, a2] * xq[a2]
                pl[np.ix_(iy, ix)] = blk
                g[z] = q["xy"](sf.fft(pl, axis=1, workers=WK))
        for s0 in range(0, G[0], 8):
            g[s0:s0 + 8] = q["yz"](sf.fft(g[s0:s0 + 8].astype(np.complex128), axis=1, workers=WK))
        for s0 in range(0, G[1], 8):
            g[:, s0:s0 + 8] = q["grid"](sf.fft(g[:, s0:s0 + 8].astype(np.complex128), axis=0, workers=WK))
        # adjoint input = the forward grid (r = A x); the scatter's stored output, then z, y, x inverse passes + crop
        for s0 in range(0, G[0], 8):
            g[s0:s0 + 8] = q["scatter"](g[s0:s0 + 8].astype(np.complex128))
        for s0 in range(0, G[1], 8):
            g[:, s0:s0 + 8] = q["zy"](sf.ifft(g[:, s0:s0 + 8].astype(np.complex128), axis=0, workers=WK))
        for s0 in range(0, G[0], 8):
            g[s0:s0 + 8] = q["yx"](sf.ifft(g[s0:s0 + 8].astype(np.complex128), axis=1, workers=WK))
        out = np.zeros(n, np.complex128)
        for a2 in range(n[0]):
            pl = sf.ifft(g[iz[a2]].astype(np.complex128), axis=1, workers=WK)[np.ix_(iy, ix)]
            for ci, c in enumerate(coils):
                for l in range(L):
                    out[a2] += np.conj(Sq[c, a2] * Pq[l, a2]) * pl[:, :, ci * L + l]
        del g
        d = out - xref
        res[name] = rel(float(np.vdot(d, d).real), ref2)
        log("e2e", name, f"{res[name]:.3e}", f"{time.time() - t:.0f} s")
    json.dump(res, open(os.path.join(OUT, "acc_e2e.json"), "w"), indent=1)


if __name__ == "__main__":
    ph = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, f in [("image", phase_image), ("forward", phase_forward), ("adjoint", phase_adjoint), ("kspace", phase_kspace), ("e2e", phase_e2e)]:
        if ph in (name, "all"):
            t = time.time()
            f()
            log(f"phase {name} done in {time.time() - t:.0f} s")
