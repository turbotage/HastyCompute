"""Shared definitions of the temporal-rank study (research/temporal_rank.md).

Grid: the 1 mm fine truth simblood_lib/truth4d.hv of ph0001 (100 cardiac frames, end-expiration r = 0, LBM chambers +
LBM coronaries + 1-D network vessels). The study works on 2 mm voxels (the reconstruction voxel), each the complex mean
of its 2 x 2 x 2 fine voxels, so partial volume and intravoxel dephasing are in the signal.

Encodes: the scan's coprime7 dual-VENC scheme (meta.json sequence.encoding.at_te): encode 0 = (-a, -a, -a), encodes
1/3/5 flip axis x/y/z, encodes 2/4/6 put a third of the moment on that axis; a = gamma * m1 = pi / 2 rad per m/s, so the
(0, odd) pairs have VENC_lo = 1.0 m/s and the (0, even) pairs VENC_hi = 1.5 m/s.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
PH = REPO / "data/hasty/phantom/factory_arms/ph0001"
SCAN = PH / "sim_flow_10min/scan"
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CACHE = HERE / "cache"  # gitignored .npy (fine-field cache, ~4 GB; delete when done)
OUT.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

GAMMA = 2.6752218744e8
# fine-grid boxes (index z, y, x on the 320 x 220 x 290 simblood grid)
CBOX = ((60, 256), (24, 174), (82, 226))  # cached fine fields
ABOX = ((76, 240), (34, 164), (88, 220))  # analysis region (inside CBOX by the respiratory margins 16 / 10 / 6 mm)
NF = 100
TE, TR, FLIP = 1.6e-3, 11.29e-3, np.deg2rad(12.0)
INFLOW_GAIN = 1.5  # sim_input blood_inflow_gain
BLOOD_LABELS = ["blood_LA", "blood_LV", "blood_RA", "blood_RV", "blood_artery", "blood_vein", "blood_pulm_artery", "blood_pulm_vein"]
CHAMBERS = ["blood_LA", "blood_LV", "blood_RA", "blood_RV"]
VESSEL_BLOOD = ["blood_artery", "blood_vein", "blood_pulm_artery", "blood_pulm_vein"]


def meta():
    return json.loads((SCAN / "meta.json").read_text())


def encode_matrix():
    """K [7, 3]: phase of encode e = -K[e] . v (rad, v in m/s) -- gamma * m1 of meta.json (sign as stored)"""
    at = meta()["sequence"]["encoding"]["at_te"]
    return np.array([[GAMMA * m for m in e["m1"]] for e in at])


def venc_pairs():
    return 1.0, 1.5


def decode(x, K=None):
    """dual-VENC velocity from 7 complex encodes x [..., 7] -> v [..., 3] m/s: per axis the VENC_lo pair (0, 2a+1)
    (period 2 m/s) and the VENC_hi pair (0, 2a+2) (period 3 m/s); the coprime periods give an unaliased range of
    +-3 m/s (meta "unaliased" 3.0): among the lo candidates v_lo + 2k in [-3, 3] the one nearest a hi candidate
    v_hi + 3j is kept (the lo value, the better-conditioned pair)"""
    K = encode_matrix() if K is None else K
    v = np.empty(x.shape[:-1] + (3,), np.float32)
    for a in range(3):
        lo, hi = 2 * a + 1, 2 * a + 2
        # phase(e) = -K[e] . v ; phase(lo) - phase(0) = -(K[lo, a] - K[0, a]) v_a
        dlo = -(K[lo, a] - K[0, a])
        dhi = -(K[hi, a] - K[0, a])
        vlo = (np.angle(x[..., lo] * np.conj(x[..., 0])) / dlo).astype(np.float32)
        vhi = (np.angle(x[..., hi] * np.conj(x[..., 0])) / dhi).astype(np.float32)
        plo, phi = 2 * np.pi / abs(dlo), 2 * np.pi / abs(dhi)
        best = np.full(vlo.shape, np.inf, np.float32)
        out = vlo.copy()
        for k in (-1, 0, 1):
            c = vlo + k * plo
            d = np.min(np.stack([np.abs(c - (vhi + j * phi)) for j in (-1, 0, 1)]), 0)
            take = d < best
            best = np.where(take, d, best)
            out = np.where(take, c, out)
        v[..., a] = out
    return v


def ss(t1):
    e1 = np.exp(-TR / np.maximum(t1, 1e-3))
    return np.sin(FLIP) * (1 - e1) / (1 - np.cos(FLIP) * e1)


def label_tables(tissues):
    """per label: (water signal, fat signal) of the spoiled GRE at TE (no texture) and the T2*/T1 for the texture"""
    tp = json.loads((PH / "tissue_params.json").read_text())["table"]
    rows = [tp.get(n) for n in tissues]
    return rows


def magnitude(lab, ta, tb, blood_u8, tissues, rows):
    """fine-voxel magnitude of the spoiled GRE at TE: label row with texture (tissue_variation.apply), blood fraction
    blended with arterial blood (x inflow gain)"""
    lab = lab.astype(np.int64)
    pd0 = np.array([r["pd"] if r else 0.0 for r in rows])
    ff0 = np.array([r.get("ff", 0.0) if r else 0.0 for r in rows])
    t10 = np.array([r["t1_s"] if r else 1.0 for r in rows])
    t2s0 = np.array([r["t2s_s"] if r else 1.0 for r in rows])
    isb = np.array([n in BLOOD_LABELS for n in tissues])
    tp = json.loads((PH / "tissue_params.json").read_text())["table"]
    t1f, t2f = tp["fat"]["t1_s"], tp["fat"]["t2s_s"]
    a = np.where(isb[lab], 0.0, ta.astype(np.float32))
    b = np.where(isb[lab], 0.0, tb.astype(np.float32))
    pd = pd0[lab] * np.clip(1 + a, 0.2, 2.0)
    t2s = t2s0[lab] * np.clip(1 + b, 0.2, 2.0)
    t1 = t10[lab] * np.clip(1 + b / 3, 0.5, 1.5)
    ff = ff0[lab]
    s = pd * ((1 - ff) * ss(t1) * np.exp(-TE / t2s) + ff * ss(t1f) * np.exp(-TE / t2f))
    s = np.where(isb[lab], s * INFLOW_GAIN, s)
    # partial blood in a non-blood voxel: blend with arterial blood
    rb = tp["blood_artery"]
    sb = INFLOW_GAIN * rb["pd"] * ss(rb["t1_s"]) * np.exp(-TE / rb["t2s_s"])
    beta = blood_u8.astype(np.float32) / 255.0
    s = np.where(isb[lab], s, beta * sb + (1 - beta) * s)
    return s.astype(np.float32)


_BLOCKS = {}


def read_frame(lt, f):
    """frame f (leading index) of an unchunked LazyTensor, decoding only the 1 MiB blocks that cover it (the module's
    Python fallback decodes the whole tensor, 12 GB for vel)"""
    import struct
    from hastycompute import value_format as vf
    key = (id(lt.mm), lt.offset)
    if key not in _BLOCKS:
        tab, p, off = [], lt.offset, 0
        raw_len = int(np.prod(lt.shape)) * lt.dtype.itemsize
        while off < raw_len:
            h = struct.unpack_from("<BBBBII", lt.mm, p)
            hl = 12 + (4 if h[3] & vf.FLAG_CRC else 0)
            tab.append((off, p + hl, h))
            p += hl + h[5]
            off += h[4]
        _BLOCKS[key] = tab
    tab = _BLOCKS[key]
    per = int(np.prod(lt.shape[1:])) * lt.dtype.itemsize
    a, b = f * per, (f + 1) * per
    bs = lt.block_size
    i0, i1 = a // bs, (b - 1) // bs
    buf = bytearray((i1 - i0 + 1) * bs)
    mv = memoryview(buf)
    for i in range(i0, i1 + 1):
        off, pp, h = tab[i]
        vf._decode_block(h, bytes(lt.mm[pp:pp + h[5]]), mv[(i - i0) * bs:(i - i0) * bs + h[4]])
    s = a - i0 * bs
    return np.frombuffer(bytes(buf[s:s + per]), lt.dtype).reshape(lt.shape[1:])


# ---------------------------------------------------------------- temporal bases
def cardiac_bspline_weight(t, b, m, deg=3):
    """recon's periodic B-spline weight (hasty_recon::vknot::cardiac_weight), t and knots in knot units"""
    d = np.mod(t - b, m)
    d = np.where(d > 0.5 * m, d - m, d)
    a = np.abs(d)
    if deg == 1:
        return np.maximum(1 - a, 0)
    if deg == 3:
        return np.where(a <= 1, 2 / 3 - a * a + 0.5 * a ** 3, np.where(a <= 2, (2 - a) ** 3 / 6, 0.0))
    raise ValueError(deg)


def cardiac_basis(phase, m, deg=3):
    """[T, m]: periodic B-splines with knot b at phase (b + 1/2)/m (the operator frames' convention; the span is the same
    as the cubic interpolating spline through m frames)"""
    t = np.mod(phase * m - 0.5, m)
    return np.stack([cardiac_bspline_weight(t, b, m, deg) for b in range(m)], 1)


def fourier_basis(phase, k):
    """[T, k]: real Fourier series of k terms (1, cos, sin, ...)"""
    cols = [np.ones_like(phase)]
    h = 1
    while len(cols) < k:
        cols.append(np.cos(2 * np.pi * h * phase))
        if len(cols) < k:
            cols.append(np.sin(2 * np.pi * h * phase))
        h += 1
    return np.stack(cols, 1)


def resp_hats(r, knots):
    """[T, len(knots)] linear hats at the knot amplitudes (clamped at the ends), as KnotBasis rdeg 1"""
    q = np.asarray(knots)
    W = np.zeros((len(r), len(q)))
    for j, a in enumerate(r):
        i = max(0, min(len(q) - 2, int(np.searchsorted(q, a, side="right") - 1)))
        v = np.clip((a - q[i]) / max(q[i + 1] - q[i], 1e-12), 0, 1)
        W[j, i] += 1 - v
        W[j, i + 1] += v
    return W


def orth(B):
    """orthonormal columns spanning B (QR)"""
    q, _ = np.linalg.qr(B)
    return q


def gram_eig(G):
    """eigen-decomposition of a Hermitian Gram, descending"""
    w, U = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    return np.maximum(w[o], 0), U[:, o]


def k_at(w, levels):
    """smallest K with relative Frobenius truncation error sqrt(sum_{k>=K} w / sum w) <= level"""
    tail = np.sqrt(np.maximum(np.cumsum(w[::-1])[::-1], 0) / w.sum())  # tail[K] = error with K kept
    tail = np.append(tail, 0.0)
    return [int(np.argmax(tail <= l)) for l in levels]


LEVELS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]


def synthetic_rank_control(rng=None, T=100, V=5000, rank=7, noise=0.0):
    """sanity control: a V x T complex signal of exact rank `rank` (random temporal + spatial factors with a decaying
    spectrum) must come out of the Gram pipeline with that rank"""
    rng = np.random.default_rng(0) if rng is None else rng
    A = rng.standard_normal((V, rank)) + 1j * rng.standard_normal((V, rank))
    B = rng.standard_normal((rank, T)) + 1j * rng.standard_normal((rank, T))
    s = 2.0 ** -np.arange(rank)
    X = (A * s) @ B
    if noise:
        X = X + noise * (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))
    return X
