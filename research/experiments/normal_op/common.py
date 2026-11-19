"""Shared loaders for the structured-normal-operator study (research/structured_normal_operator.md).

CPU only. Data: data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan (2 mm yarn 4D-flow sim, 7 encodes,
53144 readouts x 3990 ADC samples). Readout i = rotation i // 7, encode i % 7.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python", "lib"))
from hastycompute.store import load_any  # noqa: E402

SCAN = os.path.join(ROOT, "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan")
N_ENC = 7


def readout_table():
    ro = np.load(os.path.join(SCAN, "readouts.npy"))      # t_exc, enc, part, frame, cardiac phase, rot(9)
    ph = np.load(os.path.join(SCAN, "physio.npy"))        # col 3 = bellows
    return ro, ph


def encode_samples(enc=0, ro_stride=1, adc_stride=1, cols=None, sched=False):
    """samples of one encode: array [n_ro, n_adc, c] (samples.hv: k (rad/m, physical) + tau; sched_nominal.hv: the
    16-float records k, m1, m2h, w[z2, x2+y2, xz, yz] (rad/m^2), tau, delta, phi0), plus the readout indices."""
    a = load_any(os.path.join(SCAN, "sched_nominal.hv" if sched else "samples.hv"), lazy=True)
    idx = np.arange(enc, a.shape[0], N_ENC * ro_stride)
    out = []
    step = 512
    for s in range(0, len(idx), step):
        blk = idx[s:s + step]
        x = a[int(blk[0]):int(blk[-1]) + 1:N_ENC * ro_stride, ::adc_stride]
        if cols is not None:
            x = x[..., cols]
        out.append(np.ascontiguousarray(x, dtype=np.float32))
    return np.concatenate(out, 0), idx


def cardiac_cubic_weights(c, M=16):
    """periodic cubic B-spline weights: knots b at c = b / M. returns (idx [n,4], w [n,4])"""
    x = (np.asarray(c) % 1.0) * M
    i = np.floor(x).astype(np.int64)
    u = x - i
    w = np.stack([(1 - u) ** 3 / 6, (3 * u ** 3 - 6 * u ** 2 + 4) / 6, (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6,
                  u ** 3 / 6], 1)
    idx = (i[:, None] + np.arange(-1, 3)[None, :]) % M
    return idx, w


def resp_linear_weights(b, K=6, knots=None):
    """linear hats at the amplitude quantiles 0, 1/(K-1), ..., 1 (clamped). returns (idx [n,2], w [n,2], knots)"""
    b = np.asarray(b, dtype=np.float64)
    if knots is None:
        knots = np.quantile(b, np.linspace(0, 1, K))
    x = np.clip(b, knots[0], knots[-1])
    i = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, K - 2)
    f = (x - knots[i]) / (knots[i + 1] - knots[i])
    return np.stack([i, i + 1], 1), np.stack([1 - f, f], 1), knots


def knot_weights_per_readout(ro_idx, K_resp=6, M_card=16):
    """2-D knot weights (resp linear x cardiac cubic) per readout: idx [n,8] (k = r*M + c), w [n,8]"""
    ro, ph = readout_table()
    ci, cw = cardiac_cubic_weights(ro[ro_idx, 4], M_card)
    ri, rw, _ = resp_linear_weights(ph[:, 3], K_resp)
    ri, rw = ri[ro_idx], rw[ro_idx]
    idx = (ri[:, :, None] * M_card + ci[:, None, :]).reshape(-1, 8)
    w = (rw[:, :, None] * cw[:, None, :]).reshape(-1, 8)
    return idx, w
