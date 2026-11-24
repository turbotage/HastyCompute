"""Per-set vs pooled k-space sampling counts of the dev configuration (research/precond_sampling_theory.md).

Dev config: out-half ADC (|k| increasing, first 1991 of 3990 ADC samples), kcrop to the 128x72x60 grid's ball,
all 53 144 readouts, 4 respiratory bins (resp.npy amplitude quantiles), 20 cardiac frames, 7 encodes.
For a few (encode, resp bin) groups: per k-cell (cell = 1/FOV per axis) the pooled count n(k) over the group's 20
cardiac frames and the per-frame counts n_t(k). Reported per |k| shell (fraction of the ball radius).
"""
import os, sys, json
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SCAN = os.path.join(REPO, "data/hasty/phantom/factory_arms/ph0001/sim_flow_10min/scan")
meta = json.load(open(os.path.join(SCAN, "meta.json")))
fov = np.array(meta["fov"]); dims = np.array([128, 72, 60]); dx = fov / dims
k = np.load(os.path.join(REPO, "data/hasty/seq/flow_thorax/k_adc.npy"))  # cycles/m
kn = np.linalg.norm(k, axis=1)
i_max = int(np.argmax(kn))                     # first |k| maximum: end of the out half
step = len(k) / 3990.0                          # base curve points per ADC sample (2 us dwell)
idx = np.round(np.arange(0, i_max / step) * step).astype(int)
kb = k[idx]                                     # out half at the ADC raster
tab = np.load(os.path.join(SCAN, "readouts.npy"))
resp = np.load(os.path.join(SCAN, "resp.npy"))[:, 0]
enc = tab[:, 1].astype(int); frame = tab[:, 3].astype(int); rot = tab[:, 5:14].reshape(-1, 3, 3)
q = np.quantile(resp, [0.25, 0.5, 0.75]); rb = np.searchsorted(q, resp)
T = frame.max() + 1
ncell = int(np.prod(dims))
lines = [f"dev grid {dims.tolist()}, FOV {fov.round(3).tolist()} m; out-half ADC samples per readout {len(kb)}; "
         f"{len(tab)} readouts, {T} cardiac frames, 4 resp bins, 7 encodes"]
shells = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
g = np.stack(np.meshgrid(*[np.fft.fftfreq(n, 1 / n) for n in dims], indexing="ij"), -1).reshape(-1, 3)  # cell index
rad = np.sqrt((((g + 0.0) / dims * 2) ** 2).sum(1))   # fraction of the ball radius (|k dx| / (1/2))
inball = rad <= 1.0
res = {}
for (e, b) in [(0, 0), (0, 2), (3, 1)]:
    sel = np.flatnonzero((enc == e) & (rb == b))
    pooled = np.zeros(ncell, np.int64); perf = np.zeros(T * ncell, np.int32)
    kept_tot = 0
    for i0 in range(0, len(sel), 500):
        s = sel[i0:i0 + 500]
        kk = np.einsum("rab,sb->rsa", rot[s], kb)                         # physical k (cycles/m)
        keep = ((kk * dx) ** 2).sum(-1) <= 0.25                              # kcrop ball
        ci = np.round(kk * fov).astype(int) % dims                           # nearest cell (FFT order)
        lin = (ci[..., 0] * dims[1] + ci[..., 1]) * dims[2] + ci[..., 2]
        fr = np.broadcast_to(frame[s][:, None], lin.shape)
        lin, fr = lin[keep], fr[keep]
        kept_tot += lin.size
        pooled += np.bincount(lin, minlength=ncell)
        perf += np.bincount(fr * ncell + lin, minlength=T * ncell).astype(np.int32)
    perf = perf.reshape(T, ncell)
    lines.append(f"\n(encode {e}, resp bin {b}): {len(sel)} readouts, {kept_tot} kept samples (= {kept_tot / len(sel):.0f} per readout after the crop); "
                 f"per set (cardiac frame) {kept_tot / T:.0f} samples on {inball.sum()} cells of the ball")
    lines.append("shell |k|/k_ball | mean n(k) pooled over 20 frames | cells with n >= T=20 | mean per-frame n_t | cells with n_t = 0 | CV of n_t over frames")
    rows = []
    for a, bb in zip(shells[:-1], shells[1:]):
        m = inball & (rad >= a) & (rad < bb)
        npool = pooled[m]; nt = perf[:, m]
        cv = np.where(npool > 0, nt.std(0) / np.maximum(nt.mean(0), 1e-9), np.nan)
        row = dict(shell=[a, bb], n_mean=float(npool.mean()), frac_ge_T=float((npool >= T).mean()),
                   nt_mean=float(nt.mean()), frac_nt0=float((nt == 0).mean()), cv=float(np.nanmedian(cv)))
        rows.append(row)
        lines.append(f"{a:.2f}-{bb:.2f} | {row['n_mean']:9.1f} | {row['frac_ge_T']:.2f} | {row['nt_mean']:7.2f} | {row['frac_nt0']:.2f} | {row['cv']:.2f}")
    res[f"e{e}_b{b}"] = rows
    print("\n".join(lines[-12:])); sys.stdout.flush()
here = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(here, "sampling_stats.txt"), "w").write("\n".join(lines) + "\n")
json.dump(res, open(os.path.join(here, "sampling_stats.json"), "w"), indent=1)
