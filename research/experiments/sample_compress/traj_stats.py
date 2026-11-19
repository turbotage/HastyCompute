"""Direction 1/2/3 groundwork: how oversampled is each yarn readout for a 160 mm box, and where does the k-space
density come from (one readout dwelling vs many readouts crossing)?

Signal of a box-supported image along one readout (after demodulating the box centre c):
    y(t) = sum_r x(r) exp(-i [k(t).(r - c) + w(r) tau(t) + phi_conc(t, r)]),  r - c in [-h, h]^3
Instantaneous frequency of source r: -(kdot.(r-c) + w + dphi_conc/dt) / 2pi.  Over the box the band is
    B(t) = sum_a |kdot_a| h_a / pi + B_w + B_conc        (Hz, two-sided width)
Landau: the readout's signal space has dimension ~ int B(t) dt (+ O(log) edge terms) = the readout's L1 path length
in box-Nyquist cells (1/FOV_box). The ADC takes n_adc = T/dwell samples, so the intra-readout redundancy is
n_adc / int B dt.

Proxies (as normal_op/knot_gram_rank.py): 'box2mm' = native k (2 mm yarn), 'box1mm' = k x 2 (1 mm yarn with the same
samples per readout). Encode 0, all readouts (or a stride).

run: rustlib/.venv/bin/python -I research/experiments/sample_compress/traj_stats.py [ro_stride]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "normal_op"))
import common  # noqa: E402

OUT = os.path.join(HERE, "out")


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    t0 = time.time()
    meta = json.load(open(os.path.join(common.SCAN, "meta.json")))
    s, ro_idx = common.encode_samples(0, ro_stride=stride)          # [nro, na, 4]: k (rad/m), tau
    nro, na, _ = s.shape
    k = s[..., :3].astype(np.float64)
    tau = s[..., 3].astype(np.float64)
    dt = np.diff(tau, axis=1)
    print(f"{nro} readouts x {na} samples, dwell {np.median(dt)*1e6:.3f} us, tau {tau.min()*1e3:.3f}..{tau.max()*1e3:.3f} ms ({time.time()-t0:.0f}s)", flush=True)
    kdot = np.diff(k, axis=1) / dt[..., None]                     # rad/m/s, between samples
    kdot = np.concatenate([kdot[:, :1], 0.5 * (kdot[:, 1:] + kdot[:, :-1]), kdot[:, -1:]], 1)
    dwell = float(np.median(dt))
    kmag = np.linalg.norm(k, axis=2)
    res = {"n_readouts_used": nro, "n_adc": na, "dwell_s": dwell, "readout_s": float(np.median(tau[:, -1] - tau[:, 0])),
           "kmax_radm": float(kmag.max()), "k_end_over_kmax_median": float(np.median(kmag[:, -1]) / kmag.max()),
           "k_start_over_kmax_median": float(np.median(kmag[:, 0]) / kmag.max())}
    # speed profile along the readout (median over readouts), in box-1mm Nyquist cells per sample
    fov_full = np.array(meta["fov"])
    cases = {"full_fov_2mm": (1.0, fov_full / 2), "box160_2mm": (1.0, np.full(3, 0.08)), "box160_1mm": (2.0, np.full(3, 0.08)),
             "box128_1mm": (2.0, np.full(3, 0.064))}
    # off-resonance + fat band inside the box: field_ppm range (whole FOV: conservative) + fat -3.4 ppm
    ppm = np.load(os.path.join(common.SCAN, "field_ppm_recon.npy"))
    ppm = ppm[np.isfinite(ppm) & (ppm != 0)]
    f0 = 42.577e6 * meta["b0"] * 1e-6                             # Hz per ppm
    bw_hz = (np.percentile(ppm, 99.9) - np.percentile(ppm, 0.1) + 3.8) * f0
    res["offres_band_hz"] = float(bw_hz)
    res["cases"] = {}
    for name, (scale, h) in cases.items():
        B = scale * (np.abs(kdot) * h[None, None, :]).sum(2) / np.pi + bw_hz     # Hz
        nl = (B * dwell).sum(1)                                   # Landau count per readout
        per = B * dwell                                           # dof per sample (<1 = oversampled)
        res["cases"][name] = {
            "landau_per_readout_median": float(np.median(nl)), "landau_per_readout_p95": float(np.percentile(nl, 95)),
            "landau_per_readout_max": float(nl.max()),
            "redundancy_median": float(na / np.median(nl)), "redundancy_total": float(nro * na / nl.sum()),
            "dof_per_sample_max": float(per.max()),
            "frac_samples_oversampled_2x": float((per < 0.5).mean()),
            "dof_per_sample_profile_median": [float(v) for v in np.median(per, 0)[::na // 20]],
        }
        print(name, json.dumps(res["cases"][name])[:400], flush=True)

    # where does the density come from? 2N grid of the box1mm proxy (cell = 1/(2 FOV_box)), shells as knot_gram
    for name, scale in (("box160_1mm", 2.0), ("box160_2mm", 1.0)):
        fov = 0.16
        N = int(round(fov / (0.001 if scale == 2.0 else 0.002)))
        u = k.reshape(-1, 3) / (2 * np.pi) * scale * 2 * fov       # 2N cells
        cell = np.floor(u + N).astype(np.int64)
        n2 = 2 * N
        cid = (cell[:, 0] * n2 + cell[:, 1]) * n2 + cell[:, 2]
        rr = np.linalg.norm(u, axis=1) / N
        ro_of = np.repeat(np.arange(nro), na)
        shells = [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]
        out = []
        for a, b in shells:
            m = (rr >= a) & (rr < b)
            c = cid[m]
            pair = c * nro + ro_of[m]
            ns = m.sum()
            ncell = len(np.unique(c))
            npair = len(np.unique(pair))
            out.append({"shell": f"{a}-{b}", "samples_frac": float(ns / len(rr)), "samples_per_occupied_cell": float(ns / ncell),
                        "readouts_per_occupied_cell": float(npair / ncell), "samples_per_readout_visit": float(ns / npair)})
        res["density_" + name] = out
        print(name, json.dumps(out), flush=True)
        del u, cell, cid, rr
    res["seconds"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, f"traj_stats_s{stride}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
