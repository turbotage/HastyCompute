"""Q2: which per-sample quantities are functions of the k-space position (=> 'kernel-free' pointwise multipliers on
the Toeplitz grid) and which are not (=> extra moment kernels per knot pair).

1. readout time tau vs |k|: the yarn readout goes out and comes back, so tau is two-valued in |k|; per half it is a
   function of |k| (the base curve is rotated rigidly). Measures the spread of tau at fixed |k| per half.
2. per-sample concomitant residual dw_j = w_j - wbar(tau_j) (4 coefficients [z2, x2+y2, xz, yz], rad/m^2; wbar = mean
   over the encode's readouts at the same ADC index): its phase at the FOV corners, and the fraction of its variance
   that a function of the k-space cell explains (cells of the 2N grid of the 160 mm box at 2 mm: ~14 samples/cell).

run: systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 rustlib/.venv/bin/python -I \
     research/experiments/normal_op/tau_conc_kspace.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def main():
    rec, ro_idx = common.encode_samples(0, ro_stride=2, sched=True, cols=[0, 1, 2, 9, 10, 11, 12, 13])
    nro, na, _ = rec.shape
    k = rec[..., :3].astype(np.float64)
    w = rec[..., 3:7].astype(np.float64)
    tau = rec[0, :, 7].astype(np.float64)
    kr = np.linalg.norm(k, axis=-1)
    out = {"readouts": int(nro), "adc": int(na)}
    # 1. tau vs |k|
    kmean = kr.mean(0)
    iturn = int(np.argmax(kmean))
    out["turn_tau_ms"] = float(tau[iturn] * 1e3)
    out["kr_spread_over_readouts_rel_max"] = float((kr.std(0) / kmean.max()).max())
    mono_out = bool(np.all(np.diff(kmean[:iturn + 1]) > 0))
    mono_back = bool(np.all(np.diff(kmean[iturn:]) < 0))
    out["monotone_out_back"] = [mono_out, mono_back]
    # time separation of the two visits of the same |k|
    kb = kmean[iturn:][::-1]
    tb = tau[iturn:][::-1]
    t_back_at = np.interp(kmean[:iturn + 1], kb, tb)
    out["max_gap_out_back_ms"] = float(np.max(t_back_at - tau[:iturn + 1]) * 1e3)
    # 2. concomitant residual
    wbar = w.mean(0, keepdims=True)
    dw = w - wbar
    fov = np.array(json.load(open(os.path.join(common.SCAN, "meta.json")))["fov"])
    od = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    heart = np.array(json.load(open(os.path.join(od, "field_rank.json")))["heart_centre_mm"]) * 1e-3
    # boxes in scanner coordinates (FOV centre = isocentre here): torso FOV, 160 mm cube around the heart
    boxes = {"torso": (np.zeros(3), fov / 2), "box160_heart": (heart, np.full(3, 0.08))}
    out["conc"] = {}
    for name, (c, h) in boxes.items():
        corners = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        mono = np.stack([corners[:, 2] ** 2, corners[:, 0] ** 2 + corners[:, 1] ** 2, corners[:, 0] * corners[:, 2],
                         corners[:, 1] * corners[:, 2]], 1)        # [8 corners, 4]; |phase| is maximal at a corner
        ph_res = np.abs(dw.reshape(-1, 4) @ mono.T).max(1)
        ph_tot = np.abs(w.reshape(-1, 4) @ mono.T).max(1)
        out["conc"][name] = {"max_phase_total_rad": float(ph_tot.max()), "max_phase_residual_rad": float(ph_res.max()),
                             "p99_phase_residual_rad": float(np.percentile(ph_res, 99)),
                             "rms_phase_residual_rad": float(np.sqrt(np.mean(ph_res ** 2)))}
    # function of the k-cell? (160 mm box, 2 mm: cell = 1/(0.32 m))
    u = np.rint(k.reshape(-1, 3) / (2 * np.pi) * 0.32).astype(np.int64) + 80
    half_id = (np.arange(na) > iturn).astype(np.int64)
    cid = ((u[:, 0] * 160 + u[:, 1]) * 160 + u[:, 2]) * 2 + np.tile(half_id, nro)
    d = dw.reshape(-1, 4)
    expl = []
    for q in range(4):
        s1 = np.bincount(cid, d[:, q], minlength=160 ** 3 * 2)
        s0 = np.bincount(cid, minlength=160 ** 3 * 2)
        mean_c = s1[cid] / s0[cid]
        expl.append(float(1 - np.var(d[:, q] - mean_c) / np.var(d[:, q])))
    s0 = np.bincount(cid)
    out["dw_var_explained_by_kcell_half"] = expl
    out["samples_per_occupied_cell"] = float(s0[s0 > 0].mean())
    # baseline: explained fraction a random relabelling of cells would give (finite-sample bias of cell means)
    rng = np.random.default_rng(0)
    cr = rng.permutation(cid)
    s1 = np.bincount(cr, d[:, 0], minlength=160 ** 3 * 2)
    s0r = np.bincount(cr, minlength=160 ** 3 * 2)
    out["dw_var_explained_random_cells"] = float(1 - np.var(d[:, 0] - s1[cr] / s0r[cr]) / np.var(d[:, 0]))
    json.dump(out, open(os.path.join(od, "tau_conc_kspace.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
