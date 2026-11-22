"""Summarise fieldabsorb JSON outputs: L per variant (max / p90 / median over knots, per encode), fitted coefficients,
shifts, knot-to-knot variation, residual field ranges. Usage: summarize.py <run.json> [...]"""
import json
import sys

import numpy as np


def pct(a, q):
    return float(np.percentile(np.asarray(a, float), q))


def summarize(path):
    d = json.load(open(path))
    knots = d["knots"]
    encs = d["encodes"]
    print(f"\n=== {path}: grid {d['grid']}, knots2d {d['knots2d']}, {len(knots)} knots, encodes {encs}, "
          f"rule voxels {d['seg_fit_voxels']} (stride {d['seg_fit_stride']}), vcoils {d['vcoils']}, tol {d['tol_noise']:.3e}")
    print(f"today's rule (static field, phase only): L per encode {d['static_rule_L']} -> engine L = {max(d['static_rule_L'])}")
    variants = []
    for k in knots:
        for e in k["encodes"]:
            for v in e["res"]:
                if v not in variants:
                    variants.append(v)
    print("variant  | L noise rule: max / p90 / median (over knots x encodes) | L 1e-3 max/med | L 1e-4 max/med | per-encode max (noise)")
    table = {}
    for v in variants:
        rows = [(e["enc"], e["res"][v]["L"]) for k in knots for e in k["encodes"] if v in e["res"]]
        Ln = [r[1][0] for r in rows]
        L3 = [r[1][1] for r in rows]
        L4 = [r[1][2] for r in rows]
        per_enc = {en: max(r[1][0] for r in rows if r[0] == en) for en in encs if any(r[0] == en for r in rows)}
        table[v] = dict(n=len(rows), max=max(Ln), p90=pct(Ln, 90), med=pct(Ln, 50), max3=max(L3), med3=pct(L3, 50),
                        max4=max(L4), med4=pct(L4, 50), per_enc=per_enc)
        print(f"{v:9s}| {max(Ln):3d} / {pct(Ln, 90):4.1f} / {pct(Ln, 50):4.1f}  (n={len(rows)})"
              f" | {max(L3):3d} / {pct(L3, 50):4.1f} | {max(L4):3d} / {pct(L4, 50):4.1f} | {per_enc}")
    # paired change per (knot, encode) relative to a
    for v in variants:
        if v == "a" or v.endswith("_nofat"):
            continue
        dl = [e["res"][v]["L"][0] - e["res"]["a"]["L"][0] for k in knots for e in k["encodes"] if v in e["res"] and "a" in e["res"]]
        if dl:
            print(f"  {v} - a (noise rule, paired): mean {np.mean(dl):+.2f}, min {min(dl):+d}, max {max(dl):+d}")
    for v in [x for x in variants if x.endswith("_nofat")]:
        base = v.split("_")[0]
        dl = [e["res"][v]["L"][0] - e["res"][base]["L"][0] for k in knots for e in k["encodes"] if v in e["res"]]
        print(f"  {v} - {base} (water only vs with fat, paired): mean {np.mean(dl):+.2f}, min {min(dl):+d}, max {max(dl):+d}")
    # tail-optimal linear term
    opts = [e["opt"] for k in knots for e in k["encodes"] if e.get("opt")]
    if opts:
        r_lsq = [o["tail_lsq"] / o["tail_a"] for o in opts]
        r_opt = [o["tail_opt"] / o["tail_a"] for o in opts]
        xs = np.array([o["x_hz_fov"] for o in opts])
        print(f"  tail-optimal linear term ({len(opts)} knot-encodes): weighted tail at L_a-1 vs none: LSQ x{np.median(r_lsq):.3f} "
              f"(range {min(r_lsq):.3f}..{max(r_lsq):.3f}), optimum x{np.median(r_opt):.3f} (range {min(r_opt):.3f}..{max(r_opt):.3f}); "
              f"optimal term |Hz across FOV| median {np.median(np.abs(xs), axis=0).round(1).tolist()}")
    # coefficients
    p1 = np.array([k["p1_hz"] for k in knots])
    fov = np.array([k["hz_across_fov"] for k in knots])
    hpv = np.array([k["hz_per_voxel"] for k in knots])
    sh = np.array([k["shift_cells_exc"] for k in knots])
    shr = np.array([k["shift_cells_ref"] for k in knots])
    print(f"per-knot LSQ fit: constant {p1[:, 0].min():.1f}..{p1[:, 0].max():.1f} Hz (median {np.median(p1[:, 0]):.1f}); "
          f"linear Hz across the FOV x {fov[:, 0].min():.0f}..{fov[:, 0].max():.0f}, y {fov[:, 1].min():.0f}..{fov[:, 1].max():.0f}, "
          f"z {fov[:, 2].min():.0f}..{fov[:, 2].max():.0f}; Hz per voxel max |.| {np.abs(hpv).max(axis=0).round(3).tolist()}")
    print(f"max shift (cells of the sigma=1.25 grid) per axis, tau from excitation: {sh.max(axis=0).round(2).tolist()}, "
          f"from the readout centre: {shr.max(axis=0).round(2).tolist()}; global fit {np.round(d['global_shift_cells'], 2).tolist()}")
    # knot-to-knot variation: breathing (r index) vs cardiac (c index)
    ri = np.array([k["r_index"] for k in knots])
    ci = np.array([k["c_index"] for k in knots])
    for name, col in [("constant Hz", p1[:, 0]), ("x Hz/FOV", fov[:, 0]), ("y Hz/FOV", fov[:, 1]), ("z Hz/FOV", fov[:, 2])]:
        by_r = [col[ri == r].mean() for r in sorted(set(ri))]
        by_c = [col[ci == c].mean() for c in sorted(set(ci))]
        within_c = np.mean([col[ri == r].std() for r in sorted(set(ri))])
        within_r = np.mean([col[ci == c].std() for c in sorted(set(ci)) if (ci == c).sum() > 1]) if len(set(ci)) < len(ci) else float("nan")
        print(f"  {name:12s}: mean per breathing knot {np.round(by_r, 1).tolist()}; spread over cardiac knots (std within a breathing knot) {within_c:.1f}; "
              f"over breathing knots (std within a cardiac knot) {within_r:.1f}")
    fs = {key: np.array([k["field_stats_hz"][key] for k in knots]) for key in knots[0]["field_stats_hz"]}
    for key, a in fs.items():
        print(f"  residual field {key}: weighted 1-99 % span median {np.median(a[:, 1] - a[:, 0]):.0f} Hz, 0.1-99.9 % span median "
              f"{np.median(a[:, 3] - a[:, 2]):.0f} Hz (max {np.max(a[:, 3] - a[:, 2]):.0f}), weighted std median {np.median(a[:, 4]):.0f} Hz, full range median {np.median(a[:, 6] - a[:, 5]):.0f} Hz")
    tay = np.array([k["quad_taylor_rad"] for k in knots])
    tay9 = np.array([k["quad_taylor_999_rad"] for k in knots])
    print(f"  quadratic (d) Taylor |Q| |tau - tau_ref|: max {tay.max():.2f} rad, median {np.median(tay):.2f}; signal-weighted 99.9 % median {np.median(tay9):.2f} rad")
    if d.get("control"):
        for r in d["control"]["rows"]:
            print(f"  control enc {r['enc']}: zero {r['zero']}, synthetic none {r['syn_none']}, synthetic absorbed {r['syn_abs']}, tail rel diff {r['tail_rel_diff']:.1e}")
    for c in d.get("conc", []):
        print(f"  conc0 enc {c['enc']}: constant <= {c['c0_max_rad']:.3f} rad, linear shift <= {np.round(c['lin_shift_cells'], 3).tolist()} cells, "
              f"span {c['span_rad']:.3f} -> {c['span_after_lin_rad']:.3f} (no const+lin) -> {c['span_after_quad_rad']:.4f} rad (no quad), quad Taylor {c['quad_taylor_rad']:.3f} rad")
    return table


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
