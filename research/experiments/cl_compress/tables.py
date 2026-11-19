"""Markdown tables from out/<region>_<phys>.json (cl_rank.py)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tab(fn):
    o = json.load(open(HERE / "out" / fn))
    C = o["C"]
    print(f"### {o['region']} / {o['phys']}: {o['nv']} voxels, K {o['K']}, C {C}, mean w_m {o['mean_m']:.0f}, timing {o.get('timing')}")
    for key in ("eu", "ed", "em", "uu", "ud", "um"):
        r = o["res"][key]
        eps = [e for e in r["d"]]
        print(f"\n{key}: (voxel weights {'PD energy' if key[0]=='e' else 'uniform on support'}, time weights {dict(u='uniform', d='data energy', m='noise-referenced')[key[1]]})\n")
        print("| eps | fixed (today) C*L | fixed_e C*L | seg_svd C*L | Tucker Rc x Rt | (b') coil-adaptive | (c) 8 blocks | (c) 64 blocks | (a) L=20 | **(d) joint R** | e-basis on uniform w | today/d | seg_svd/d |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for e in eps:
            f = lambda v, m=C: (v * m if v else None)
            d = r["d"][e]
            fx = f(r["fixed_L"][e]); fe = f(r["fixed_e_L"][e]); ss = f(r["seg_svd_L"][e])
            tk = r["tucker"][e]
            tks = f"{tk[0]} ({tk[1]}x{tk[2]})" if tk else None
            cross = r.get("d_cross_u", {}).get(e)
            lab = e if float(e) not in (r.get("eps_noise") or -1,) else f"{float(e):.1e} (R_pred 0.2/1.5)"
            ratio = lambda a: f"{a/d:.2f}" if (a and d) else "-"
            print(f"| {lab} | {fx} | {fe} | {ss} | {tks} | {r['block_coil'][e]} | {r.get('blk8',{}).get(e)} | {r.get('blk64',{}).get(e)} | {r['a_R']['20'][e]} | **{d}** | {cross} | {ratio(fx)} | {ratio(ss)} |")
        if "holdout_d" in r:
            print("\nholdout (voxel parity):", {k: f"R {v[0]}: {v[1]:.2e}" for k, v in r["holdout_d"].items()})
        print("a check (err at R=CL vs fixed):", {k: [f"{x:.2e}" for x in v] for k, v in r.get("a_check_full_vs_fixed", {}).items()})
        print("fixed err per L:", {L: f"{v:.1e}" for L, v in r["fixed_err"].items() if int(L) % 2 == 0})
        print("fixed_e err per L:", {L: f"{v:.1e}" for L, v in r["fixed_e_err"].items() if int(L) % 2 == 0})
        if "block_coil_alloc" in r:
            print("coil-adaptive L_s at 1e-3:", r["block_coil_alloc"].get("0.001"))
    pv = o.get("pervoxel", {})
    for key, d in pv.items():
        print(f"\nper-voxel rho ({key} basis, uniform time):")
        for lab, v in d.items():
            print(f"  {lab}: pct {{{', '.join(f'{k}: {x:.1e}' for k, x in v['pct'].items())}}}  PD-energy frac with rho>1e-4/1e-3/1e-2: {', '.join(f'{x:.1e}' for x in v['energy_frac_above_eps'].values())}")


for fn in sys.argv[1:]:
    tab(fn)
