#!/usr/bin/env python3
"""Markdown tables from nc_toy.py results.  Usage: python nc_tables.py OUTDIR [WT|WL]"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
prior = sys.argv[2] if len(sys.argv) > 2 else "WT"
files = sorted(out.glob(f"nc_{prior}_R*.json"), key=lambda p: float(p.stem.split("_R")[1]))
res = [json.loads(f.read_text()) for f in files]


def g(v):
    return "—" if v is None else (f"{v:.0f}" if isinstance(v, (int, float)) and v >= 1 else f"{v:.2g}")


def e(v):
    return "—" if v is None else f"{v:.1e}"


print("## setup per R\n")
print("| R | spokes/set | pooled R | angular undersampling | lambda factor | truth err x*_l1 | held-out x*_l1 |"
      " pg residual | ADMM check | dist x_g→x* (g=1/0.1/0.01) | truth x_g (1/0.1/0.01) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in res:
    ref = r["ref"]
    sg = [ref[f"smooth_gamma{k}"] for k in ("1", "0.1", "0.01")]
    print(f"| {r['R']:g} | {min(r['spokes_per_set'])}–{max(r['spokes_per_set'])} | {r['pooled_R']:.2f} |"
          f" {r['angular_undersampling']:.0f} | {r['lam_factor']} | {ref['truth_err_xs_l1']:.4f} |"
          f" {ref['heldout_xs_l1']:.3f} | {e(ref.get('pg_residual'))} | {e(ref['admm_vs_admm_rho1_warm'])} |"
          f" {' / '.join(f'{s['dist_to_l1']:.2g}' for s in sg)} | {' / '.join(f'{s['truth_err']:.4f}' for s in sg)} |")
print()
for r in res:
    print(f"## R = {r['R']:g}\n")
    print("| method | target | →1e-1 | →1e-2 | →1e-3 | err@40 | err@200 | truth@40 | objgap@40 | held-out@40 | notes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in r["runs"].items():
        if "best_truth_it" in s:
            print(f"| {name} | (none) | | | | | | | | | best truth {s['best_truth_err']:.3f} @ {s['best_truth_it']},"
                  f" held-out pick {s['heldout_min_it']} → {s['truth_at_heldout_min']:.3f}, @100 {s['truth_at_100']:.3f} |")
            continue
        cp = s["cp"].get("40", {})
        tgt = "l1" if s["target"] == "l1" else s["target"].replace("smooth gamma=", "γ=")
        note = "DIVERGED" if s["diverged"] else ""
        print(f"| {name} | {tgt} | {g(s['to_1e-1'])} | {g(s['to_1e-2'])} | {g(s['to_1e-3'])} | {e(s['at40'])} |"
              f" {e(s['final'])} | {g(s['truth_at40'])} | {e(cp.get('objgap'))} | {g(cp.get('heldout'))} | {note} |")
    print()
