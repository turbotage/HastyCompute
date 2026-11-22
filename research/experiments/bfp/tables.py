"""Markdown tables for research/bfp_storage.md from the acc_*.json / bench logs (prints to stdout)."""
import json, os, re, sys
H = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(H, f)))
F = lambda v: f"{v:.1e}"
fmts = ["f32", "fp24", "bf16"] + [f"bfp{b}" for b in [8, 10, 12, 14, 16, 18, 20, 22, 24]]
what = sys.argv[1]
if what == "image":
    r = J("acc_image.json")
    for f in fmts:
        a, s = r["fixed"][f], r["svd"][f]
        print(f"| {f} | {F(a['pad_x'])} | {F(a['pad_S'])} / {F(a['crop_S'])} | {F(a['pad_P'])} / {F(a['crop_P'])} | {F(s['pad_P'])} / {F(s['crop_P'])} | {F(a['pad_all'])} / {F(a['crop_all'])} |")
if what == "P":
    r = J("acc_image.json")
    a = r["fixed"]; p = r["P_alternatives"]
    print(f"| stored f32 (today) | 80 | {F(a['f32']['buf_P_sigw'])} |")
    for b in [16, 20, 22, 24]:
        print(f"| BFP {b} | {10 * (4 * b + 1) / 16:.1f} | {F(a[f'bfp{b}']['buf_P_sigw'])} |")
    for b in [12, 16, 20, 24]:
        print(f"| phase angle {b} bits + decay from R2* (4 B) | {10 * b / 8 + 4:.0f} | {F(p[f'angle{b}'])} |")
    print(f"| on the fly, f32 radians (f, R2*) | 8 | {F(p['onthefly_f32_radians'])} |")
    print(f"| on the fly, u32 turns (phase0, dphase, R2*) | 12 | {F(p['onthefly_u32_turns'])} |")
if what == "forward":
    r = J("acc_forward.json")
    print("| format | x → y | y → z | grid (read by the gather) | all three chained | quadrature of the three | chained rms error / σ_k |")
    print("|---|---|---|---|---|---|---|")
    for f in fmts:
        st = r["stage"]
        ch = r["chained"].get(f)
        c = f"{F(ch['chained'])} | {F(ch['quadrature_of_stages'])} | {F(ch['rms_err_over_sigma_k'])}" if ch else "— | — | —"
        print(f"| {f} | {F(st['xy'][f])} | {F(st['yz'][f])} | {F(st['grid'][f])} | {c} |")
if what == "adjoint":
    r = J("acc_adjoint.json")
    for kind in ["data", "noise"]:
        print(f"\n{kind}:\n")
        print("| format | scatter output | z → y | y → x | chained, crop output | quadrature of the three |")
        print("|---|---|---|---|---|---|")
        for f in fmts:
            st = r[kind]["stage"]; ch = r[kind]["chained"].get(f)
            c = f"{F(ch['crop_output_rel'])} | {F(ch['quadrature_of_grid_stages'])}" if ch else "— | —"
            print(f"| {f} | {F(st['scatter'][f])} | {F(st['zy'][f])} | {F(st['yx'][f])} | {c} |")
if what == "kspace":
    r = J("acc_kspace.json")
    print({k: r[k] for k in ["sigma", "readouts", "peak_over_sigma", "rms_over_sigma"]})
    print("| format | blocks along samples (one coil): rms / σ | max / σ | blocks along coils (one sample): rms / σ | max / σ |")
    print("|---|---|---|---|---|")
    for f in ["f32"] + [f"bfp{b}" for b in [8, 10, 12, 14, 16, 18, 20, 22, 24]]:
        a, b = r["samples_per_coil"][f], r["coils_per_sample"][f]
        print(f"| {f} | {F(a['rms_err_over_sigma'])} | {F(a.get('max_err_over_sigma', float('nan')))} | {F(b['rms_err_over_sigma'])} | {F(b.get('max_err_over_sigma', float('nan')))} |")
if what == "bench":
    rows = {}
    for line in open(os.path.join(H, sys.argv[2] if len(sys.argv) > 2 else "bench.log")):
        m = re.match(r"\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\|\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
        if m:
            b, lay = int(m[1]), m[2]
            rows.setdefault(b, {})[lay] = (m[3], m[5], m[6], m[7], m[9], m[10])
    for b, d in sorted(rows.items()):
        cell = lambda lay, i: f"{d[lay][i]} ms, {d[lay][i + 1]} %, {d[lay][i + 2]}" if lay in d and d[lay][i] != "-" else "—"
        print(f"| {b} | {cell('warp', 0)} | {cell('warp', 3)} | {cell('hw-complex', 0)} | {cell('hw-complex', 3)} | {cell('run', 0)} | {cell('run', 3)} | {cell('random-c', 3)} |")
