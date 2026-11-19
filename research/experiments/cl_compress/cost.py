"""Per-encode-pass cost (forward + adjoint, 30 M samples, 96 knots, s = 8, order 3) from the channel counts.

Unit costs from temporal_operator.md section 22.6 (RTX 4090, measured/extrapolated there for C*L = 300 channels):
  box 160 mm at 1 mm (cb 10, 100 channels per pass): image 25 s + sample 122 s (mid of 117-127) = 147 s / 300 ch
  torso 1 mm (cb 1, 10 channels per pass):           image 651 s + sample 612 s = 1263 s / 300 ch
Both sides scale with the channel count (image: FFTs; sample: passes of a fixed channel batch, section 22.3).
Joint form adds per pass: mixing (C x batch MACs/sample, B rows read from L2 at ~4 TB/s) and Phi generation
(CK MACs per voxel per channel, once per batch, fp32 at ~40 TFLOP/s effective) [estimates].
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIT = {"box": (147.0 / 300, 100, 4.1e6), "torso": (1263.0 / 300, 10, 94.4e6)}
M, C = 30e6, 30


def extra(region, R, CK):
    per_ch, batch, nv = UNIT[region]
    passes = -(-R // batch)
    mix = 2 * passes * (M * C * batch * 8 / 4e12)  # B-table bytes per sample per pass, both directions
    phi = 2 * passes * (8 * nv * batch * CK / 40e12)  # generate Phi for the batch once per pass-direction
    return mix + phi


def main():
    rows = []
    for reg, fn in (("box", "box_ratemap.json"), ("torso", "object_ratemap.json")):
        p = HERE / "out" / fn
        if not p.exists():
            continue
        o = json.load(open(p))
        CK = o["C"] * o["K"]
        per_ch = UNIT[reg][0]
        for key, eps in (("em", None), ("eu", "0.001"), ("eu", "0.0001"), ("ed", "0.001")):
            r = o["res"][key]
            e = eps or str(r["eps_noise"])
            ch = {"today (fixed, count LS)": r["fixed_L"][e], "fixed_e": r["fixed_e_L"][e], "seg_svd": r["seg_svd_L"][e]}
            ch = {k: (v * C if v else None) for k, v in ch.items()}
            ch["(b') coil-adaptive"] = r["block_coil"][e]
            ch["(d) joint"] = r["d"][e]
            cells = []
            for k, v in ch.items():
                if v is None:
                    cells.append(f"{k}: > {24 * C} ch")
                    continue
                t = v * per_ch + (extra(reg, v, CK) if k == "(d) joint" else 0.0)
                cells.append(f"{k}: {v} ch, {t:.0f} s")
            lab = f"{key} eps {float(e):.1e}" + (" (noise rule)" if eps is None else "")
            rows.append(f"| {reg} (unit {'box' if reg == 'box' else 'object R'}) | {lab} | " + " | ".join(cells) + " |")
            if k == "(d) joint":
                rows[-1] += f" mixing+Phi {extra(reg, v, CK):.1f} s |"
    print("\n".join(rows))


if __name__ == "__main__":
    main()
