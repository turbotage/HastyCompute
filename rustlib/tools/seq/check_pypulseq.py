"""Cross-check hasty-seq against pypulseq: SAFE PNS (pypulseq.utils.safe_pns_prediction) and gradient moments
(independent fine-grid numerical integration). Input: <run>/check/ written by `seqdesign <run> --check`.
    rustlib/.venv/bin/python -I rustlib/tools/seq/check_pypulseq.py <run>
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pypulseq.utils.safe_pns_prediction import safe_gwf_to_pns, safe_example_hw


def hw_from(c):
    hw = SimpleNamespace(name=c["hw_name"], checksum="", dependency="")
    for ax, a in zip("xyz", c["hw"]):
        setattr(hw, ax, SimpleNamespace(tau1=a["tau"][0], tau2=a["tau"][1], tau3=a["tau"][2], a1=a["a"][0], a2=a["a"][1], a3=a["a"][2],
                                        stim_limit=a["stim_limit"], stim_thresh=a["stim_thresh"], g_scale=a["g_scale"]))
    return hw


def moments_fine(g, dt, up=64):
    # piecewise-linear G on a fine grid, cumulative trapezoid of G, G t, G t² (t from 0), sampled back at the raster points
    n = g.shape[0]
    tf = np.arange((n - 1) * up + 1) * dt / up
    gf = np.stack([np.interp(tf, np.arange(n) * dt, g[:, a]) for a in range(3)], 1)
    out = []
    for p in range(3):
        f = gf * tf[:, None] ** p
        c = np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (f[1:] + f[:-1]) * (dt / up), 0)])
        out.append(c[::up])
    return np.concatenate(out, 1)


def main():
    run = Path(sys.argv[1]) / "check"
    c = json.loads((run / "check.json").read_text())
    dt = c["dt"]
    hw = hw_from(c)
    ex = safe_example_hw()
    same_as_example = all(getattr(hw, a).__dict__ == getattr(ex, a).__dict__ for a in "xyz")
    print(f"hw {c['hw_name']} (identical to pypulseq safe_example_hw: {same_as_example})")
    ok = True
    for name in c["cases"]:
        gwf = np.load(run / f"{name}_gwf.npy")
        pr = np.load(run / f"{name}_pns.npy")
        pp, _ = safe_gwf_to_pns(gwf, np.zeros(gwf.shape[0]), dt, hw, True)
        pn = np.sqrt((pp ** 2).sum(1))
        e_ax = np.abs(pp - pr[:, :3]).max()
        e_n = np.abs(pn - pr[:, 3]).max()
        rel = e_n / max(pn.max(), 1e-30)
        m_r = np.load(run / f"{name}_moments.npy")
        m_p = moments_fine(gwf, dt)
        scale = np.abs(m_p).max(0) + 1e-30
        e_m = (np.abs(m_r - m_p) / scale).max(0)
        # the fine-grid trapezoid has O(h²) error; exact Simpson (Rust) should agree to that level
        good = rel < 1e-9 and e_m.max() < 1e-4
        ok &= good
        print(f"  {name:18s} n={gwf.shape[0]:6d}  PNS max {pn.max():7.2f}% (rust {pr[:, 3].max():7.2f}%)  |Δ| axis {e_ax:.1e} norm {e_n:.1e} rel {rel:.1e}"
              f"  moments rel err M0 {e_m[:3].max():.1e} M1 {e_m[3:6].max():.1e} M2 {e_m[6:].max():.1e}  {'OK' if good else 'MISMATCH'}")
    print("all OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
