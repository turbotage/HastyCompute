"""Grid-refinement regions of a cardiac LBM case (hasty-cfd refine.rs, kernels/lbm_core.h "grid refinement"): level 1
(dx / 2) over spheres around the named valves along their path through the cycle — the valve centre, points downstream
along its normal (the jet, `down`) and half a radius upstream, each moved by the cardiac motion at every 4th motion
sample — of radius R (lumen) + `margin`; level 2 (dx / 4), when asked, with half the margin and the downstream reach.

lbm_case.py --refine uses `spheres`; as a script it adds the regions to an existing case: a new case directory with the
refined case.json, its large arrays symlinked to the old ones (no copies):
    rustlib/.venv/bin/python -I rustlib/tools/phantom/refine_spheres.py <case_dir> <new_case_dir> --refine aortic,pulmonary|all
        [--refine-margin-mm 6] [--refine-down-mm 10] [--refine-levels 1]"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402


def spheres(valves, cm, names, margin=6e-3, down=10e-3, levels=1):
    """the case's "refine" list: [{"level": l, "spheres": [{"center", "radius"}]}] (m)"""
    U = np.asarray(cm["U_mm"], np.float32) * 1e-3  # [T, 3, Z, Y, X]
    m_o = np.asarray(cm["origin_m"], float)
    m_h = float(cm["voxel_m"])

    def moved(X):  # X [n, 3] reference → [T', n, 3] lab positions at every 4th motion sample
        ic = ((X - m_o) / m_h - 0.5)[:, ::-1].T
        return np.stack([X + np.stack([ndimage.map_coordinates(U[it, a], ic, order=1, mode="nearest") for a in range(3)], -1) for it in range(0, U.shape[0], 4)])

    out = []
    for lev in range(1, levels + 1):
        sph = []
        for v in valves:
            if v["name"] not in names:
                continue
            c0, nv, R = np.asarray(v["center"], float), np.asarray(v["normal"], float), float(v.get("lumen_radius") or v["radius"])
            ss = np.arange(-0.5 * R, down / lev + 1e-9, max(0.5 * R, 2e-3))
            for P in moved(c0[None] + ss[:, None] * nv[None]).reshape(-1, 3):
                sph.append({"center": P.tolist(), "radius": R + margin / lev})
        if not sph:
            raise SystemExit(f"refine {names}: none of the case's valves {[v['name'] for v in valves]}")
        out.append({"level": lev, "spheres": sph})
        print(f"refinement level {lev}: {len(sph)} spheres around {', '.join(v['name'] for v in valves if v['name'] in names)} (margin {margin / lev * 1e3:.1f} mm, downstream {down / lev * 1e3:.1f} mm)")
    return out


if __name__ == "__main__":
    A = sys.argv[1:]

    def opt(n, d=None):
        return A[A.index(n) + 1] if n in A else d

    src, dst = Path(A[0]), Path(A[1])
    case = json.loads((src / "case.json").read_text())
    names = [v["name"] for v in case["valves"]] if opt("--refine", "all") == "all" else opt("--refine").split(",")
    mf = Path(case["motion"]["file"])
    cm = vf.read(mf if mf.is_absolute() else src / mf)
    case["refine"] = spheres(case["valves"], cm, names, float(opt("--refine-margin-mm", 6.0)) * 1e-3, float(opt("--refine-down-mm", 10.0)) * 1e-3, int(opt("--refine-levels", 1)))
    case["source_refine"] = {"from": str(src), "args": A[2:]}
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix == ".npy" and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve())
    (dst / "case.json").write_text(json.dumps(case, indent=1))
    print(f"→ {dst / 'case.json'}")
