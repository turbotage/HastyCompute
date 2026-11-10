"""The excited slab of a scanner-like prescription, from the phantom's anatomy (prep_sim --slab auto, the LBM spin
history of the same protocol): axial, from 10 mm below the heart (a bit of the IVC) to 5 mm above the aortic arch.

    rustlib/.venv/bin/python -I rustlib/tools/phantom/slab.py <phantom_dir>      → prints and writes <phantom>/slab.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402


def auto_slab(ph, labels=None, below=10e-3, above=5e-3):
    """(z_lo, z_hi) in m; `labels`: a tissue_*.hv value (default the phantom's own atlas or p0 at 1 mm)"""
    ph = Path(ph)
    if labels is None:
        p0 = ph / "atlas" if (ph / "atlas" / "tissue_1mm.hv").exists() else REPO / "data/hasty/phantom/p0"
        labels = vf.read(p0 / "tissue_1mm.hv")
    lab = np.asarray(labels["labels"])
    names = list(labels["tissues"])
    o, h = np.asarray(labels["origin_m"], float), float(labels["voxel_m"])
    hz = np.nonzero(np.isin(lab, [names.index(c) for c in ("blood_LA", "blood_LV", "blood_RA", "blood_RV", "myocardium")]).any((1, 2)))[0]
    z_lo = o[2] + hz.min() * h - below
    G = np.load(ph / "graph.npz")
    GJ = json.loads((ph / "graph.json").read_text())
    eo = G["edge_offsets"]
    arch = [k for k, n in enumerate(GJ["edge_names"]) if "aortic arch" in n.lower()]
    z_hi = (max(float((G["points"][eo[k] : eo[k + 1], 2] + G["radius"][eo[k] : eo[k + 1]]).max()) for k in arch) + above) if arch else o[2] + hz.max() * h + 50e-3
    return float(z_lo), float(z_hi)


if __name__ == "__main__":
    ph = Path(sys.argv[1])
    z0, z1 = auto_slab(ph)
    out = {"z_lo": z0, "z_hi": z1, "center_z": 0.5 * (z0 + z1), "thk": z1 - z0}
    (ph / "slab.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
