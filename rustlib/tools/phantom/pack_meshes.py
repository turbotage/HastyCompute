"""Pack the atlas meshes of each tissue group for the voxeliser (`phantomvox`).

    python -I pack_meshes.py <export_dir> <out.hv>

Groups (label priority low → high; later groups overwrite earlier ones). Flags for the voxeliser:
  per_member       evaluate each member inside its own bounding box (muscles, bones)
  envelope         the group is a shell; the voxeliser also returns the enclosed solid (skin → body, organ walls →
                   lumen, myocardium → chambers)
  member_envelope  also return which member's inner surface encloses each cavity voxel (heart chambers)
Every group keeps its member meshes separate.
Licence of the meshes: Z-Anatomy CC BY-SA 4.0 / BodyParts3D CC BY-SA 2.1 JP.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

E = Path(sys.argv[1])
M = json.loads((E / "meshes.json").read_text())

SKIP = re.compile(r"(?i)fascia|intermuscular septum|\.[gjio]$")  # sheets and empty junction / label objects
# name, system, member regex, flags
RECIPE = [
    ("body", "9__Regions_of_human_body", r".*", {"envelope": True}),
    ("muscle", "4__Muscular_system", r".*", {"per_member": True}),
    ("bone", "1__Skeletal_system", r"^(?!.*[Cc]artilage)", {"per_member": True}),
    ("cartilage", "1__Skeletal_system", r"[Cc]artilage", {"per_member": True}),
    ("liver", "8__Visceral_systems", r"^(Liver|.*segment of liver.*)$", {}),
    ("gallbladder", "8__Visceral_systems", r"^Gallbladder$", {}),
    ("spleen", "6__Lymphoid_organs", r"^Spleen$", {}),
    ("thymus", "6__Lymphoid_organs", r"^(Left|Right) lobe of thymus$", {}),
    ("stomach", "8__Visceral_systems", r"^(Stomach|Mucosa of stomach)$", {"envelope": True}),
    ("pancreas", "8__Visceral_systems", r"^Pancreas$", {}),
    ("intestine", "8__Visceral_systems", r"^(Ascending colon|Transverse colon|Descending colon|Sigmoid colon|Jejunum|Ileum|Duodenum|Caecum|Rectum)$", {"envelope": True}),
    ("thyroid", "8__Visceral_systems", r"^Thyroid gland$", {}),
    ("lung", "8__Visceral_systems", r"^(Superior|Middle|Inferior) lobe of (left|right) lung$", {}),
    ("airway", "8__Visceral_systems", r"^(Trachea|Tracheobronchial tree\.j)$", {"envelope": True}),
    ("heart_wall", "5__Cardiovascular_system", r"^(Left|Right) (atrium|ventricle)$", {"envelope": True, "member_envelope": True}),
    ("papillary", "5__Cardiovascular_system", r"papillary muscle", {}),
    ("valve", "5__Cardiovascular_system", r"(leaflet|Aortic valve\.j)", {}),
]

out = {}
for name, system, pat, flags in RECIPE:
    if system not in M:
        print(f"{name}: no system {system}")
        continue
    data = np.load(E / f"meshes_{system}.npz")
    V, T, off = data["verts"], data["tris"], data["tri_offsets"]
    rx = re.compile(pat)
    tris, members, offs = [], [], [0]
    for k, o in enumerate(M[system]["objects"]):
        if not rx.search(o["name"]) or (flags.get("per_member") and SKIP.search(o["name"])):
            continue
        t = T[off[k] : off[k + 1]]
        if len(t) == 0:
            continue
        tris.append(V[t].reshape(-1, 9))  # each row: three vertices
        members.append(o["name"])
        offs.append(offs[-1] + len(t))
    if not tris:
        print(f"{name}: no meshes matched")
        continue
    tv = np.concatenate(tris).astype(np.float32)
    if name == "body":  # drop stray atlas parts far outside the body (labels / legends)
        keep = np.abs(tv.reshape(-1, 3, 3)[:, :, 0]).max(axis=1) < 0.4
        tv = tv[keep]
        offs = [0, len(tv)]
        members = ["body (all skin patches)"]
    out[name] = {"tri": tv, "members": members, "offsets": np.asarray(offs, np.int64), **{k: bool(flags.get(k, False)) for k in ("per_member", "envelope", "member_envelope")}}
    print(f"{name:12s} {len(members):4d} meshes {len(tv):8d} triangles  bbox {tv.reshape(-1, 3).min(0).round(3)} … {tv.reshape(-1, 3).max(0).round(3)}")
vf.write(sys.argv[2], {"groups": [{"name": k, **v} for k, v in out.items()], "units": "m",
                      "licence": "Z-Anatomy CC BY-SA 4.0; BodyParts3D CC BY-SA 2.1 JP"})
print("wrote", sys.argv[2])
