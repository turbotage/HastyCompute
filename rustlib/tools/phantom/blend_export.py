"""Export the Z-Anatomy atlas for the flow phantom (run inside Blender, scripts in the .blend disabled):

    blender -b --disable-autoexec Startup.blend --python blend_export.py -- <out_dir>

- vessels.npz: every cardiovascular curve sampled at ≤ 0.5 mm arc length in world coordinates (m): points [N, 3],
  radius [N] (m; bevel depth × point radius × object scale), vessel [N] (index), spline [N] (index within the vessel),
  ordered along each spline; vessels.json: names, parent names (the atlas's vessel hierarchy), sample ranges.
- meshes_<system>.npz + meshes.json: every mesh object (modifiers applied, triangulated, world coordinates, float32),
  vertices / triangles concatenated with per-object offsets, plus name, collection path and a closed-surface flag
  (every edge shared by exactly two triangles) — only closed surfaces can be voxelised as solids. Objects with a
  mirroring world transform (det < 0, the atlas's ".r" copies) get their faces reversed so every surface keeps its
  outward orientation (needed by the winding-number voxeliser).
Licence of the exported data: CC BY-SA 4.0 (Z-Anatomy; BodyParts3D CC BY-SA 2.1 JP). "Kidney.l/.r" (CC BY-NC 4.0, by
Lissie Cowley) is not exported.
"""
import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import geometry

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)
EXCLUDE = {"Kidney.l", "Kidney.r"}  # CC BY-NC (not redistributable under BY-SA)


def collection_paths():
    paths = {}

    def rec(c, path):
        p = path + [c.name]
        for o in c.objects:
            paths.setdefault(o.name, "/".join(p))
        for ch in c.children:
            rec(ch, p)

    rec(bpy.context.scene.collection, [])
    return paths


cpath = collection_paths()

# ── vessels ────────────────────────────────────────────────────────────────────────────────────────────────────────
cv = bpy.data.collections["5: Cardiovascular system"]
pts, rad, vid, sid, info = [], [], [], [], []
for i, o in enumerate(o for o in cv.all_objects if o.type == "CURVE"):
    M = o.matrix_world
    scale = float(np.cbrt(abs(np.linalg.det(np.array(M.to_3x3())))))
    bevel = o.data.bevel_depth
    start = len(pts)
    for si, s in enumerate(o.data.splines):
        bp = s.bezier_points
        segs = len(bp) - 1 + (1 if s.use_cyclic_u else 0)
        for k in range(segs):
            a, b = bp[k], bp[(k + 1) % len(bp)]
            p0, p1 = M @ a.co, M @ b.co
            chord = (p1 - p0).length
            n = max(2, int(np.ceil(chord / 0.5e-3)) + 1)
            sub = geometry.interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, n)
            for j, q in enumerate(sub):
                if k > 0 and j == 0:
                    continue  # shared with the previous segment
                t = j / (n - 1)
                w = M @ q
                pts.append((w.x, w.y, w.z))
                rad.append(bevel * ((1 - t) * a.radius + t * b.radius) * scale)
                vid.append(i)
                sid.append(si)
    info.append({"name": o.name, "parent": o.parent.name if o.parent is not None else None, "start": start, "end": len(pts), "splines": len(o.data.splines)})
np.savez_compressed(out / "vessels.npz", points=np.asarray(pts, np.float32), radius=np.asarray(rad, np.float32), vessel=np.asarray(vid, np.int32), spline=np.asarray(sid, np.int32))
(out / "vessels.json").write_text(json.dumps({"vessels": info, "units": "m", "source": "Z-Anatomy (CC BY-SA 4.0), BodyParts3D (CC BY-SA 2.1 JP)"}, indent=0))
print(f"EXPORT vessels: {len(info)} curves, {len(pts)} samples")

# ── meshes ─────────────────────────────────────────────────────────────────────────────────────────────────────────
dg = bpy.context.evaluated_depsgraph_get()
systems = {}
for o in bpy.data.objects:
    if o.type != "MESH" or o.name in EXCLUDE:
        continue
    path = cpath.get(o.name, "")
    top = path.split("/")[1] if "/" in path else "other"
    systems.setdefault(top, []).append(o)
index = {}
for top, objs in systems.items():
    V, T, offs, meta = [], [], [0], []
    nv = 0
    for o in objs:
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        closed = all(len(e.link_faces) == 2 for e in bm.edges) and len(bm.edges) > 0
        bm.transform(o.matrix_world)
        mirrored = o.matrix_world.to_3x3().determinant() < 0
        if mirrored:  # a mirroring transform flips the orientation; keep outward normals (winding number +1 inside)
            bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
        v = np.array([tuple(x.co) for x in bm.verts], np.float32).reshape(-1, 3)
        t = np.array([[x.verts[0].index, x.verts[1].index, x.verts[2].index] for x in bm.faces], np.int32).reshape(-1, 3)
        bm.free()
        oe.to_mesh_clear()
        V.append(v)
        T.append(t + nv)
        nv += len(v)
        offs.append(offs[-1] + len(t))
        meta.append({"name": o.name, "path": cpath.get(o.name, ""), "closed": bool(closed), "mirrored": bool(mirrored), "verts": int(len(v)), "tris": int(len(t))})
    key = "".join(ch if ch.isalnum() else "_" for ch in top)[:40]
    np.savez_compressed(out / f"meshes_{key}.npz", verts=np.concatenate(V) if V else np.zeros((0, 3), np.float32), tris=np.concatenate(T) if T else np.zeros((0, 3), np.int32), tri_offsets=np.asarray(offs, np.int64))
    index[key] = {"collection": top, "objects": meta}
    print(f"EXPORT meshes {top}: {len(objs)} objects, {sum(m['tris'] for m in meta)} triangles, {sum(m['closed'] for m in meta)} closed")
(out / "meshes.json").write_text(json.dumps(index, indent=0))
print("EXPORT done", out)
