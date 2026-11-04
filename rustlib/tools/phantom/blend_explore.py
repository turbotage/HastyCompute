"""List the structure of a Blender file (run inside Blender: blender -b --disable-autoexec FILE --python this)."""
import collections
import bpy

def walk(c, depth=0, out=None):
    out = out if out is not None else []
    kinds = collections.Counter(o.type for o in c.objects)
    out.append(("  " * depth) + f"{c.name}: {dict(kinds)}")
    for ch in c.children:
        walk(ch, depth + 1, out)
    return out

for line in walk(bpy.context.scene.collection)[:400]:
    print("TREE", line)
curves = [o for o in bpy.data.objects if o.type == "CURVE"]
print("TOTAL objects", len(bpy.data.objects), "curves", len(curves), "meshes", sum(o.type == "MESH" for o in bpy.data.objects))
for o in curves[:5]:
    cd = o.data
    s = cd.splines[0]
    pts = s.bezier_points if s.type == "BEZIER" else s.points
    print("CURVE", o.name, "| collections", [c.name for c in o.users_collection], "| type", s.type, "n", len(pts), "bevel", cd.bevel_depth, "bevel_obj", cd.bevel_object, "radius0", pts[0].radius, "scale", tuple(o.matrix_world.to_scale()))
