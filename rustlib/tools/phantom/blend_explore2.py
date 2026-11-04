import collections
import bpy

def members(name):
    c = bpy.data.collections.get(name)
    out = []
    def rec(c):
        out.extend(c.objects)
        for ch in c.children:
            rec(ch)
    rec(c)
    return out

cv = members("5: Cardiovascular system")
curves = [o for o in cv if o.type == "CURVE"]
meshes = [o for o in cv if o.type == "MESH"]
for o in curves[:8]:
    cd = o.data
    ns = len(cd.splines)
    s = cd.splines[0]
    pts = s.bezier_points if s.type == "BEZIER" else s.points
    radii = [p.radius for p in pts]
    print("CURVE", repr(o.name), "splines", ns, "type", s.type, "npts", len(pts), "bevel_depth", round(cd.bevel_depth, 5), "bevel_mode", cd.bevel_mode, "radius range", round(min(radii), 4), round(max(radii), 4), "dims", tuple(round(x, 3) for x in o.dimensions), "parent", o.parent.name if o.parent else None)
print("CURVE bevel depths", collections.Counter(round(o.data.bevel_depth, 4) for o in curves).most_common(6))
print("CURVE spline types", collections.Counter(s.type for o in curves for s in o.data.splines))
print("CURVE names sample", [o.name for o in curves[::40]])
print("MESH cardiovascular", [o.name for o in meshes])
vis = [o for o in members("8: Visceral systems") if o.type == "MESH"]
print("MESH visceral sample", [o.name for o in vis[::6]])
print("UNITS", bpy.context.scene.unit_settings.system, bpy.context.scene.unit_settings.scale_length)
