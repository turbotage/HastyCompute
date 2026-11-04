import bpy
c = bpy.data.collections.get("5: Cardiovascular system")
def rec(c, d=0):
    print("CV", "  " * d + c.name, len(c.objects), sum(o.type == "CURVE" for o in c.objects))
    for ch in c.children:
        rec(ch, d + 1)
rec(c)
r = bpy.data.collections.get("9: Regions of human body")
print("REGIONS", [o.name for o in r.all_objects if o.type == "MESH"][:40])
v = bpy.data.collections.get("8: Visceral systems")
print("KIDNEY-ish", [o.name for o in v.all_objects if o.type == "MESH" and ("idney" in o.name or "enal" in o.name)][:30])
print("VIS CURVES", [o.name for o in v.all_objects if o.type == "CURVE"][:10])
