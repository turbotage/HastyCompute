"""Anatomical vessel variants of a phantom's graph (research/flow_phantom.md, variants): hypoplastic / absent / added
segments of named vessels, e.g. circle-of-Willis variants, as data instead of code. Applied by make_phantom.py
(`--vessel-variant <preset|spec.json>`) to the graph copy before the radius jitter, the network and the clearance
solver, so the variant is cleared and its flows follow from the network like any other phantom.

    python -I vessel_variants.py <graph_dir> --spec <preset|spec.json> [--out <dir>]   (in place without --out)

Spec: {"ops": [op, …]}, applied in order. A vessel is selected by its edge name (fnmatch pattern, e.g.
"Posterior communicating artery.*"); an edge at a junction by {"name": N, "at": [X, Y]}: the edge named N incident to
the node where an edge named X meets an edge named Y (X = Y: two edges of that name, e.g. the A1 / A2 node of an
anterior cerebral artery).
- {"op": "scale", "name": N | "edge": {...}, "factor": f}            radius × f (hypoplasia f < 1)
- {"op": "radius", "name": N | "edge": {...}, "radius_mm": r}        radius set to r (a drawn calibre corrected)
- {"op": "remove", "name": N | "edge": {...}}                        aplasia: the edges go (orphan nodes too); fails
                                                                      when a root-less piece would remain
- {"op": "connect", "name": N, "a": {"at": [X, Y]}, "b": {"at": [X, Y]}, "radius_mm": r}
                                                                      a new straight segment between two junction
                                                                      nodes (a communicating artery missing in the atlas)
Presets: cow_atlas (as drawn: no ACoA, PCoA r 1.7 mm), cow_complete (ACoA r 0.75 mm, PCoAs r 0.65 mm), cow_no_pcoa
(both PCoAs absent, ACoA present), cow_fetal_pcoa_l (left PCoA r 1.1 mm, left P1 hypoplastic ×0.4), cow_a1_hypo_r
(right A1 hypoplastic ×0.4, ACoA present).
"""
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "python/lib/hastycompute").is_dir()) / "python/lib"))
from hastycompute.store import load_any  # noqa: E402

_ACOA = {"op": "connect", "name": "Anterior communicating artery", "a": {"at": ["Anterior cerebral artery.l", "Anterior cerebral artery.l"]},
         "b": {"at": ["Anterior cerebral artery.r", "Anterior cerebral artery.r"]}, "radius_mm": 0.75}
PRESETS = {
    "cow_atlas": {"ops": []},
    "cow_complete": {"ops": [_ACOA, {"op": "radius", "name": "Posterior communicating artery.*", "radius_mm": 0.65}]},
    "cow_no_pcoa": {"ops": [_ACOA, {"op": "remove", "name": "Posterior communicating artery.*"}]},
    "cow_fetal_pcoa_l": {"ops": [_ACOA, {"op": "radius", "name": "Posterior communicating artery.r", "radius_mm": 0.65},
                                 {"op": "radius", "name": "Posterior communicating artery.l", "radius_mm": 1.1},
                                 {"op": "scale", "edge": {"name": "Posterior cerebral artery.l", "at": ["Basilar artery", "Posterior cerebral artery.l"]}, "factor": 0.4}]},
    "cow_a1_hypo_r": {"ops": [_ACOA, {"op": "radius", "name": "Posterior communicating artery.*", "radius_mm": 0.65},
                              {"op": "scale", "edge": {"name": "Anterior cerebral artery.r", "at": ["Middle cerebral artery (M1-segment).r", "Anterior cerebral artery.r"]}, "factor": 0.4}]},
}


def load_spec(s):
    if s in PRESETS:
        return json.loads(json.dumps(PRESETS[s]))
    if s.strip().startswith("{"):
        return json.loads(s)
    return json.loads(Path(s).read_text())


def apply(g, gj, spec, log=print):
    """apply the ops to a graph (g: dict of arrays as in graph.npz, gj: graph.json dict); returns (g, gj)"""
    g = {k: np.array(v) for k, v in g.items()}
    # edges whose calibre the variant sets: the Murray / velocity floors leave them alone (murray_floor.py)
    if "variant_fixed" not in g:
        g["variant_fixed"] = np.zeros(len(g["edge_a"]), np.int8)
    for op in spec.get("ops", []):
        names = gj["edge_names"]
        eo, ea, eb = g["edge_offsets"], g["edge_a"], g["edge_b"]
        inc = {}
        for k in range(len(ea)):
            inc.setdefault(int(ea[k]), []).append(k)
            inc.setdefault(int(eb[k]), []).append(k)

        def junction(xy):
            x, y = xy
            hits = [n for n, ks in inc.items() if (sum(names[k] == x for k in ks) >= 2 if x == y else (any(names[k] == x for k in ks) and any(names[k] == y for k in ks)))]
            if len(hits) != 1:
                raise SystemExit(f"vessel variant: junction {xy} matches {len(hits)} nodes")
            return hits[0]

        def select(op_):
            if "edge" in op_:
                e = op_["edge"]
                n = junction(e["at"])
                ks = [k for k in inc[n] if fnmatch.fnmatchcase(names[k], e["name"])]
            else:
                ks = [k for k, nm in enumerate(names) if fnmatch.fnmatchcase(nm, op_["name"])]
            if not ks:
                raise SystemExit(f"vessel variant: {op_} selects no edge")
            return ks

        kind = op["op"]
        if kind in ("scale", "radius"):
            ks = select(op)
            for k in ks:
                sl = slice(int(eo[k]), int(eo[k + 1]))
                for key in ("radius", "radius_atlas", "radius_ref"):
                    if key in g:
                        g[key][sl] = g[key][sl] * op["factor"] if kind == "scale" else op["radius_mm"] * 1e-3
                g["variant_fixed"][k] = 1
            log(f"vessel variant {kind}: {len(ks)} edges ({sorted({names[k] for k in ks})}) " + (f"× {op['factor']}" if kind == "scale" else f"r {op['radius_mm']} mm"))
        elif kind == "remove":
            ks = set(select(op))
            keep = [k for k in range(len(ea)) if k not in ks]
            g, gj = _subset(g, gj, keep)
            log(f"vessel variant remove: {len(ks)} edges ({sorted({names[k] for k in ks})})")
            _check_roots(g, gj)
        elif kind == "connect":
            na, nb = junction(op["a"]["at"]), junction(op["b"]["at"])
            pa, pb = g["node_pos"][na].astype(np.float64), g["node_pos"][nb].astype(np.float64)
            n = max(2, int(np.ceil(np.linalg.norm(pb - pa) / 0.5e-3)) + 1)
            pts = pa + np.linspace(0, 1, n)[:, None] * (pb - pa)
            r = np.full(n, op["radius_mm"] * 1e-3, np.float32)
            comp = int(g["edge_comp"][inc[na][0]])
            assert comp == int(g["edge_comp"][inc[nb][0]]), "vessel variant connect: the two nodes lie in different trees"
            g["points"] = np.vstack([g["points"], pts.astype(g["points"].dtype)])
            for key in ("radius", "radius_atlas", "radius_ref"):
                if key in g:
                    g[key] = np.concatenate([g[key], r.astype(g[key].dtype)])
            g["edge_offsets"] = np.append(g["edge_offsets"], g["edge_offsets"][-1] + n)
            g["edge_a"] = np.append(g["edge_a"], na).astype(g["edge_a"].dtype)
            g["edge_b"] = np.append(g["edge_b"], nb).astype(g["edge_b"].dtype)
            g["direction"] = np.append(g["direction"], 1).astype(g["direction"].dtype)
            g["edge_comp"] = np.append(g["edge_comp"], comp).astype(g["edge_comp"].dtype)
            g["variant_fixed"] = np.append(g["variant_fixed"], 1).astype(np.int8)
            gj["edge_names"] = gj["edge_names"] + [op["name"]]
            log(f"vessel variant connect: {op['name']} {np.linalg.norm(pb - pa) * 1e3:.1f} mm, r {op['radius_mm']} mm, nodes {na} – {nb}")
        else:
            raise SystemExit(f"vessel variant: unknown op {kind}")
    gj.setdefault("vessel_variants", []).append(spec)
    return g, gj


def _subset(g, gj, keep):
    """the graph restricted to the edges `keep` (orphan nodes dropped, indices remapped)"""
    eo = g["edge_offsets"]
    seg = [np.arange(eo[k], eo[k + 1]) for k in keep]
    sidx = np.concatenate(seg) if seg else np.zeros(0, int)
    lens = np.array([len(s) for s in seg])
    used = sorted({int(g["edge_a"][k]) for k in keep} | {int(g["edge_b"][k]) for k in keep})
    remap = {n: i for i, n in enumerate(used)}
    out = dict(g)
    out["node_pos"] = g["node_pos"][used]
    out["edge_a"] = np.array([remap[int(g["edge_a"][k])] for k in keep], g["edge_a"].dtype)
    out["edge_b"] = np.array([remap[int(g["edge_b"][k])] for k in keep], g["edge_b"].dtype)
    out["edge_offsets"] = np.r_[0, np.cumsum(lens)].astype(g["edge_offsets"].dtype)
    for key in ("points", "radius", "radius_atlas", "radius_ref"):
        if key in g:
            out[key] = g[key][sidx]
    for key in ("direction", "edge_comp", "variant_fixed"):
        if key in g:
            out[key] = g[key][keep]
    gj = dict(gj)
    gj["edge_names"] = [gj["edge_names"][k] for k in keep]
    # node references in the components (roots) follow the remap; a removed root fails the check below
    comps = []
    for c in gj["components"]:
        c = dict(c)
        if "roots" in c:
            c["roots"] = [remap[r] for r in c["roots"] if r in remap]
        if c.get("root_node") is not None:
            c["root_node"] = remap.get(c["root_node"], None)
        comps.append(c)
    gj["components"] = comps
    return out, gj


def _check_roots(g, gj):
    """every connected piece must contain a root node of its component (else it would carry no flow)"""
    nn = len(g["node_pos"])
    par = list(range(nn))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in zip(g["edge_a"], g["edge_b"]):
        par[f(int(a))] = f(int(b))
    roots = {f(r) for c in gj["components"] for r in (c.get("roots") or ([c["root_node"]] if c.get("root_node") is not None else []))}
    pieces = {f(int(a)) for a in g["edge_a"]}
    bad = pieces - roots
    if bad:
        raise SystemExit(f"vessel variant: {len(bad)} pieces of the graph lost their root (a removal cut a tree)")


if __name__ == "__main__":
    A = sys.argv[1:]
    D = Path(A[0])
    spec = load_spec(A[A.index("--spec") + 1])
    out = Path(A[A.index("--out") + 1]) if "--out" in A else D
    g = dict(load_any(D / "graph.npz"))
    gj = json.loads((D / "graph.json").read_text())
    g, gj = apply(g, gj, spec)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "graph.npz", **g)
    (out / "graph.json").write_text(json.dumps(gj))
    print(f"→ {out}")
