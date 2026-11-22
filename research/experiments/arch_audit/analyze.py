#!/usr/bin/env python3
"""Digest kernaudit results.jsonl into the tables of research/arch_targets.md §5.

usage: analyze.py out/results.jsonl [out_lb4/results.jsonl ...]   (first file = baseline; others = what-if variants)
"""
import json, sys, collections

ARCHS = ["sm_80", "sm_89", "sm_90", "sm_100", "sm_120"]


def load(p):
    return [json.loads(l) for l in open(p)]


def key(r):
    return (r["case"], r["family"], r["cfg"])


def main():
    base = load(sys.argv[1])
    ok = [r for r in base if r["error"] is None]
    print(f"# rows {len(base)}, compile errors {len(base) - len(ok)}")
    for r in base:
        if r["error"]:
            print("  ERR", r["arch"], r["case"], r["family"], r["cfg"], r["error"].splitlines()[0][:150])

    print("\n## flag counts per family x arch: kernels / spill / stack-only / below 8 warps")
    fam = collections.defaultdict(lambda: collections.Counter())
    for r in ok:
        c = fam[(r["family"], r["arch"])]
        c["n"] += 1
        sp = r["spill_st"] > 0 or r["spill_ld"] > 0
        c["spill"] += sp
        c["stack"] += (r["stack"] > 0 and not sp)
        c["low"] += r["warps_sm"] < 8
    fams = sorted({f for f, _ in fam})
    print("| family | " + " | ".join(ARCHS) + " |")
    for f in fams:
        cells = []
        for a in ARCHS:
            c = fam.get((f, a))
            cells.append("-" if not c else f"{c['n']} / {c['spill']} / {c['stack']} / {c['low']}")
        print(f"| {f} | " + " | ".join(cells) + " |")

    print("\n## spilling configs (regs st/ld per arch)")
    by = collections.defaultdict(dict)
    for r in ok:
        by[key(r)][r["arch"]] = r
    for k, d in sorted(by.items()):
        if any(r["spill_st"] or r["spill_ld"] for r in d.values()):
            cells = []
            for a in ARCHS:
                r = d.get(a)
                cells.append("-" if r is None else (f"{r['regs']} S{r['spill_st']}/{r['spill_ld']}" if (r['spill_st'] or r['spill_ld']) else f"{r['regs']}"))
            print(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cells) + " |")

    print("\n## per case/family: max warps/SM over candidates, and median regs, per arch")
    grp = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ok:
        if r["family"] == "whatif":
            continue
        grp[(r["case"], r["family"])][r["arch"]].append(r)
    for k in sorted(grp):
        cells = []
        for a in ARCHS:
            rs = grp[k].get(a, [])
            if not rs:
                cells.append("-")
                continue
            regs = sorted(x["regs"] for x in rs)
            cells.append(f"{len(rs)}c, max {max(x['warps_sm'] for x in rs)}w, med {regs[len(regs)//2]}r")
        print(f"| {k[0]} | {k[1]} | " + " | ".join(cells) + " |")

    print("\n## limiter shift: configs limited by smem on sm_89 -> limiter / warps on sm_90 and sm_80")
    shift = collections.Counter()
    gains = collections.defaultdict(list)
    for k, d in by.items():
        a89 = d.get("sm_89")
        if not a89 or "smem" not in a89["limiter"]:
            continue
        for t in ["sm_80", "sm_90"]:
            r = d.get(t)
            if r:
                shift[(k[1], t, r["limiter"])] += 1
                gains[(k[1], t)].append(r["warps_sm"] / max(1, a89["warps_sm"]))
    for (f, t, lim), n in sorted(shift.items()):
        print(f"  {f:10s} {t}: {lim:22s} {n}")
    for (f, t), g in sorted(gains.items()):
        g = sorted(g)
        print(f"  {f:10s} {t}: warps ratio vs sm_89 median {g[len(g)//2]:.2f} max {g[-1]:.2f} (n={len(g)})")

    print("\n## smem-limited share per family x arch (non-whatif)")
    for f in fams:
        if f == "whatif":
            continue
        cells = []
        for a in ARCHS:
            rs = [r for r in ok if r["family"] == f and r["arch"] == a]
            if not rs:
                cells.append("-")
                continue
            cells.append(f"{sum('smem' in r['limiter'] for r in rs)}/{len(rs)}")
        print(f"| {f} | " + " | ".join(cells) + " |")

    print("\n## register difference vs sm_89 (same case/family/cfg): median / min / max")
    for f in fams:
        for a in ["sm_80", "sm_90", "sm_100", "sm_120"]:
            diffs = [d[a]["regs"] - d["sm_89"]["regs"] for k, d in by.items() if k[1] == f and a in d and "sm_89" in d]
            if diffs:
                diffs.sort()
                print(f"  {f:10s} {a}: median {diffs[len(diffs)//2]:+d}, min {diffs[0]:+d}, max {diffs[-1]:+d} (n={len(diffs)})")

    keys = [
        ("vel2d_o3_cb5", "ra_adj", "nxg4 cb7 chk16 listmax1024"),
        ("vel2d_o3_cb5", "ra_adj", "nxg2 cb4 chk16 listmax1024"),
        ("vel2d_o3_cb10", "ra_adj", "nxg4 cb7 chk16 listmax1024"),
        ("vel2d_o3_cb5", "kl_adj", "nxg4 cb7 chk16"),
        ("vel2d_o3_cb5", "vt_fwd", "4x4x16 smax48 nwc4 nws1 vxpad0"),
        ("vel2d_o3_cb10", "vt_fwd", "4x4x16 smax48 nwc7 nws1 vxpad0"),
        ("static_o0_10x10", "vs_fwd", "4x4x16 smax64 nwc7 nws1 vxpad0"),
        ("static_o0_48x1", "ra_adj", "nxg2 cb3 chk16 listmax1024"),
        ("static_o0_48x1", "ra_adj", "nxg4 cb3 chk16 listmax1024"),
        ("static_o2_cb5", "ra_adj", "nxg4 cb7 chk16 listmax1024"),
        ("static_o2_cb5", "vt_fwd", "4x4x16 smax48 nwc4 nws1 vxpad0"),
    ]
    print("\n## key shapes: regs / warps (limiter) [spill]")
    print("| case | family | cfg | " + " | ".join(ARCHS) + " |")
    for k in keys:
        d = by.get(k, {})
        cells = []
        for a in ARCHS:
            r = d.get(a)
            if r is None:
                cells.append("-")
            else:
                s = f"{r['regs']} / {r['warps_sm']} ({r['limiter']})"
                if r["spill_st"] or r["spill_ld"]:
                    s += f" S{r['spill_st']}/{r['spill_ld']}"
                cells.append(s)
        print(f"| {k[0]} | {k[1]} | {k[2]} | " + " | ".join(cells) + " |")

    # what-if comparisons
    for p in sys.argv[2:]:
        w = {key(r) + (r["arch"],): r for r in load(p) if r["error"] is None}
        b = {key(r) + (r["arch"],): r for r in ok}
        print(f"\n## what-if {p} vs baseline: per family x arch: n, spills base->new, warps up / down / same, regs median change")
        agg = collections.defaultdict(lambda: collections.Counter())
        rd = collections.defaultdict(list)
        for k, r in w.items():
            if k not in b:
                continue
            o = b[k]
            c = agg[(k[1], k[3])]
            c["n"] += 1
            c["sp0"] += bool(o["spill_st"] or o["spill_ld"])
            c["sp1"] += bool(r["spill_st"] or r["spill_ld"])
            c["up"] += r["warps_sm"] > o["warps_sm"]
            c["down"] += r["warps_sm"] < o["warps_sm"]
            rd[(k[1], k[3])].append(r["regs"] - o["regs"])
        for (f, a), c in sorted(agg.items()):
            d = sorted(rd[(f, a)])
            print(f"  {f:10s} {a}: n {c['n']}, spills {c['sp0']}->{c['sp1']}, warps up {c['up']} down {c['down']}, regs median {d[len(d)//2]:+d}")


if __name__ == "__main__":
    main()
