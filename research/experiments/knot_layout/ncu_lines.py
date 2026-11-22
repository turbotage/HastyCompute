import csv, sys, collections
rows = list(csv.reader(open(sys.argv[1])))
hdr = None
for i, r in enumerate(rows):
    if len(r) > 4 and r[0] == "Line No":
        hdr = r; start = i + 1; break
col = {n: k for k, n in enumerate(hdr)}
si = hdr.index("Warp Stall Sampling (All Samples)")
stall_cols = [k for k, n in enumerate(hdr) if n.startswith("stall_") and "Not Issued" not in n]
lines = []
for r in rows[start:]:
    if r and r[0] not in ("", "-"):
        try:
            lines.append((int(r[0]), r[1], int(r[si]) if r[si] not in ("-", "") else 0, {hdr[k]: int(r[k]) for k in stall_cols if r[k] not in ("-", "")}))
        except ValueError:
            pass
tot = sum(l[2] for l in lines)
print("total samples", tot)
# region aggregation by line ranges given as name:lo-hi
regions = [a.split(":") for a in sys.argv[2:]]
agg = collections.OrderedDict()
for name, rng in regions:
    lo, hi = map(int, rng.split("-"))
    s = sum(l[2] for l in lines if lo <= l[0] <= hi)
    st = collections.Counter()
    for l in lines:
        if lo <= l[0] <= hi:
            st.update(l[3])
    top = ", ".join(f"{k[6:]} {v/ max(s,1)*100:.0f}%" for k, v in st.most_common(4))
    print(f"{name:12s} {s/tot*100:5.1f} %   [{top}]")
print("top lines:")
for l in sorted(lines, key=lambda l: -l[2])[:25]:
    print(f"{l[0]:5d} {l[2]/tot*100:5.1f}%  {l[1].strip()[:110]}")
