#!/usr/bin/env python3
"""§17 tables from the opbench logs of research/experiments/knot_layout (kl3_*.log): per case the autotune's knot-0
adjoint kernel times and the --kl-compare rows (previous path / skip / kl kernel)."""
import re, sys, glob, os
D = os.path.dirname(os.path.abspath(__file__))
for f in sorted(glob.glob(os.path.join(D, sys.argv[1] if len(sys.argv) > 1 else "kl3_*.log"))):
    s = open(f).read()
    case = os.path.basename(f)[:-4]
    m = re.search(r"autotune knot-path layout .*?all items ([\d.]+) ms, skip empty tiles ([\d.]+) ms, whole-tile kl kernel ([\w.]+) ms \((.*?)\) -> (.*)", s)
    a = re.search(r"autotune adjoint kernels .*?tile ([\d.]+) ms, ra ([\d.]+) ms", s)
    v = re.search(r"autotune forward kernels .*?tile ([\d.]+) ms, vs ([\d.]+) ms", s)
    print(f"{case}: knot-0 kernels: adjoint tile {a and a.group(1)} / ra {a and a.group(2)} ms; forward tile {v and v.group(1)} / vt {v and v.group(2)} ms")
    if m:
        print(f"   knot-0 adjoint incl. split zeroing: all items {m.group(1)}, skip {m.group(2)}, kl kernel {m.group(3)} ({m.group(4)}) -> {m.group(5)}")
    for r in re.finditer(r"(kl off|skip|kl kern) +M +(\d+): forward +([\d.]+) ms, adjoint +([\d.]+) ms, gradient \(forward \+ adjoint\) +([\d.]+) ms", s):
        print(f"   {r.group(1):8s} M {r.group(2)}: forward {r.group(3)}, adjoint {r.group(4)}, gradient {r.group(5)}")
    ff = re.search(r"FFT column: .*? — forward ([\d.]+) ms × (\d+), inverse ([\d.]+) ms × (\d+)", s)
    if ff:
        print(f"   FFT per call: forward {float(ff.group(1)) * int(ff.group(2)):.0f} ms, inverse {float(ff.group(3)) * int(ff.group(4)):.0f} ms")
