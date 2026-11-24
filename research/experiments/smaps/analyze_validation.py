"""Summarise the coil-map validation arms (run_validation.sh): map error over the support (diagnostic, truth),
per-bin maps vs pooled, held-out residual per report (the acceptance number). Usage: python analyze_validation.py"""
import os, re, glob
R = os.path.join(os.path.dirname(__file__), "../../../data/hasty/phantom/factory_arms/ph0001/smaps_val")
for d in sorted(glob.glob(os.path.join(R, "*"))):
    log = os.path.join(d, "log.txt")
    if not os.path.exists(log):
        continue
    L = open(log, errors="replace").read().splitlines()
    name = os.path.basename(d)
    sup = [l for l in L if l.startswith("coil maps vs the true maps over the")]
    fit = [l for l in L if l.startswith("--smaps-fit")]
    rb = [l for l in L if l.startswith("coil maps, respiratory bin")]
    ho = [(int(m.group(2)), float(m.group(1))) for l in L for m in [re.search(r"= ([0-9.]+) at it (\d+)", l)] if "held-out residual" in l and m]
    print(f"== {name}  ({'DONE' if os.path.exists(os.path.join(d, 'DONE')) else 'not done'})")
    for l in sup + fit + rb:
        print("   " + l[:220])
    if ho:
        print("   held-out ‖E_h x − y_h‖/‖y_h‖: " + ", ".join(f"it {i}: {v:.5f}" for i, v in ho))
        print(f"   min {min(v for _, v in ho):.5f} at it {min(ho, key=lambda p: p[1])[0]}")
