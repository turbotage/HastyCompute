"""EPG / Ernst signal ratio of single-flip RF-spoiled steady states, and the T1 error of a classic 2-angle VFA (B1 known)."""
import subprocess, csv, io
import numpy as np
BIN = "../../../rustlib/target/epg-check/release/t1bias"
TIS = {"WM": (0.83, 0.07), "GM": (1.33, 0.095), "blood": (1.75, 0.15), "CSF": (4.2, 1.5)}

def steady(flip, tr, secs=8):
    out = subprocess.run([BIN, f"{flip}:{tr}:{secs}", "0"], capture_output=True, text=True, check=True).stdout
    d = {}
    for r in csv.DictReader(io.StringIO(out)):
        if r["b1"] != "1": continue
        d[(r["tissue"], r["inc"])] = np.hypot(float(r["re"]), float(r["im"]))  # last row of a (tissue,inc) wins
    return d

def ernst(a, tr, t1):
    E = np.exp(-tr * 1e-3 / t1); a = np.deg2rad(a)
    return np.sin(a) * (1 - E) / (1 - E * np.cos(a))

for tr in (11, 20):
    print(f"\nTR {tr} ms: EPG/Ernst steady-state ratio (TE 0)")
    print("flip   " + "  ".join(f"{t}/{i}" for t in ("WM", "GM", "blood") for i in ("117", "150")))
    S = {a: steady(a, tr) for a in (3, 6, 10, 15, 20, 30)}
    for a in S:
        print(f"{a:4d}   " + "  ".join(f"{S[a][(t, i)] / ernst(a, tr, TIS[t][0]):6.3f}" for t in ("WM", "GM", "blood") for i in ("117", "150")))
    # classic 2-angle VFA (3 deg + 15 deg or 6 + 20), linearised DESPOT1: S/sin a = E1 S/tan a + M0 (1-E1)
    for a1, a2 in ((3, 15), (6, 20), (3, 30)):
        s = []
        for t in ("WM", "GM", "blood"):
            for i in ("117", "150"):
                y = np.array([S[a][(t, i)] / np.sin(np.deg2rad(a)) for a in (a1, a2)] if a1 in S and a2 in S else [np.nan, np.nan])
                x = np.array([S[a][(t, i)] / np.tan(np.deg2rad(a)) for a in (a1, a2)] if a1 in S and a2 in S else [np.nan, np.nan])
                slope = (y[1] - y[0]) / (x[1] - x[0]); t1 = -tr * 1e-3 / np.log(slope)
                s.append(f"{100 * (t1 / TIS[t][0] - 1):7.1f}")
        print(f"2-angle VFA {a1}/{a2} deg, B1 known, T1 error %: " + "  ".join(s))
