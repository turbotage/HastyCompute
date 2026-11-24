"""Run 1/2 table: held-out curve and pc RMSE / median every 5 iterations, s/it, first iteration reaching a target.
usage: python analyze_guard.py <run dir> [<run dir> ...]   (target: held-out 0.0615, pc RMSE 0.242 = zmshr1_120 at it 120)"""
import sys, glob, os, re

T_HO, T_PC = 0.0615, 0.242


def rows(p):
    with open(p) as fh:
        h = fh.readline().strip().split(",")
        return [dict(zip(h, l.strip().split(","))) for l in fh if l.strip()]


for d in sys.argv[1:]:
    ho = sorted(glob.glob(os.path.join(d, "holdout_*.csv")))
    pc = sorted(glob.glob(os.path.join(d, "pcscore_*.csv")))
    tm = sorted(glob.glob(os.path.join(d, "timing_*.csv")))
    if not ho:
        print(d, "no holdout csv")
        continue
    h = {int(r["it"]): (float(r["rel"]), float(r["seconds"])) for r in rows(ho[-1])}
    p = {}
    for r in rows(pc[-1]) if pc else []:
        p[int(r["it"])] = (float(r["rmse"]), float(r["median"]))
    t = rows(tm[-1]) if tm else []
    secs = [sum(float(r[k]) for k in ("gradient", "step", "prior", "report")) for r in t]
    log = open(os.path.join(d, "log.txt")).read()
    rej = len(re.findall(r"iterate rejected|last iterate is rejected", log))
    warm = re.findall(r"warm-full.*?\(([\d.]+)s\)", log)
    cal = re.findall(r"mean 1 − \|⟨ŝ, s⟩\| ([\d.e+-]+)", log)
    setup = re.findall(r"lambda_max\(N\) = .*?setup (\d+)s", log)
    print(f"## {os.path.basename(d)}: {len(secs)} iterations, {sum(secs)/max(len(secs),1):.1f} s/it, guard rejections {rej}, map error {cal[-1] if cal else '-'}, setup {setup[-1] if setup else '-'} s")
    print("| it | held-out | s (solve) | pc RMSE | pc median |")
    print("|---|---|---|---|---|")
    for k in sorted(h):
        pr = p.get(k)
        print(f"| {k} | {h[k][0]:.5f} | {h[k][1]:.0f} | {pr[0]:.3f} | {pr[1]:.3f} |" if pr else f"| {k} | {h[k][0]:.5f} | {h[k][1]:.0f} | - | - |")
    a = [k for k in sorted(h) if h[k][0] <= T_HO]
    b = [k for k in sorted(p) if p[k][0] <= T_PC]
    print(f"first it with held-out ≤ {T_HO}: {a[0] if a else 'not reached'}" + (f" ({h[a[0]][1]:.0f} s)" if a else "") + f"; first it with pc RMSE ≤ {T_PC}: {b[0] if b else 'not reached'}")
    print()
