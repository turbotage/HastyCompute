"""Markdown tables for research/precond_estimator.md §6 from the --pp-study summaries (pp_summary.txt) of the campaign
run directories. Usage: summarize.py <run dir> [<run dir> ..]. Reads text only (no arrays)."""
import re
import sys

PAT = re.compile(
    r"pp (?P<name>\S+) spec (?P<spec>\S+) mu_rel (?P<mu>\S+) band (?P<band>\w+): iters (?P<it>\d+) final rel (?P<rel>\S+) "
    r"conv \[(?P<conv>[^\]]*)\]; it to 1% / 0.1% of x̂ (?P<i1>\S+) / (?P<i01>\S+); vs x̂ end (?P<eref>\S+); "
    r"vs x★ end all (?P<all>\S+) body (?P<body>\S+) blood (?P<blood>\S+) air (?P<air>\S+) corr (?P<corr>\S+); "
    r"best all (?P<best>\S+) at it (?P<bit>\d+); kappa_ritz (?P<kap>\S+); (?:ritz extremes (?P<tmax>\S+) / (?P<tmin>\S+) \(ratio (?P<tr>\S+)\); )?(?:x̂-tol it/s 1e-2 (?P<t2>\S+) 1e-3 (?P<t3>\S+) 1e-4 (?P<t4>\S+); setup (?P<setup>\S+)s solve (?P<solve>\S+)s; )?ms per P (?P<msp>\S+), per Q (?P<msq>\S+);")


def opt(s):
    m = re.match(r"Some\((\d+)\)", s)
    return m.group(1) if m else "–"


for d in sys.argv[1:]:
    print(f"\n### {d}\n")
    print("| μ_rel | band | run | preconditioner | iters (conv) | to 1 % / 0.1 % of x̂ | ‖x − x̂‖/‖x̂‖ end | vs x★ end (best @it) | corr PD | κ_eff (Ritz) | θ_max/θ_min (Ritz) | it/s to 1e-2, 1e-3, 1e-4 of x̂ | setup s | ms per P |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for line in open(f"{d}/pp_summary.txt", encoding="utf-8"):
        m = PAT.search(line)
        if not m:
            continue
        g = m.groupdict()
        conv = opt(g["conv"].split(",")[0].strip())
        print(f"| {g['mu']} | {g['band']} | {g['name']} | {g['spec']} | {g['it']} ({conv}) | {opt(g['i1'])} / {opt(g['i01'])} | {g['eref']} | "
              f"{g['all']} ({g['best']} @{g['bit']}) | {g['corr']} | {g['kap']} | {g['tr'] or '–'} | {g['t2'] or '–'}, {g['t3'] or '–'}, {g['t4'] or '–'} | {g['setup'] or '–'} | {g['msp']} |")
