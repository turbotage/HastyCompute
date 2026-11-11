"""sweep.py <ph0001 dir> <module dir> <case: water|fat> <variant> <out.jsonl> <cands,comma> <Ls,comma>"""
import sys, time, json, os, numpy as np
sys.path.insert(0, sys.argv[2])
import segeval as se
D, _, kind, variant, out = sys.argv[1:6]
if ':' in sys.argv[6]:  # cand:L/L/..;cand:L/..
    plan = [(c.split(':')[0], [int(x) for x in c.split(':')[1].split('/')]) for c in sys.argv[6].split(';')]
else:
    plan = [(c, [int(x) for x in sys.argv[7].split(',')]) for c in sys.argv[6].split(',')]
done = set()
if os.path.exists(out):
    for line in open(out):
        r = json.loads(line); done.add((r['case'], r['variant'], r['cand'], r['L']))
case = se.Case(D, conc=True, fat=(kind == 'fat'), nr=48)
se.setup_v3(case)
# field variants
p = case.pos
w = case.wEs
def wfit(deg):
    cols = [np.ones(case.nf), p[:, 0], p[:, 1], p[:, 2]]
    if deg >= 2:
        cols += [p[:, 0] ** 2, p[:, 1] ** 2, p[:, 2] ** 2, p[:, 0] * p[:, 1], p[:, 0] * p[:, 2], p[:, 1] * p[:, 2]]
    A = np.stack(cols, 1)
    sw = np.sqrt(w / w.mean())
    c, *_ = np.linalg.lstsq(A[::3] * sw[::3, None], case.om[::3] * sw[::3], rcond=None)
    return A @ c, c
lo = np.percentile(p[case.wEs > np.median(case.wEs)], 1, axis=0); hi = np.percentile(p[case.wEs > np.median(case.wEs)], 99, axis=0)
ctr = 0.5 * (lo + hi); half = 0.5 * (hi - lo)
u = (p - ctr) / half  # ~[-1, 1] over the tissue
parts = variant.split('+')
for part in parts:
    if part == 'raw':
        pass
    elif part == 'shimlin':   # linear shim residual: +-150 Hz along y, +-100 Hz along z over the tissue extent
        case.om = case.om + se.TWO_PI * (150 * u[:, 1] + 100 * u[:, 2])
    elif part == 'shimquad':  # Z2-like quadratic residual, about +-150 Hz over the tissue
        case.om = case.om + se.TWO_PI * 150 * (2 * u[:, 2] ** 2 - u[:, 0] ** 2 - u[:, 1] ** 2) / 2
    elif part == 'shimquadn':  # the same Z2-like residual with the opposite sign (widens the ph0001 field)
        case.om = case.om - se.TWO_PI * 150 * (2 * u[:, 2] ** 2 - u[:, 0] ** 2 - u[:, 1] ** 2) / 2
    elif part == 'abs1':      # degree-1 part removed (per-sample k-shift + phase, exact)
        f, c = wfit(1); case.om = case.om - f; print('abs1 removed coeffs (Hz, Hz/m):', np.round(c / se.TWO_PI, 1))
    elif part == 'abs2':      # degree-2 part removed entirely (ideal, NOT exactly absorbable by the Taylor path)
        f, c = wfit(2); case.om = case.om - f
    elif part == 'abs2t':     # linear part exact, quadratic part per sample via Taylor-2 around the readout centre
        f, c = wfit(2); f1 = c[0] + p @ c[1:4]; case.om = case.om - f1
        case.taylorQ = f - f1; case.tref = 0.5 * (case.tau[0] + case.tau[-1])
        print('abs2t: max |Q| (Hz)', np.abs(case.taylorQ).max() / se.TWO_PI, 'max Taylor phase (rad)', (np.abs(case.taylorQ) * 0.5 * (case.tau[-1] - case.tau[0])).max())
    else:
        raise ValueError(part)
hz = case.om / se.TWO_PI
pc = np.percentile(hz, [0, 1, 5, 95, 99, 100])
print(f'variant {variant}: field Hz pct 0/1/5/95/99/100 {np.round(pc, 0)}', flush=True)
Y = case.signal_near()
r = Y / case.Emeas[case.near]
scale = float(np.median(r[case.kap[case.near] < 30]))
case.scale_model = scale
print('scale model/meas', scale, flush=True)
fams = {}
def family(name):
    if name not in fams:
        cfg = dict(svd=('count', 'none', 'uniform'), svdw=('count', 'data', 'uniform'), svdw_mass=('mass', 'data', 'uniform'),
                   svdn=('count', 'snr', 'rust'), svdn_e=('energy', 'snr', 'rust'), svdn_m=('mass', 'snr', 'rust'), svdu=('count', 'none', 'rust'), svdm=('count', 'mix', 'rust'), svdm_e=('energy', 'mix', 'rust'), svdm_m=('mass', 'mix', 'rust'))[name]
        t = time.time(); fams[name] = se.SvdFamily(case, vw=cfg[0], tw=cfg[1], grid=cfg[2]); print(f'  gram {name} {time.time()-t:.0f}s', flush=True)
    return fams[name]
for cand, Ls in plan:
    for L in Ls:
        if (kind, variant, cand, L) in done:
            continue
        t = time.time()
        extra = {}
        if cand == 'fixed':
            b = se.basis_fixed(case, L)
        elif cand == 'cheb':
            b = se.basis_fixed(case, L, nodes='cheb')
        elif cand == 'fixed_e':
            b = se.basis_fixed(case, L, vw='energy')
        else:
            fam = family(cand); b = fam.basis(L); extra['tail'] = float(fam.tail(L))
        res = se.evaluate3(case, b)
        ig = se.integrate2(case, res, scale)
        rec = dict(case=kind, variant=variant, cand=cand, L=L, **ig, P_prop=res['P_prop'], P_snr1=res['P_snr1'], **extra, secs=time.time() - t)
        print(json.dumps(rec), flush=True)
        with open(out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
