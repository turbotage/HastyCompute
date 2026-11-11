"""Recon-side refinement on MC draws: rerun the estimate of a draw (same seeds as mc.py) and project the residual onto
extended refit directions of the *model* (the nonparametric estimate): per-axis delays + field deg 1 + per-encode
(the `refined` level) plus 12 cross-term directions (6 pairs x tau 0.5 / 3 ms), 6 B0-term directions (3 axes x tau
0.3 / 3 ms) and, per resonance found by the parametric fit, f0 and peak height (multiplicative correction on the
table). Linearised one-step refit (no iteration). -> <OUT>/mcrefit_<tag>.jsonl

    mc_refit.py --tag R_rand_T200 --ndraw 16 --workers 5
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gmodel as gm, thinslice as ts, truths, v2seq, mc

OUT = mc.OUT
RC = []


class ResRatio(gm.Resp):
    """(1 + g' BP(f0')) / (1 + g BP(f0)): moves one resonance of a table response"""

    def __init__(self, f0, z, g, f1, g1):
        self.a = gm.MultiRes(gm.Ideal(), [(f1, z, g1)])
        self.b = gm.MultiRes(gm.Ideal(), [(f0, z, g)])

    def at(self, f):
        return self.a.at(f) / self.b.at(f)

    def key(self):
        return repr((self.a.res, self.b.res))


def work(job):
    tag, d, opt = job
    import rcalc
    if not RC:
        RC.append(rcalc.RCalc())
    rc = RC[0]
    v, rots = rc.v, rc.rots
    t0 = time.time()
    rng = np.random.default_rng(1000 + d)
    tadc = opt.get('tadc') or 80e-3
    prot = ts.Protocol(snr0=opt['snr'], n_avg=opt['navg'], thk=opt['thk'], t_adc=tadc, t_frame=tadc + 0.02)
    truth = truths.draw(np.random.default_rng(5000 + d))
    nu = ts.Nuisance(rng, prot)
    Y, sig = ts.simulate(truth, prot, nu, rng)
    RK, RP = mc.ref_predictions(prot, gm.clinical_example())
    ep, en, diag = mc.estimate(Y, sig, prot, RK, RP)
    model = en
    K, P = v2seq.trajectories(v, truth, rots, ref=model)
    q = rc.q_of(K, P)
    K0, P0 = v2seq.trajectories(v, model, rots, ref=gm.ideal())

    def dirof(m2, step):
        K1, P1 = v2seq.trajectories(v, m2, rots, ref=gm.ideal())
        return rc.q_of(K1 - K0, P1 - P0) / step
    dx = []
    for a in range(3):
        for b in range(3):
            if a != b:
                for tau in (0.5e-3, 3e-3):
                    dx.append(dirof(model.copy_with(cross=model.cross + [(a, b, gm.HighPass(tau, 1e-4))]), 1e-4))
        for tau in (0.3e-3, 3e-3):
            b0 = list(model.b0); b0[a] = gm.Add(b0[a], gm.HighPass(tau, 1e-6))
            dx.append(dirof(model.copy_with(b0=b0), 1e-6))
    dres = []
    for a in range(3):
        th, nres = np.array(diag[a]['th']), diag[a]['nres']
        for j in range(nres):
            f0, z, g = th[7 + 3 * j], np.exp(th[8 + 3 * j]), th[9 + 3 * j]
            for f1, g1, step in ((f0 + 0.05, g, 0.05), (f0, g + 1e-3, 1e-3)):
                ax = list(model.axis); ax[a] = gm.Mul(model.axis[a], ResRatio(f0, z, g, f1, g1))
                dres.append(dirof(model.copy_with(axis=ax), step))
    base = rc.dirs_enc() + rc.dirs_field() + rc.dirs_delay()
    out = dict(draw=d, tag=tag, sec=0.0, n_res=len(dres) // 2)
    out['refined'] = rc.R(rc.project(q, base))
    out['refined_xb'] = rc.R(rc.project(q, base + dx))
    out['refined_xbr'] = rc.R(rc.project(q, base + dx + dres)) if dres else out['refined_xb']
    out['loc_refined_xbr'] = rc.R(rc.project(q, base + dx + dres, True), True) if dres else rc.R(rc.project(q, base + dx, True), True)
    out['sec'] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='R_rand_T200')
    ap.add_argument('--ndraw', type=int, default=16)
    ap.add_argument('--workers', type=int, default=5)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(OUT, f'mc_{a.tag}.jsonl'))]
    opt = recs[0]['opt']
    fn = os.path.join(OUT, f'mcrefit_{a.tag}.jsonl')
    done = {json.loads(l)['draw'] for l in open(fn)} if os.path.exists(fn) else set()
    jobs = [(a.tag, d, opt) for d in range(a.ndraw) if d not in done]
    import multiprocessing as mp
    with mp.get_context('fork').Pool(a.workers) as pool:
        for r in pool.imap_unordered(work, jobs):
            with open(fn, 'a') as fo:
                fo.write(json.dumps(r) + '\n')
            print(time.strftime('%H:%M:%S'), f"draw {r['draw']}: refined {r['refined']:.3g} +cross/B0 {r['refined_xb']:.3g} +res {r['refined_xbr']:.3g} ({r['n_res']} res, {r['sec']:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
