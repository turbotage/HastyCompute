"""Deterministic sensitivity of the v2 readout's R to GIRF model errors (no measurement noise): omission per chain,
single-class relative errors (girf.rs perturbations), absolute errors per class, unmeasured cross terms, thermal
mismatch, nonlinearity, long eddy terms, and the Fisher information of recon-side refinement parameters.
Writes <OUT>/sens.json (resumable per entry).

    sens.py [--parts omission,classes,absolute,thermal,fisher] [--nrand 3]
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gmodel as gm, truths, v2seq, rcalc

OUT = rcalc.OUT


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='omission,classes,absolute,thermal,refine_ext,fisher,res_sweep')
    ap.add_argument('--nrand', type=int, default=3)
    a = ap.parse_args()
    fn = os.path.join(OUT, 'sens.json')
    res = json.load(open(fn)) if os.path.exists(fn) else {}

    def save():
        json.dump(res, open(fn, 'w'), indent=1)
    rc = rcalc.RCalc()
    v, rots = rc.v, rc.rots

    def Rq(truth, model):
        K, P = v2seq.trajectories(v, truth, rots, ref=model)
        q = rc.q_of(K, P)
        out = rc.Rset(q)
        out['dk_rms_cpm'] = float(np.sqrt((K[:, v.recon] ** 2).sum(-1).mean()) / 2 / np.pi)
        out['dk_echo_cpm'] = float(np.linalg.norm(K[:, 0], axis=1).mean() / 2 / np.pi)
        out['dphi_rms'] = float(np.sqrt((P[:, v.recon] ** 2).mean()))
        return out

    def put(sec, key, f):
        d = res.setdefault(sec, {})
        if key in d:
            return
        t0 = time.time()
        d[key] = f()
        save()
        r = d[key]
        log(f'{sec}/{key}: raw {r["raw"]:.3g} aff {r["affine"]:.3g} enc {r["enc"]:.3g} field {r["field"]:.3g} ref {r["refined"]:.3g} | loc {r["loc_raw"]:.3g}/{r["loc_refined"]:.3g} ({time.time() - t0:.1f}s)')

    chains = [('clinical', gm.clinical_example())] + gm.variants()
    rand = [(f'rand{j}', truths.draw(np.random.default_rng(5000 + j), f'rand{j}')) for j in range(a.nrand)]
    for _, g in chains:
        g.tail = max(g.tail, 30e-3)
    parts = a.parts.split(',')

    if 'omission' in parts:
        for name, g in chains + rand:
            put('omission', name, lambda: Rq(g, gm.ideal()))
        # the budget's T10 setting: readout spoke from rest (no encode, no previous TR) = girf_err.npy style
        def spoke_only(g):
            K = np.zeros((rc.nr, len(v.t), 3)); P = np.zeros((rc.nr, len(v.t)))
            for i in range(rc.nr):
                R = rots['R'][i, -1]
                nom = np.r_[v.base @ R.T, np.zeros((1, 3))]
                y, b = g.apply(nom, v.dt, ntail=0, nfft=1 << 14)
                k = gm.k_at(y, v.dt, -gm.PRE * v.dt, v.t, 0.0)
                K[i] = k - v.k_base @ R.T
                P[i] = gm.phase_at(b, v.dt, -gm.PRE * v.dt, v.t, 0.0)
            q = rc.q_of(K, P)
            o = rc.Rset(q); o['dk_rms_cpm'] = float(np.sqrt((K[:, v.recon] ** 2).sum(-1).mean()) / 2 / np.pi)
            o['dk_echo_cpm'] = 0.0; o['dphi_rms'] = float(np.sqrt((P[:, v.recon] ** 2).mean()))
            return o
        for name in ('clinical', 'resonance'):
            put('omission_spoke_from_rest', name, lambda: spoke_only(dict(chains + rand)[name]))
        # 5 % of the full-TR omission residual (R is linear in q for small q)
        for name in ('clinical', 'resonance', 'rand0'):
            g = dict(chains + rand)[name]
            def five(g=g):
                K, P = v2seq.trajectories(v, g, rots, ref=gm.ideal())
                q = 0.05 * rc.q_of(K, P)
                o = rc.Rset(q); o['dk_rms_cpm'] = 0.05 * float(np.sqrt((K[:, v.recon] ** 2).sum(-1).mean()) / 2 / np.pi)
                o['dk_echo_cpm'] = 0.05 * float(np.linalg.norm(K[:, 0], axis=1).mean() / 2 / np.pi); o['dphi_rms'] = 0.05 * float(np.sqrt((P[:, v.recon] ** 2).mean()))
                return o
            put('residual_5pct_full_tr', name, five)

    if 'classes' in parts:
        for name, g in [('resonance', dict(chains)['resonance'])] + rand:
            for cls in gm.CLASSES:
                for eps in (0.1, 0.01):
                    gp = gm.perturbed_class(g, cls, eps)
                    if gp.key() == g.key():
                        continue
                    put(f'classes_{name}', f'{cls}_{eps:g}', lambda: Rq(g, gp))

    if 'absolute' in parts:
        for name, g in [('resonance', dict(chains)['resonance'])] + rand:
            sec = f'absolute_{name}'
            put(sec, 'delay_10ns_all', lambda: Rq(g, gm.with_delay_offset(g, 10e-9)))
            put(sec, 'delay_10ns_x', lambda: Rq(g, gm.with_delay_offset(g, [10e-9, 0, 0])))
            put(sec, 'delay_10ns_y', lambda: Rq(g, gm.with_delay_offset(g, [0, 10e-9, 0])))
            put(sec, 'delay_10ns_z', lambda: Rq(g, gm.with_delay_offset(g, [0, 0, 10e-9])))
            put(sec, 'gain_1e-3_all', lambda: Rq(g, gm.with_gain(g, 1e-3)))
            put(sec, 'gain_1e-3_x', lambda: Rq(g, gm.with_gain(g, [1e-3, 0, 0])))
            put(sec, 'cross_omitted', lambda: Rq(g, gm.without(g, 'cross')))
            put(sec, 'b0_omitted', lambda: Rq(g, gm.without(g, 'b0')))
            put(sec, 'res_omitted', lambda: Rq(g, gm.without(g, 'res')))
            if g.nl:
                put(sec, 'nonlinearity_omitted', lambda: Rq(g, g.copy_with(nl=0.0)))
            # long eddy term (tau >= 10 ms) omitted, per axis (random truths)
            if isinstance(g.axis[0], gm.Mul):
                def nolong(g):
                    ax = []
                    for r in g.axis:
                        mr = r.parts[0]
                        b = mr.base
                        ax.append(gm.Mul(gm.MultiRes(gm.Param(b.delay, b.tau_lp, [e for e in b.eddy if e[1] < 10e-3]), mr.res), *r.parts[1:]))
                    return g.copy_with(axis=ax)
                put(sec, 'long_eddy_omitted', lambda: Rq(g, nolong(g)))
                def norip(g):
                    return g.copy_with(axis=[gm.Mul(*r.parts[:2]) for r in g.axis])
                put(sec, 'ripple_omitted', lambda: Rq(g, norip(g)))
                def nolp2(g):
                    return g.copy_with(axis=[gm.Mul(r.parts[0], r.parts[2]) for r in g.axis])
                put(sec, 'lp2_omitted', lambda: Rq(g, nolp2(g)))
                # resonance f0 +1 Hz, peak +0.1 % (absolute) on every resonance
                def resabs(g, df=0.0, dg=0.0, dz=1.0):
                    ax = []
                    for r in g.axis:
                        mr = r.parts[0]
                        ax.append(gm.Mul(gm.MultiRes(mr.base, [(f0 + df, z * dz, gg + dg) for f0, z, gg in mr.res]), *r.parts[1:]))
                    return g.copy_with(axis=ax)
                put(sec, 'res_f0_+1Hz', lambda: Rq(g, resabs(g, df=1.0)))
                put(sec, 'res_gain_+0.1pct', lambda: Rq(g, resabs(g, dg=1e-3)))
                put(sec, 'res_zeta_x1.1', lambda: Rq(g, resabs(g, dz=1.1)))
                # eddy amplitude +0.01 % absolute per term class (short / ms / long)
                def eddyabs(g, j, da):
                    ax = []
                    for r in g.axis:
                        mr = r.parts[0]
                        b = mr.base
                        ed = [(x + (da if i == j else 0), t) for i, (x, t) in enumerate(b.eddy)]
                        ax.append(gm.Mul(gm.MultiRes(gm.Param(b.delay, b.tau_lp, ed), mr.res), *r.parts[1:]))
                    return g.copy_with(axis=ax)
                for j, lab in enumerate(['short', 'ms', 'long']):
                    put(sec, f'eddy_{lab}_+0.01pct', lambda: Rq(g, eddyabs(g, j, 1e-4)))
                put(sec, 'b0_+1e-6m', lambda: Rq(g, g.copy_with(b0=[gm.Add(b, gm.HighPass(2e-3, 1e-6)) for b in g.b0])))

    if 'thermal' in parts:
        for name, g in [('resonance', dict(chains)['resonance'])] + rand[:2]:
            sec = f'thermal_{name}'
            for dT in (1.0, 5.0, 10.0):
                put(sec, f'dT_{dT:g}', lambda: Rq(truths.thermal(g, dT), g))
            for c in truths.THERMAL:
                put(sec, f'dT_5_only_{c}', lambda: Rq(truths.thermal(g, 5.0, only=c), g))

    if 'refine_ext' in parts:
        # recon-side refinement with more directions: derivatives of the model's played trajectory w.r.t. added
        # cross terms (6 pairs x tau 0.5 / 3 ms), B0 terms (3 axes x tau 0.3 / 3 ms), long eddies (3 axes x 10 / 30 ms)
        def numdirs(model, kind):
            K0, P0 = v2seq.trajectories(v, model, rots, ref=gm.ideal())
            d = []
            for a_ in range(3):
                for b_ in range(3):
                    for tau in ((0.5e-3, 3e-3) if kind == 'cross' and a_ != b_ else ()):
                        m2 = model.copy_with(cross=model.cross + [(a_, b_, gm.HighPass(tau, 1e-4))])
                        K1, P1 = v2seq.trajectories(v, m2, rots, ref=gm.ideal())
                        d.append(rc.q_of(K1 - K0, P1 - P0) / 1e-4)
                if kind == 'b0':
                    for tau in (0.3e-3, 3e-3):
                        b0 = list(model.b0); b0[a_] = gm.HighPass(tau, 1e-6) if b0[a_] is None else gm.Add(b0[a_], gm.HighPass(tau, 1e-6))
                        K1, P1 = v2seq.trajectories(v, model.copy_with(b0=b0), rots, ref=gm.ideal())
                        d.append(rc.q_of(K1 - K0, P1 - P0) / 1e-6)
                if kind == 'long':
                    for tau in (10e-3, 30e-3):
                        ax = list(model.axis); ax[a_] = gm.Mul(ax[a_], gm.Param(0.0, 0.0, [(1e-4, tau)]))
                        K1, P1 = v2seq.trajectories(v, model.copy_with(axis=ax), rots, ref=gm.ideal())
                        d.append(rc.q_of(K1 - K0, P1 - P0) / 1e-4)
            return d
        for name, g in [('resonance', dict(chains)['resonance'])] + rand:
            sec = f'refine_ext_{name}'
            for case, model, kind in [('cross_omitted', gm.without(g, 'cross'), 'cross'), ('b0_omitted', gm.without(g, 'b0'), 'b0')]:
                def f(model=model, kind=kind):
                    K, P = v2seq.trajectories(v, g, rots, ref=model)
                    q = rc.q_of(K, P)
                    extra = numdirs(model, kind)
                    o = rc.Rset(q, refine=rc.dirs_delay() + extra)
                    o['dk_rms_cpm'] = float(np.sqrt((K[:, v.recon] ** 2).sum(-1).mean()) / 2 / np.pi)
                    return o
                put(sec, case, f)

    if 'res_sweep' in parts:
        # one resonance (peak 5 %) on every axis at f0, on the clinical base chain; model errors: f0 +0.05 %, peak
        # +10 % relative, resonance omitted; R vs f0 (where is the v2 readout vulnerable?)
        base = gm.clinical_example(); base.tail = 0.15
        for zeta in (0.005, 0.03):
            for f0 in np.r_[np.arange(300, 1500, 100), np.arange(1500, 4001, 250)]:
                tr_ = base.copy_with(axis=[gm.Resonant(r, f0, zeta, 0.05) for r in base.axis])
                put(f'res_sweep_z{zeta:g}', f'{f0:.0f}_f0+0.05pct', lambda: Rq(tr_, tr_.copy_with(axis=[gm.Resonant(r, f0 * 1.0005, zeta, 0.05) for r in base.axis])))
                put(f'res_sweep_z{zeta:g}', f'{f0:.0f}_gain+10pct', lambda: Rq(tr_, tr_.copy_with(axis=[gm.Resonant(r, f0, zeta, 0.055) for r in base.axis])))

    if 'refine_res' in parts:
        # recon-side refit of every resonance's f0 and peak height (linearised directions of the model's played
        # trajectory), on top of the 'refined' set; cases: f0 +1 Hz on every resonance (rand0-2), thermal drift 5 C
        def resdirs(model):
            K0, P0 = v2seq.trajectories(v, model, rots, ref=gm.ideal())
            d = []
            for a_ in range(3):
                r = model.axis[a_]
                inner = r.parts[0] if isinstance(r, gm.Mul) else r
                res = inner.res if isinstance(inner, gm.MultiRes) else [(inner.f0, inner.zeta, inner.gain)] if isinstance(inner, gm.Resonant) else []
                for j in range(len(res)):
                    for which, step in ((0, 0.05), (2, 1e-3)):
                        rr = [list(x) for x in res]; rr[j][which] += step
                        if isinstance(inner, gm.MultiRes):
                            new_inner = gm.MultiRes(inner.base, rr)
                        else:
                            new_inner = gm.Resonant(inner.base, rr[0][0], rr[0][1], rr[0][2])
                        ax = list(model.axis)
                        ax[a_] = gm.Mul(new_inner, *r.parts[1:]) if isinstance(r, gm.Mul) else new_inner
                        K1, P1 = v2seq.trajectories(v, model.copy_with(axis=ax), rots, ref=gm.ideal())
                        d.append(rc.q_of(K1 - K0, P1 - P0) / step)
            return d

        def shift_f0(g, df):
            ax = []
            for r in g.axis:
                mr = r.parts[0]
                ax.append(gm.Mul(gm.MultiRes(mr.base, [(f0 + df, z, gg) for f0, z, gg in mr.res]), *r.parts[1:]))
            return g.copy_with(axis=ax)
        cases = [(f'rand{j}_f0+1Hz', shift_f0(g, 1.0), g) for j, (_, g) in enumerate(rand)]
        cases += [('resonance_thermal_dT5', truths.thermal(dict(chains)['resonance'], 5.0), dict(chains)['resonance'])]
        cases += [(f'rand{j}_thermal_dT5', truths.thermal(g, 5.0), g) for j, (_, g) in enumerate(rand[:2])]
        for case, truth, model in cases:
            def f(truth=truth, model=model):
                K, P = v2seq.trajectories(v, truth, rots, ref=model)
                q = rc.q_of(K, P)
                o = rc.Rset(q, refine=rc.dirs_delay() + resdirs(model))
                o['dk_rms_cpm'] = float(np.sqrt((K[:, v.recon] ** 2).sum(-1).mean()) / 2 / np.pi)
                return o
            put('refine_res', case, f)

    if 'fisher' in parts:
        # CRLB of per-axis delays (and a global k0 rate) from the whole 10-min scan, image known up to the
        # x-absorbable / per-encode / field directions (Schur complement); recon samples and navigator samples
        d_del = rc.dirs_delay()
        nuis = rc.dirs_enc() + rc.dirs_field()
        out = {}
        for nav in (False, True):
            F = rc.fisher(d_del + nuis, nav=nav)
            p = len(d_del)
            A, B, C = F[:p, :p], F[:p, p:], F[p:, p:]
            S = A - B @ np.linalg.pinv(C, rcond=1e-12) @ B.T
            crlb = np.sqrt(np.diag(np.linalg.inv(S)))
            crlb_known = np.sqrt(np.diag(np.linalg.inv(A)))
            out['nav' if nav else 'recon'] = dict(delay_sd_ns=(crlb * 1e9).tolist(), delay_sd_ns_no_nuisance=(crlb_known * 1e9).tolist())
        # per-TR navigator: sd of a k offset (3) and a global phase from the navigator samples of ONE readout,
        # and from its reconstruction samples (image known)
        for nav in (True, False):
            w = rc.wnav if nav else rc.ww
            Fr = 2.0 / rc.sigma ** 2 * np.einsum('rn,rnab->rab', w * rc.nr, rc.G)  # per readout, 4 x 4
            sds = np.array([np.sqrt(np.diag(np.linalg.inv(F + 1e-30 * np.eye(4)))) for F in Fr])
            out['per_tr_' + ('nav' if nav else 'recon')] = dict(k_offset_sd_cpm=(np.median(sds[:, :3], 0) / 2 / np.pi).tolist(), phase_sd_mrad=float(np.median(sds[:, 3]) * 1e3))
        res['fisher'] = out
        save()
        log('fisher', json.dumps(out))
    log('done')


if __name__ == '__main__':
    main()
