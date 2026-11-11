"""Handling of the in-readout flow displacement by a lagged known warp of the moving (blood) voxels inside the time
segments, against the exact source-level truth (1 mm blood sources, own velocity, own field and T2*).

Model for a moving 2 mm voxel v (band-limited blood image x of the frame, v_hat = amplitude-weighted source velocity,
normalised-convolution extended to the ringing neighbours):
    exact time:  x_v e^{-i k.(r_v + d_v(tau))} rho_s(k, frac d_v(tau)) e^{-i wbar_v tau - R2*_b tau - i w.m(r_v)}
    segmented:   x_v sum_l a'_l(tau) D'_l(v) e^{-i k.(r_v + d_v(tau_l))} rho_s(k, frac d_v(tau_l)) e^{-R2*_b tau - i dw.m}
d_v(tau) = v_hat t1(tau) (+ 1/2 a_hat t2(tau)); rho_s = the push stencil's k-response relative to the exact shift
(separable, = 1 for the ideal off-grid shift); a', D' = the svdn basis of the object in the QDEIM cardinal form
(nodes tau_l, D'_l ~ g(., tau_l)). Truth: sum_s a_s e^{-i k.r_s} e^{-i m1.v_s} e^{-i w_s tau - tau/T2*_s - i w.m(r_s)}.
Reported: R of (model - truth) with motion, minus nothing; the static floor R (both without motion); per region.

    warp_eval.py [--designs v2,current] [--phase systole|diastole] [--nr 32]
"""
import json, os, sys, time, argparse
import numpy as np, torch
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(__file__))
import pb, flow, seg, truth_image

ap = argparse.ArgumentParser()
ap.add_argument('--designs', default='v2,current')
ap.add_argument('--phases', default='systole,diastole')
ap.add_argument('--nr', type=int, default=32)
ap.add_argument('--tag', default='')
ap.add_argument('--light', action='store_true', help='fewer stencils / variants (cost)')
ap.add_argument('--shear-only', action='store_true', help='only the sheared-voxel (intravoxel velocity gradient) variants and their references')
ap.add_argument('--matched', action='store_true', help='truth sources carry the field and R2* of their model voxel (isolates the motion handling from the intravoxel field of the blood)')
a = ap.parse_args()
OUT = pb.OUT
PH = {'systole': 3, 'diastole': 12}
TRS = flow.TRS


def stencil_resp(name, theta, f):
    """push stencil response relative to the exact shift: e^{i theta f} sum_m w(m - f) e^{-i theta m}; theta = k_a D,
    f = fractional displacement in voxels (torch, broadcast)"""
    if name == 'ideal':
        return torch.ones_like(theta, dtype=torch.complex64)
    if name == 'nearest':
        m = torch.round(f)
        return torch.polar(torch.ones_like(theta), theta * (f - m)).to(torch.complex64)
    if name == 'trilinear':
        taps, ker = [0, 1], lambda d: torch.clamp(1 - d.abs(), min=0)
    elif name == 'cubic':  # Keys a = -0.5 (Catmull-Rom)
        def ker(d):
            d = d.abs()
            return torch.where(d < 1, 1.5 * d ** 3 - 2.5 * d ** 2 + 1, torch.where(d < 2, -0.5 * d ** 3 + 2.5 * d ** 2 - 4 * d + 2, torch.zeros_like(d)))
        taps = [-1, 0, 1, 2]
    elif name.startswith('lanczos'):
        A = int(name[7:])
        def ker(d):
            return torch.where(d.abs() < A, torch.sinc(d) * torch.sinc(d / A), torch.zeros_like(d))
        taps = list(range(-A + 1, A + 1))
    fl = torch.floor(f); fr = f - fl
    acc = torch.zeros_like(theta, dtype=torch.complex64)
    for m in taps:
        acc = acc + ker(m - fr).to(torch.complex64) * torch.polar(torch.ones_like(theta), -theta * (m - fr)).to(torch.complex64)
    return acc


def build_model_voxels(obj, ps, b):
    """2 mm band-limited blood image of frame b on the object voxels + v_hat, a_hat (normalised convolution)"""
    v = flow.sim()
    presc = v['presc']; c = np.array(presc['center']); slab = presc['slab']
    fr = v['blood'][b]
    pos = np.asarray(fr['pos'][:]).astype(np.float64)
    dims = obj.dims
    lat = (pos - c) / 0.0005 + 4 * np.array([d // 2 for d in dims])
    off = np.median(lat - np.rint(lat), axis=0)
    li = np.rint(lat - off).astype(np.int64)
    okf = np.all((li >= 0) & (li < 4 * np.array(dims)), axis=1)
    keep = ps.keep  # the PointSet's sources
    amp = ps.t_a.real.cpu().numpy().astype(np.float64)
    full = np.zeros(len(pos)); full[keep] = amp
    xb = truth_image.bandlimit(li[okf], full[okf], dims, off[2]).ravel(order='F')
    # normalised convolution of the source velocity / acceleration on the 2 mm grid
    nvox = dims[0] * dims[1] * dims[2]
    q = obj.idx[ps.vox]
    A = np.bincount(q, weights=amp, minlength=nvox)
    out = {}
    for nm, val in [('v', ps.v), ('a', ps.t_acc.cpu().numpy())]:
        num = [gaussian_filter(np.bincount(q, weights=amp * val[:, k], minlength=nvox).reshape(dims, order='F'), 0.7) for k in range(3)]
        den = gaussian_filter(A.reshape(dims, order='F'), 0.7)
        out[nm] = np.stack([(nn / np.maximum(den, 1e-12 * den.max())).ravel(order='F') for nn in num], 1)
    # voxels: blood support dilated by 2
    from scipy.ndimage import binary_dilation
    sup = binary_dilation((A > 0).reshape(dims, order='F'), iterations=2).ravel(order='F')
    sel = np.flatnonzero(sup[obj.idx])  # object voxel indices
    gq = obj.idx[sel]
    # Jacobian J_ab = dv_a / dx_b of v_hat (central differences on the 2 mm grid): the intravoxel velocity gradient
    J = np.stack([np.stack(np.gradient(out['v'][:, a].reshape(dims, order='F'), obj.res), -1).reshape(-1, 3, order='F')[gq] for a in range(3)], 1)
    return dict(sel=sel, x=xb[gq].astype(np.complex64), v=out['v'][gq], a=out['a'][gq], J=J, n_support=int((A > 0).sum()))


if __name__ == '__main__':
    torch.set_num_threads(6)
    obj0 = pb.Obj()
    results = {}
    fn = os.path.join(OUT, f'warp_eval{a.tag}.json')
    if os.path.exists(fn):
        results = json.load(open(fn))
    for dname in a.designs.split(','):
        obj = pb.use_truth(obj0, dname)
        des = pb.Design(dname)
        nsel = des.sample_select(kc=150.0, stride=12)
        n_, w_ = nsel
        te = des.te
        # the object's svdn basis (B0 + encode-mean concomitant, energy voxel weights, noise time weights) in cardinal form
        gl = json.load(open(os.path.join(OUT, f'global_{dname}.json')))
        n_sel = np.array(gl['energy']['n']); e_sel = np.array(gl['energy']['E'])
        E_all = np.exp(np.interp(np.arange(des.na), n_sel, np.log(np.maximum(e_sel, 1e-30))))
        wn = seg.noise_time_weights(E_all, des.kabs, obj.nc, obj.sigma)
        rmean = pb.scan_readouts(400, offset=1)['R']
        wbar = np.mean([des.conc_w(R) for R in rmean], 0)
        t_wbar = torch.tensor(wbar, dtype=torch.float64, device=pb.DEV)
        m4 = pb.monos(obj.t_pos)
        xs = seg.smooth_mag(obj, obj.x_tot)
        wv = xs.astype(np.float64) ** 2 * obj.cE; wv = wv / np.median(wv[wv > 0])
        def g_b0(t, sl):
            n = torch.clamp(torch.round((t - des.tau[0]) / (des.tau[1] - des.tau[0])).long(), 0, des.na - 1)
            return pb.polar(t[:, None] * obj.t_om[sl][None, :] + t_wbar[n] @ m4[sl].T)
        fam = seg.Family(obj, g_b0, des.tau, wn, wv)
        Lr = gl['T1_rule']['svdn_L_rule']
        bases = {}
        for L in sorted(set([Lr, Lr + 2])):
            b = fam.basis(L)
            tg = torch.tensor(des.tau, dtype=torch.float64, device=pb.DEV)
            nodes_t, cond, piv = seg.deim_nodes(b, tg)
            A = b.interp(tg)  # na x L
            Ap = A[torch.tensor(piv, device=pb.DEV)]
            Api = torch.linalg.inv(Ap.to(torch.complex128)).to(torch.complex64)
            bases[L] = dict(Acard=(A @ Api), Dcard=(Ap @ b.D), nodes=np.array(nodes_t), piv=piv, cond=float(cond))
            pb.log(f'{dname}: svdn L={L} cardinal nodes (ms after TE) {np.round((np.array(nodes_t) - te) * 1e3, 2).tolist()}, cond {cond:.1f}')
            del b
        for ph in a.phases.split(','):
            key = f'{dname}_{ph}'
            if key in results:
                continue
            f = PH[ph]
            ps = flow.blood_frame(obj, 5 * f, TRS[dname])
            if a.matched:
                ps.t_om = obj.t_om[torch.tensor(ps.vox, device=pb.DEV)].clone()
                ps.t_r2 = torch.full_like(ps.t_r2, 31.0)
            mv = build_model_voxels(obj, ps, 5 * f)
            sel = mv['sel']; nm = len(sel)
            pb.log(f'{key}: {ps.N} sources, {nm} model voxels ({mv["n_support"]} with sources)')
            # regions on 2 mm voxels: the region with the largest (smoothed) source amplitude; ring voxels without sources
            # by the same smoothed majority. Truth region = the sources whose 2 mm voxel is in the region.
            dims = obj.dims; nvox = dims[0] * dims[1] * dims[2]
            ampn = ps.t_a.real.cpu().numpy()
            names = ['coronary', 'vessels', 'chambers', 'other']
            other = ~(ps.region_np['coronary'] | ps.region_np['vessels'] | ps.region_np['chambers'])
            rmask = dict(ps.region_np); rmask['other'] = other
            Am = np.stack([gaussian_filter(np.bincount(obj.idx[ps.vox[rmask[r]]], weights=ampn[rmask[r]], minlength=nvox).reshape(dims, order='F'), 0.7).ravel(order='F')[obj.idx[sel]] for r in names], 1)
            Am[:, 0] *= 3.0  # coronary voxels win against the adjacent chambers / aorta unless clearly dominated
            lab = Am.argmax(1)
            regs = {r: lab == i for i, r in enumerate(names[:3])}
            vlab = np.full(obj.N, -1); vlab[sel] = lab
            src_lab = vlab[ps.vox]
            for i, r in enumerate(names[:3]):
                ps.region[r] = torch.tensor(src_lab == i, device=pb.DEV)
                ps.region_np[r] = src_lab == i
            # model point set (2 mm voxels)
            mps = flow.PointSet(obj, obj.pos[sel], mv['x'], obj.om[sel], np.full(nm, 31.0), mv['v'], acc=mv['a'],
                                region={k: v for k, v in regs.items()})
            t_sel = torch.tensor(sel, device=pb.DEV)
            vhat_variants = {
                'true': mv['v'],
                'bias0.8': 0.8 * mv['v'],
                'noise0.05': mv['v'] + np.random.default_rng(1).normal(0, 0.05, mv['v'].shape),
            }
            # smooth4mm: Gaussian smoothing of v_hat over the model voxels on the grid (sigma 2 voxels)
            dims = obj.dims; nvox = dims[0] * dims[1] * dims[2]
            vg = np.zeros((nvox, 3)); wgt = np.zeros(nvox)
            vg[obj.idx[sel]] = mv['v'] * np.abs(mv['x'])[:, None]; wgt[obj.idx[sel]] = np.abs(mv['x'])
            sm = [gaussian_filter(vg[:, k].reshape(dims, order='F'), 1.0).ravel(order='F') for k in range(3)]
            sw = gaussian_filter(wgt.reshape(dims, order='F'), 1.0).ravel(order='F')
            vhat_variants['smooth4mm'] = np.stack([s_[obj.idx[sel]] for s_ in sm], 1) / np.maximum(sw[obj.idx[sel]], 1e-12)[:, None]
            t_vh = {k: torch.tensor(v, dtype=torch.float64, device=pb.DEV) for k, v in vhat_variants.items()}
            # T12: field gradient at the model voxels: the 2 mm field map (central differences) or the oracle
            # (amplitude-weighted source-field gradient of the voxel's blood, normalised convolution)
            hz3 = np.asarray(np.load(os.path.join(pb.SCAN, 'field_ppm_recon.npy'), mmap_mode='r')).astype(np.float64) * 2 * np.pi * pb.GBAR * obj.b0 * 1e-6
            gmap = np.stack(np.gradient(hz3, obj.res), -1).reshape(-1, 3, order='F')[obj.idx[sel]]
            gsrc_v = ps.extra['gradw'].cpu().numpy()
            qv = obj.idx[ps.vox]
            Aq = np.bincount(qv, weights=ampn, minlength=nvox)
            den = gaussian_filter(Aq.reshape(dims, order='F'), 0.7).ravel(order='F')
            gsrc = np.stack([gaussian_filter(np.bincount(qv, weights=ampn * gsrc_v[:, k], minlength=nvox).reshape(dims, order='F'), 0.7).ravel(order='F') / np.maximum(den, 1e-12 * den.max()) for k in range(3)], 1)[obj.idx[sel]]
            t_g = {'map': torch.tensor(gmap, dtype=torch.float64, device=pb.DEV), 'src': torch.tensor(gsrc, dtype=torch.float64, device=pb.DEV)}
            t_ah = torch.tensor(mv['a'], dtype=torch.float64, device=pb.DEV)
            t_J = torch.tensor(mv['J'], dtype=torch.float64, device=pb.DEV)  # nv x 3 x 3

            def model(variant='true', stencil='ideal', segL=None, accel=False, static=False, perp=False, region=None, pb0=None, shear=None):
                vh = t_vh[variant]
                def fn(ctx, sl):
                    base = mps.t_a[sl][None, :] * torch.exp(-31.0 * ctx.tau.to(torch.float32))[:, None]
                    msk = None if region is None else mps.region[region][sl][None, :].to(torch.complex64)
                    if segL is None:
                        ph = ctx.tau[:, None] * mps.t_om[sl][None, :] + ctx.w4 @ mps.mono[sl].T
                        z = base * pb.polar(ph)
                        if not static:
                            d = ctx.t1[:, None, None] * vh[sl][None, :, :]  # ns x nv x 3 (m)
                            if accel:
                                d = d + ctx.t2[:, None, None] * t_ah[sl][None, :, :]
                            sh = (ctx.k[:, None, :] * d).sum(2)
                            z = z * pb.polar(sh)
                            if stencil != 'ideal':
                                th = ctx.k[:, None, :] * obj.res
                                rho = torch.prod(stencil_resp(stencil, th.to(torch.float32), (d / obj.res).to(torch.float32)), 2)
                                z = z * rho
                            if perp:
                                z = z * pb.polar(ctx.m1perp @ vh[sl].T)
                            if pb0 is not None:
                                z = z * pb.polar(0.5 * (ctx.t ** 2)[:, None] * (t_g[pb0][sl] * vh[sl]).sum(1)[None, :])
                            if shear is not None:
                                # sub-voxel points move with v + J dr: phase t1 (J^T k).dr over the voxel; 2 points per axis
                                # at +-D/4 (the simulator's 1 mm sources): prod cos(u_a D / 4); a uniform box: prod sinc(u_a D / 2)
                                u = ctx.t1[:, None, None] * torch.einsum('vab,na->nvb', t_J[sl], ctx.k)
                                if shear == 'cos':
                                    fac = torch.prod(torch.cos(u * obj.res / 4), 2)
                                else:
                                    ar = u * obj.res / 2
                                    fac = torch.prod(torch.where(ar.abs() < 1e-8, torch.ones_like(ar), torch.sin(ar) / ar), 2)
                                z = z * fac.to(torch.float32)
                    else:
                        B = bases[segL]
                        nidx = torch.tensor(ctx.n, device=pb.DEV)
                        Ac = B['Acard'][nidx]  # ns x L
                        Dc = B['Dcard'][:, t_sel[sl]]  # L x nv
                        dq = (ctx.w4 - t_wbar[nidx]) @ mps.mono[sl].T
                        z = torch.zeros((len(ctx.n), len(sl)), dtype=torch.complex64, device=pb.DEV)
                        for l in range(segL):
                            pl = int(B['piv'][l])
                            term = Ac[:, l:l + 1] * Dc[l][None, :]
                            if not static:
                                d = float(des.t1[pl]) * vh[sl]
                                if accel:
                                    d = d + float(des.t2[pl]) * t_ah[sl]
                                sh = ctx.k @ d.T
                                term = term * pb.polar(sh)
                                if stencil != 'ideal':
                                    th = ctx.k[:, None, :] * obj.res
                                    rho = torch.prod(stencil_resp(stencil, th.to(torch.float32), (d[None, :, :] / obj.res).expand(len(ctx.n), -1, -1).to(torch.float32)), 2)
                                    term = term * rho
                                if pb0 is not None:
                                    tl = float(des.t[pl])
                                    term = term * pb.polar(0.5 * tl ** 2 * (t_g[pb0][sl] * vh[sl]).sum(1)[None, :])
                            z = z + term
                        z = z * base * pb.polar(dq)
                    return z if msk is None else z * msk
                return fn

            def truth(static=False, region=None, accel=False, pb0=False):
                def fn(ctx, sl):
                    z = flow.carrier(ps, ctx, sl)
                    if not static:
                        ph = ctx.m1 @ ps.t_v[sl].T
                        if accel:
                            ph = ph + ctx.m2h @ ps.t_acc[sl].T
                        if pb0:
                            ph = ph + 0.5 * (ctx.t ** 2)[:, None] * (ps.extra['gradw'][sl] * ps.t_v[sl]).sum(1)[None, :]
                        z = z * pb.polar(ph)
                    if region is not None:
                        z = z * ps.region[region][sl][None, :].to(torch.complex64)
                    return z
                return fn

            ros = pb.scan_readouts(a.nr, offset=9)
            t0 = time.time()
            Yt, _ = pb.eval_terms(ps, des, ros, {'moving': truth(), 'static': truth(static=True), 'moving_acc': truth(accel=True),
                                                 'moving_pb0': truth(pb0=True), 'moving_all': truth(accel=True, pb0=True),
                                                 'moving_cor': truth(region='coronary'), 'static_cor': truth(static=True, region='coronary'),
                                                 'moving_ves': truth(region='vessels'), 'static_ves': truth(static=True, region='vessels')},
                                  nsel=nsel, frame_override=f, complex_out=True)
            pb.log(f'{key}: truth ({time.time() - t0:.0f}s)')
            # name -> (model fn, truth key, (static model key, static truth key) for the floor)
            spec = {'static': (model(static=True), 'static', None), 'omit': (model(static=True), 'moving', ('static', 'static'))}
            for st in (['ideal', 'cubic', 'lanczos3'] if a.light else ['ideal', 'nearest', 'trilinear', 'cubic', 'lanczos3', 'lanczos4']):
                spec[f'exact_{st}'] = (model(stencil=st), 'moving', ('static', 'static'))
            spec['exact_ideal_perp'] = (model(perp=True), 'moving', ('static', 'static'))
            for vv in ([] if a.light else ['bias0.8', 'noise0.05', 'smooth4mm']):
                spec[f'exact_ideal_{vv}'] = (model(variant=vv), 'moving', ('static', 'static'))
            spec['acc_truth_exact_ideal_noacc'] = (model(), 'moving_acc', ('static', 'static'))
            spec['acc_truth_exact_ideal_acc'] = (model(accel=True), 'moving_acc', ('static', 'static'))
            spec['acc_truth_omit'] = (model(static=True), 'moving_acc', ('static', 'static'))
            spec['pb0_truth_omit'] = (model(static=True), 'moving_pb0', ('static', 'static'))
            spec['pb0_truth_exact_ideal_nopb0'] = (model(), 'moving_pb0', ('static', 'static'))
            spec['pb0_truth_exact_ideal_pb0map'] = (model(pb0='map'), 'moving_pb0', ('static', 'static'))
            spec['pb0_truth_exact_ideal_pb0src'] = (model(pb0='src'), 'moving_pb0', ('static', 'static'))
            spec['all_truth_omit'] = (model(static=True), 'moving_all', ('static', 'static'))
            spec['all_truth_exact_lanczos3_full'] = (model(stencil='lanczos3', accel=True, pb0='map'), 'moving_all', ('static', 'static'))
            for L in ([min(bases)] if a.light else bases):
                spec[f'seg{L}_static'] = (model(segL=L, static=True), 'static', None)
                for st in ['ideal', 'cubic', 'lanczos3']:
                    spec[f'seg{L}_{st}'] = (model(segL=L, stencil=st), 'moving', (f'seg{L}_static', 'static'))
                spec[f'seg{L}_ideal_acc_truth_acc'] = (model(segL=L, accel=True), 'moving_acc', (f'seg{L}_static', 'static'))
                spec[f'all_truth_seg{L}_lanczos3_full'] = (model(segL=L, stencil='lanczos3', accel=True, pb0='map'), 'moving_all', (f'seg{L}_static', 'static'))
                spec[f'all_truth_seg{L}_cubic_full'] = (model(segL=L, stencil='cubic', accel=True, pb0='map'), 'moving_all', (f'seg{L}_static', 'static'))
            for rg, sfx in [('coronary', 'cor'), ('vessels', 'ves')]:
                spec[f'static_{sfx}'] = (model(static=True, region=rg), f'static_{sfx}', None)
                spec[f'omit_{sfx}'] = (model(static=True, region=rg), f'moving_{sfx}', (f'static_{sfx}', f'static_{sfx}'))
                for st in (['ideal', 'lanczos3'] if a.light else ['ideal', 'trilinear', 'cubic', 'lanczos3']):
                    spec[f'exact_{st}_{sfx}'] = (model(stencil=st, region=rg), f'moving_{sfx}', (f'static_{sfx}', f'static_{sfx}'))
                spec[f'exact_ideal_perp_{sfx}'] = (model(perp=True, region=rg), f'moving_{sfx}', (f'static_{sfx}', f'static_{sfx}'))
                Lm = min(bases)
                spec[f'seg{Lm}_static_{sfx}'] = (model(segL=Lm, static=True, region=rg), f'static_{sfx}', None)
                spec[f'seg{Lm}_cubic_{sfx}'] = (model(segL=Lm, stencil='cubic', region=rg), f'moving_{sfx}', (f'seg{Lm}_static_{sfx}', f'static_{sfx}'))
            if a.shear_only:
                spec = {'static': (model(static=True), 'static', None), 'omit': (model(static=True), 'moving', ('static', 'static')),
                        'exact_ideal': (model(), 'moving', ('static', 'static')), 'exact_lanczos3': (model(stencil='lanczos3'), 'moving', ('static', 'static')),
                        'exact_ideal_shear_cos': (model(shear='cos'), 'moving', ('static', 'static')), 'exact_ideal_shear_sinc': (model(shear='sinc'), 'moving', ('static', 'static')),
                        'exact_lanczos3_shear_cos': (model(stencil='lanczos3', shear='cos'), 'moving', ('static', 'static'))}
                for rg, sfx in [('coronary', 'cor'), ('vessels', 'ves')]:
                    spec[f'static_{sfx}'] = (model(static=True, region=rg), f'static_{sfx}', None)
                    spec[f'omit_{sfx}'] = (model(static=True, region=rg), f'moving_{sfx}', (f'static_{sfx}', f'static_{sfx}'))
                    spec[f'exact_ideal_{sfx}'] = (model(region=rg), f'moving_{sfx}', (f'static_{sfx}', f'static_{sfx}'))
                    spec[f'exact_ideal_shear_cos_{sfx}'] = (model(region=rg, shear='cos'), f'moving_{sfx}', (f'static_{sfx}', f'static_{sfx}'))
                    spec[f'exact_lanczos3_shear_cos_{sfx}'] = (model(region=rg, stencil='lanczos3', shear='cos'), f'moving_{sfx}', (f'static_{sfx}', f'static_{sfx}'))
            t0 = time.time()
            Ym, _ = pb.eval_terms(mps, des, ros, {k: v[0] for k, v in spec.items()}, nsel=nsel, frame_override=f, complex_out=True)
            pb.log(f'{key}: models ({time.time() - t0:.0f}s)')
            nvr = {r: max(int(m_.sum()), 1) for r, m_ in regs.items()}
            nv_all = int(mv['n_support'])
            out = dict(n_model_voxels=nm, n_support=nv_all, nvox_regions=nvr, L_rule=Lr, nodes_ms={L: ((bases[L]['nodes'] - te) * 1e3).tolist() for L in bases}, cond={L: bases[L]['cond'] for L in bases})
            def Rd(Y):
                e = (np.abs(Y) ** 2).sum(2)
                return pb.R_split(obj, e, nsel, des)
            out['truth_motion_omission'] = Rd(Yt['moving'] - Yt['static'])
            out['truth_motion_acc_omission'] = Rd(Yt['moving_acc'] - Yt['static'])
            out['truth_motion_all_omission'] = Rd(Yt['moving_all'] - Yt['static'])
            out['gradient_hz_cm_p50_p95_map_src'] = [np.percentile(np.linalg.norm(gmap, axis=1), [50, 95]).tolist(), np.percentile(np.linalg.norm(gsrc, axis=1), [50, 95]).tolist()]
            out['gradient_hz_cm_p50_p95_map_src'] = [[v / (2 * np.pi * 100) for v in q] for q in out['gradient_hz_cm_p50_p95_map_src']]
            out['truth_motion_omission_cor'] = Rd(Yt['moving_cor'] - Yt['static_cor'])
            out['truth_motion_omission_ves'] = Rd(Yt['moving_ves'] - Yt['static_ves'])
            for k, (_, tk, fl) in spec.items():
                out[k] = Rd(Ym[k] - Yt[tk])
                if fl is not None:
                    out[k]['R_over_floor'] = Rd((Ym[k] - Yt[tk]) - (Ym[fl[0]] - Yt[fl[1]]))['R']
            out['R_loc_factor_all'] = float(np.sqrt(obj.N / nv_all))
            out['R_loc_factor_cor'] = float(np.sqrt(obj.N / nvr['coronary']))
            out['R_loc_factor_ves'] = float(np.sqrt(obj.N / nvr['vessels']))
            results[key] = out
            json.dump(results, open(fn, 'w'), indent=1)
            pb.log(f'{key}: ' + ', '.join(f'{k} {v["R"]:.3g}' + (f'/{v["R_over_floor"]:.3g}' if isinstance(v, dict) and 'R_over_floor' in v else '') for k, v in out.items() if isinstance(v, dict) and 'R' in v))
    pb.log('done')
