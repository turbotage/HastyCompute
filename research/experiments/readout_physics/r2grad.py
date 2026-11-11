"""T4 follow-up: which decay / frequency model closes the R2* gap (truth = the voxel's own tissue sources, H tables).

Variants (all with the voxel amplitude frozen at the echo, X0, then a per-voxel time factor relative to TE):
  table          e^{-R2*_tab (tau - TE)}                         (truth_r2s: PD-weighted mean of 1/T2*)
  wmean          the map frequency replaced by the signal-weighted source frequency (what a data-estimated map gives)
  gsinc          prod_a sinc(g_a D tau / 2) / (same at TE), g = gradient of the 2 mm field map (box voxel, linear field)
  gcos           prod_a cos(g_a D tau / 4) / (same at TE): the simulator's sub-voxel geometry (2 points per axis at
                 +-D/4) in a linear field
  combinations, and a per-voxel complex rate z refitted on top of gsinc (noise-time-weighted Gauss-Newton).
Also: how linear is the intravoxel field of the simulator (source-field spread vs the gradient prediction).
Writes <OUT>/r2grad_<design>.json.
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import pb, seg

OUT = pb.OUT
TRS = {'current': 0.01129, 'v2': 0.0141}
R2B = 31.0

if __name__ == '__main__':
    torch.set_num_threads(6)
    obj0 = pb.Obj()
    hz3 = np.asarray(np.load(os.path.join(pb.SCAN, 'field_ppm_recon.npy'), mmap_mode='r')).astype(np.float64) * 2 * np.pi * pb.GBAR * obj0.b0 * 1e-6
    gmap = np.stack(np.gradient(hz3, obj0.res), -1).reshape(-1, 3, order='F')[obj0.idx]  # rad/s/m
    tg = torch.tensor(gmap, dtype=torch.float64, device=pb.DEV)
    for dname in ['v2', 'current']:
        fn = os.path.join(OUT, f'r2grad_{dname}.json')
        if os.path.exists(fn):
            continue
        obj = pb.use_truth(obj0, dname)
        des = pb.Design(dname)
        te = des.te
        nsel = des.sample_select(kc=150.0, stride=8)
        ros = pb.scan_readouts(48, offset=7)
        gt = pb.geff_tables(obj, tr=TRS[dname])
        tH, Hw, Hf = gt['water'][0], gt['water'][1], gt['fat'][1]
        wm_w = torch.tensor(gt['water'][3], dtype=torch.float64, device=pb.DEV)
        wm_f = torch.tensor(gt['fat'][3], dtype=torch.float64, device=pb.DEV)
        Aw = torch.tensor(gt['water'][2], dtype=torch.float64, device=pb.DEV)
        Af = torch.tensor(gt['fat'][2], dtype=torch.float64, device=pb.DEV)
        m4 = pb.monos(obj.t_pos)
        TE_t = torch.tensor([te], dtype=torch.float64, device=pb.DEV)
        G = json.load(open(os.path.join(OUT, f'global_{dname}.json')))
        n_sel = np.array(G['energy']['n']); e_sel = np.array(G['energy']['E'])
        E_all = np.exp(np.interp(np.arange(des.na), n_sel, np.log(np.maximum(e_sel, 1e-30))))
        wn = seg.noise_time_weights(E_all, des.kabs, obj.nc, obj.sigma)

        def Xfull(ctx, sl):
            hw = pb.interp_t(tH, Hw[:, sl], ctx.tau); hf = pb.interp_t(tH, Hf[:, sl], ctx.tau)
            return obj.t_xw[sl][None, :] * hw + obj.t_xf[sl][None, :] * hf * pb.fat_c(ctx.tau)[:, None] + obj.t_xb[sl][None, :] * torch.exp(-R2B * ctx.tau.to(torch.float32))[:, None]

        def parts0(ctx, sl):
            hw = pb.interp_t(tH, Hw[:, sl], TE_t); hf = pb.interp_t(tH, Hf[:, sl], TE_t)
            return obj.t_xw[sl][None, :] * hw, obj.t_xf[sl][None, :] * hf * pb.fat_c(ctx.tau)[:, None], obj.t_xb[sl][None, :] * float(np.exp(-R2B * te))

        def C(ctx, sl):
            return pb.polar(ctx.tau[:, None] * obj.t_om[sl][None, :] + ctx.w4 @ m4[sl].T)

        def gfac(ctx, sl, kind):
            g = tg[sl]  # nv x 3
            def f(t):
                a = t[:, None, None] * g[None, :, :] * obj.res
                if kind == 'sinc':
                    a = a / 2
                    return torch.prod(torch.where(a.abs() < 1e-8, torch.ones_like(a), torch.sin(a) / a), 2)
                return torch.prod(torch.cos(a / 4), 2)
            return (f(ctx.tau) / torch.clamp(f(TE_t), min=1e-3)).to(torch.float32)

        def model(table=False, wmean=False, grad=None):
            def fn(ctx, sl):
                w0, f0, b0 = parts0(ctx, sl)
                dt = (ctx.tau - te)
                if wmean:
                    w0 = w0 * pb.polar(dt[:, None] * (wm_w[sl] - obj.t_om[sl])[None, :] * (Aw[sl] > 0)[None, :])
                    f0 = f0 * pb.polar(dt[:, None] * (wm_f[sl] - obj.t_om[sl])[None, :] * (Af[sl] > 0)[None, :])
                z = w0 + f0
                if table:
                    z = z * torch.exp(-obj.t_r2[sl][None, :] * dt.to(torch.float32)[:, None])
                if grad is not None:
                    z = z * gfac(ctx, sl, grad)
                return C(ctx, sl) * (Xfull(ctx, sl) - (z + b0))
            return fn

        # complex rate refit on top of the gradient factor (and without, for reference)
        tt = torch.tensor(des.tau[::10], dtype=torch.float64, device=pb.DEV)
        ww = torch.tensor(wn[::10], dtype=torch.float64, device=pb.DEV)
        class C_: pass
        cx = C_(); cx.tau = tt
        zfit = {}
        for key, gk in [('cplx', None), ('cplx+gsinc', 'sinc')]:
            z = torch.zeros(obj.N, dtype=torch.complex128, device=pb.DEV)
            for c0 in range(0, obj.N, 50000):
                sl = torch.arange(c0, min(c0 + 50000, obj.N), device=pb.DEV)
                Y = Xfull(cx, sl).to(torch.complex128)
                w0, f0, b0 = parts0(cx, sl)
                B = (w0 + f0).to(torch.complex128)
                Y = Y - b0.to(torch.complex128)
                if gk is not None:
                    B = B * gfac(cx, sl, gk).to(torch.float64)
                zz = obj.t_r2[sl].to(torch.complex128)
                dt_ = (tt - te)[:, None]
                for it in range(12):
                    mdl = B * torch.exp(-zz[None, :] * dt_)
                    r = Y - mdl
                    J = -dt_ * mdl
                    num = (ww[:, None] * J.conj() * r).sum(0); den = (ww[:, None] * (J.abs() ** 2)).sum(0)
                    zz = zz + num / torch.clamp(den, min=1e-30)
                    zz = torch.complex(torch.clamp(zz.real, 0, 5000), torch.clamp(zz.imag, -2 * np.pi * 300, 2 * np.pi * 300))
                z[sl] = zz
            zfit[key] = z.to(torch.complex64)

        def model_z(key, gk):
            def fn(ctx, sl):
                w0, f0, b0 = parts0(ctx, sl)
                z = (w0 + f0) * torch.exp(-zfit[key][sl][None, :] * (ctx.tau - te).to(torch.complex64)[:, None])
                if gk is not None:
                    z = z * gfac(ctx, sl, gk)
                return C(ctx, sl) * (Xfull(ctx, sl) - (z + b0))
            return fn

        ef = {'table': model(table=True), 'wmean': model(wmean=True), 'wmean+table': model(table=True, wmean=True), 'gsinc': model(grad='sinc'),
              'table+gsinc': model(table=True, grad='sinc'), 'table+gcos': model(table=True, grad='cos'), 'wmean+table+gsinc': model(table=True, wmean=True, grad='sinc'),
              'wmean+table+gcos': model(table=True, wmean=True, grad='cos'), 'cplx': model_z('cplx', None), 'cplx+gsinc': model_z('cplx+gsinc', 'sinc'),
              'omission': model()}
        t0 = time.time()
        E, ns = pb.eval_terms(obj, des, ros, ef, nsel=nsel)
        res = {k: pb.R_split(obj, e, ns, des) for k, e in E.items()}
        # how linear is the simulator's intravoxel field: source-field spread per voxel vs the gradient prediction
        m_ = obj.wE > 1e-3 * np.median(obj.wE)
        res['note'] = 'R of C (X_truth - X_model); X_truth from the sources (H tables)'
        res['gradient_hz_cm_pct_50_90_99'] = (np.percentile(np.linalg.norm(gmap[m_], axis=1), [50, 90, 99]) / (2 * np.pi * 100)).tolist()
        res['g_D_tmax_over4_gt1_energy_fraction'] = float((obj.wE[m_] * (np.abs(gmap[m_]).max(1) * obj.res * (des.tau[-1]) / 4 > 1)).sum() / obj.wE[m_].sum())
        json.dump(res, open(fn, 'w'), indent=1)
        pb.log(f'{dname} ({time.time() - t0:.0f}s): ' + ', '.join(f'{k} {v["R"]:.3g}' for k, v in res.items() if isinstance(v, dict) and 'R' in v))
    pb.log('done')
