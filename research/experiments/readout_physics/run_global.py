"""Global (static-tissue) terms on the truth-based image: T1 B0, T2 concomitant, T3 fat, T4 R2* (incl. the effective
intravoxel decay of the sources and complex-rate segments), T9 voxel basis, T10 GIRF. Both designs.

Carrier X(v, tau) = x_w H_w + x_f H_f c(tau) + x_b e^{-R2*_b tau} (H: intravoxel T2* and field spread of the voxel's
sources relative to its map field wbar_v), C(v, tau) = e^{-i wbar_v tau - i w_ro(tau).m(v)}.
Writes <OUT>/global_<design>.json (+ .npz with the per-sample error energies) and per-(design) segment tables.

    run_global.py [--designs current,v2] [--nr 48]
"""
import json, os, sys, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import pb, seg

ap = argparse.ArgumentParser()
ap.add_argument('--designs', default='current,v2')
ap.add_argument('--nr', type=int, default=48)
ap.add_argument('--parts', default='energy,t1,t2,t3,t4,t4seg,t9,t10')
ap.add_argument('--tag', default='')
ap.add_argument('--region', default='', help='restrict the error images to a 2 mm region (blood|vessels|coronary|myocardium); rule L values only')
a = ap.parse_args()
OUT = pb.OUT
TRS = {'current': 0.01129, 'v2': 0.0141}
R2B = 31.0  # blood 1/T2* (median of the blood sources)


def main(dname):
    fn = os.path.join(OUT, f'global_{dname}{a.tag}.json')
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    def save():
        json.dump(res, open(fn, 'w'), indent=1)
    obj = pb.use_truth(OBJ0, dname)
    global VOX
    VOX = region_vox(obj, a.region) if a.region else None
    if VOX is not None:
        res['region'] = dict(name=a.region, n_vox=int(len(VOX)), n_obj=int(obj.N))
    des = pb.Design(dname)
    nsel = des.sample_select(kc=150.0, stride=8)
    ros = pb.scan_readouts(a.nr, offset=7)
    te = des.te
    if not hasattr(OBJ0, 'gt') or OBJ0.gt_design != dname:
        OBJ0.gt = pb.geff_tables(obj, tr=TRS[dname]); OBJ0.gt_design = dname
    gt = OBJ0.gt
    tH, Hw, Hf = gt['water'][0], gt['water'][1], gt['fat'][1]
    m4 = pb.monos(obj.t_pos)
    nv = obj.N
    # encode-mean concomitant coefficients of the readout part (mean over 2000 rotations of the scan)
    rmean = pb.scan_readouts(400, offset=1)['R']
    wbar = np.mean([des.conc_w(R) for R in rmean], 0)  # na x 4
    t_wbar = torch.tensor(wbar, dtype=torch.float64, device=pb.DEV)
    # degree-1 energy-weighted fit of the field (b0-absorb 1)
    xs = seg.smooth_mag(obj, obj.x_tot)
    wv = xs.astype(np.float64) ** 2 * obj.cE; wv = wv / np.median(wv[wv > 0]) if (wv > 0).any() else wv
    fit = np.arange(0, nv, 3)
    Afit = np.c_[np.ones(len(fit)), obj.pos[fit]] * np.sqrt(wv[fit])[:, None]
    cfit = np.linalg.lstsq(Afit, obj.om[fit] * np.sqrt(wv[fit]), rcond=None)[0]
    om_lin = cfit[0] + obj.pos @ cfit[1:]
    t_omlin = torch.tensor(om_lin, dtype=torch.float64, device=pb.DEV)
    t_omres = obj.t_om - t_omlin
    res['b0_fit_deg1'] = dict(c0_hz=float(cfit[0] / 2 / np.pi), grad_hz_per_m=(cfit[1:] / 2 / np.pi).tolist(),
                              field_pct_1_99_hz=np.percentile(obj.hz[wv > 1e-3], [1, 99]).tolist(), resid_pct_1_99_hz=np.percentile((obj.om - om_lin)[wv > 1e-3] / 2 / np.pi, [1, 99]).tolist())
    nidx = lambda ctx: torch.tensor(ctx.n, device=pb.DEV)

    def wbar_at(ctx):
        return t_wbar[nidx(ctx)]

    def Xfull(ctx, sl):
        hw = pb.interp_t(tH, Hw[:, sl], ctx.tau); hf = pb.interp_t(tH, Hf[:, sl], ctx.tau)
        c = pb.fat_c(ctx.tau)[:, None]
        return obj.t_xw[sl][None, :] * hw + obj.t_xf[sl][None, :] * hf * c + obj.t_xb[sl][None, :] * torch.exp(-R2B * ctx.tau.to(torch.float32))[:, None]

    TE_t = torch.tensor([te], dtype=torch.float64, device=pb.DEV)

    def X0(ctx, sl):
        """amplitudes frozen at the echo (no R2*, no intravoxel dephasing after TE), fat spectrum kept"""
        hw = pb.interp_t(tH, Hw[:, sl], TE_t); hf = pb.interp_t(tH, Hf[:, sl], TE_t)
        c = pb.fat_c(ctx.tau)[:, None]
        return obj.t_xw[sl][None, :] * hw + obj.t_xf[sl][None, :] * hf * c + obj.t_xb[sl][None, :] * float(np.exp(-R2B * te))

    def Cfull(ctx, sl):
        return pb.polar(ctx.tau[:, None] * obj.t_om[sl][None, :] + ctx.w4 @ m4[sl].T)

    # ── data energy profile of the truth model (time weights of the rule, the reference energy) ──
    if VOX is not None and 'energy' not in res:
        g0 = json.load(open(os.path.join(OUT, f'global_{dname}.json')))
        res['energy'] = g0['energy']
    if 'energy' in a.parts and 'energy' not in res:
        E, _ = pb.eval_terms(obj, des, ros, {'model': lambda ctx, sl: Cfull(ctx, sl) * Xfull(ctx, sl)}, nsel=nsel)
        e = E['model'].mean(0)
        res['energy'] = dict(n=nsel[0].tolist(), E=e.tolist())
        save(); pb.log(f'{dname}: energy profile')
    n_sel = np.array(res['energy']['n']); e_sel = np.array(res['energy']['E'])
    E_all = np.exp(np.interp(np.arange(des.na), n_sel, np.log(np.maximum(e_sel, 1e-30))))
    wn = seg.noise_time_weights(E_all, des.kabs, obj.nc, obj.sigma)
    res['energy_summary'] = dict(E_vol_over_noise=float(seg.vol_energy(E_all, des.kabs) / (obj.nc * obj.sigma ** 2)), mean_w=float(wn.mean()),
                                 E0_over_noise=float(E_all[0] / (obj.nc * obj.sigma ** 2)), Ekmax_over_noise=float(E_all[np.argmax(des.kabs)] / (obj.nc * obj.sigma ** 2)))

    def run(name, efuns):
        """evaluate efuns (dict) and store R (+ k split) under res[name][key]"""
        todo = {k: f for k, f in efuns.items() if k not in res.get(name, {})}
        if not todo:
            return
        t0 = time.time()
        E, ns = pb.eval_terms(obj, des, ros, todo, nsel=nsel, vox=VOX)
        d = res.setdefault(name, {})
        for k, e in E.items():
            d[k] = pb.R_split(obj, e, ns, des)
            d[k]['E_n'] = e.mean(0).tolist()
            if VOX is not None:
                d[k]['R_loc'] = d[k]['R'] * float(np.sqrt(obj.N / len(VOX)))
        save(); pb.log(f'{dname} {name}: {list(todo)} ({time.time() - t0:.0f}s) ' + ', '.join(f'{k} {d[k]["R"]:.3g}' for k in todo))

    # ── T1 B0 ──
    def g_b0(t, sl, extra=None):
        """the segments' water function e^{-i psi_res}: residual field (degree 1 absorbed) + encode-mean concomitant"""
        n = torch.clamp(torch.round((t - des.tau[0]) / (des.tau[1] - des.tau[0])).long(), 0, des.na - 1)
        ph = t[:, None] * t_omres[sl][None, :] + t_wbar[n] @ m4[sl].T
        z = pb.polar(ph)
        if extra is not None:
            z = z * extra(t, sl)
        return z

    tw_tau = des.tau
    fam_cache = {}

    def family(key, g):
        if key not in fam_cache:
            fam_cache[key] = seg.Family(obj, g, tw_tau, wn, wv)
        return fam_cache[key]

    def rule_L(fam, R=0.2, margin=1.5):
        for L in range(2, 40):
            if margin * fam.tail(L) * np.sqrt(wn.mean()) <= R:
                return L
        return 40

    def seg_resid(basis, g, Xamp):
        """C_lin [g - sum a D] e^{-i dw.m} Xamp; C_lin = the absorbed degree-1 part"""
        def f(ctx, sl):
            approx = basis.approx(ctx.tau, sl)
            dq = (ctx.w4 - wbar_at(ctx)) @ m4[sl].T
            return pb.polar(ctx.tau[:, None] * t_omlin[sl][None, :] + dq) * (g(ctx.tau, sl) - approx) * Xamp(ctx, sl)
        return f

    xtot_static = lambda ctx, sl: obj.t_x[sl][None, :]
    if 't1' in a.parts:
        run('T1', {'omission': lambda ctx, sl: (pb.polar(ctx.tau[:, None] * obj.t_om[sl][None, :]) - 1) * pb.polar(ctx.w4 @ m4[sl].T) * Xfull(ctx, sl)})
        fam = family('b0', g_b0)
        Lr = rule_L(fam)
        res.setdefault('T1_rule', {})['svdn_L_rule'] = Lr
        res['T1_rule']['svdn_tail_pred'] = {L: fam.tail(L) * np.sqrt(wn.mean()) for L in range(4, 17)}
        Ls = sorted(set([Lr - 2, Lr - 1, Lr, Lr + 1, Lr + 2])) if VOX is None else [Lr]
        for L in Ls:
            b = fam.basis(L)
            run('T1', {f'svdn_L{L}': seg_resid(b, g_b0, xtot_static)})
            del b
        for L in ([12, 14, 17] if VOX is None else []):
            b = seg.basis_fixed(obj, g_b0, L, des.tau[0], des.tau[-1])
            run('T1', {f'fixed_L{L}': seg_resid(b, g_b0, xtot_static)})
            b = seg.basis_fixed(obj, g_b0, L, des.tau[0], des.tau[-1], wv_all=wv)
            run('T1', {f'fixed_e_L{L}': seg_resid(b, g_b0, xtot_static)})
            del b
        torch.cuda.empty_cache()

    # ── T2 concomitant ──
    if 't2' in a.parts:
        def t2_res(ctx, sl):
            q = (ctx.w4 - wbar_at(ctx)) @ m4[sl].T  # f64
            z = torch.polar(torch.ones_like(q), -q)
            tay = 1 - 1j * q - 0.5 * q ** 2
            base = pb.polar(ctx.tau[:, None] * obj.t_om[sl][None, :] + wbar_at(ctx) @ m4[sl].T)
            return base * (z - tay).to(torch.complex64) * Xfull(ctx, sl)
        def t2_dev_omit(ctx, sl):
            q = (ctx.w4 - wbar_at(ctx)) @ m4[sl].T
            base = pb.polar(ctx.tau[:, None] * obj.t_om[sl][None, :] + wbar_at(ctx) @ m4[sl].T)
            return base * (pb.polar(q) - 1) * Xfull(ctx, sl)
        run('T2', {'omission': lambda ctx, sl: pb.polar(ctx.tau[:, None] * obj.t_om[sl][None, :]) * (pb.polar(ctx.w4 @ m4[sl].T) - 1) * Xfull(ctx, sl),
                   'deviation_omitted': t2_dev_omit, 'taylor2_residual': t2_res})
        dq_max = 0.0
        for R in ros['R'][:16]:
            dw = des.conc_w(R) - wbar
            # largest |dw.m| over the support
            mm = m4.cpu().numpy()
            dq_max = max(dq_max, float(np.abs(dw @ mm[::50].T).max()))
        res['T2']['max_deviation_phase_rad'] = dq_max
        save()

    # ── T3 fat ──
    if 't3' in a.parts:
        ffs = torch.tensor(obj.ff, device=pb.DEV)
        ffm = ffs.mean(1)
        def fat_frame(ref):
            def f(ctx, sl):
                d = ffs[sl, ctx.frame] - (ffm[sl] if ref == 'mean' else ffs[sl, 0])
                return Cfull(ctx, sl) * obj.t_x[sl][None, :] * d[None, :] * (pb.fat_c(ctx.tau)[:, None] - 1)
            return f
        run('T3', {'omission': lambda ctx, sl: Cfull(ctx, sl) * obj.t_xf[sl][None, :] * pb.interp_t(tH, Hf[:, sl], ctx.tau) * (pb.fat_c(ctx.tau)[:, None] - 1),
                   'one_map_mean_vs_frame': fat_frame('mean'), 'one_map_frame0_vs_frame': fat_frame('frame0')})
        # joint water + fat folded (one map = the static truth fat share): svdn at the rule's L, and fixed nodes
        ffold = torch.tensor(np.where(np.abs(obj.x_tot) > 0, obj.xf / np.where(np.abs(obj.x_tot) > 0, obj.x_tot, 1), 0), dtype=torch.complex64, device=pb.DEV)
        def g_fat(t, sl):
            return g_b0(t, sl) * ((1 - ffold[sl])[None, :] + ffold[sl][None, :] * pb.fat_c(t)[:, None])
        famf = family('fat', g_fat)
        Lr = rule_L(famf)
        res.setdefault('T3_rule', {})['svdn_L_rule'] = Lr
        res['T3_rule']['svdn_tail_pred'] = {L: famf.tail(L) * np.sqrt(wn.mean()) for L in range(6, 20)}
        for L in (sorted(set([Lr - 1, Lr, Lr + 1])) if VOX is None else [Lr]):
            b = famf.basis(L)
            run('T3', {f'joint_svdn_L{L}': seg_resid(b, g_fat, xtot_static)})
            del b
        for L in ([15, 18, 22] if VOX is None else []):
            b = seg.basis_fixed(obj, g_fat, L, des.tau[0], des.tau[-1], wv_all=wv)
            run('T3', {f'joint_fixed_e_L{L}': seg_resid(b, g_fat, xtot_static)})
            del b
        torch.cuda.empty_cache()

    # ── T4 R2* ──
    if 't4' in a.parts:
        r2tab = obj.t_r2
        def Xt(ctx, sl):
            return X0(ctx, sl) * torch.exp(-r2tab[sl][None, :] * (ctx.tau - te).to(torch.float32)[:, None])
        # per-voxel effective complex rate z = R2*_eff + i dw_eff: weighted Gauss-Newton fit of X / X0 over the window
        if not hasattr(OBJ0, 'zfit') or OBJ0.zfit_design != dname:
            tt = torch.tensor(des.tau[::10], dtype=torch.float64, device=pb.DEV)
            ww = torch.tensor(wn[::10], dtype=torch.float64, device=pb.DEV)
            class C_: pass
            ctx = C_(); ctx.tau = tt
            z = torch.zeros(nv, dtype=torch.complex128, device=pb.DEV)
            for c0 in range(0, nv, 50000):
                sl = torch.arange(c0, min(c0 + 50000, nv), device=pb.DEV)
                Y = Xfull(ctx, sl).to(torch.complex128); B = X0(ctx, sl).to(torch.complex128)
                zz = r2tab[sl].to(torch.complex128)
                dt_ = (tt - te)[:, None]
                for it in range(12):
                    model = B * torch.exp(-zz[None, :] * dt_)
                    r = Y - model
                    J = -dt_ * model  # d model / d z
                    num = (ww[:, None] * J.conj() * r).sum(0); den = (ww[:, None] * (J.abs() ** 2)).sum(0)
                    zz = zz + num / torch.clamp(den, min=1e-30)
                    zz = torch.complex(torch.clamp(zz.real, 0, 5000), torch.clamp(zz.imag, -2 * np.pi * 300, 2 * np.pi * 300))
                z[sl] = zz
            OBJ0.zfit = z; OBJ0.zfit_design = dname
            zr = z.real.cpu().numpy(); zi = z.imag.cpu().numpy() / 2 / np.pi
            m_ = wv > 1e-3
            res['T4_fit'] = dict(R2eff_pct_1_50_99=np.percentile(zr[m_], [1, 50, 99]).tolist(), R2tab_pct_1_50_99=np.percentile(obj.r2[m_], [1, 50, 99]).tolist(),
                                 dfeff_hz_pct_1_50_99=np.percentile(zi[m_], [1, 50, 99]).tolist(),
                                 R2eff_minus_tab_energy_weighted_mean=float(((zr - obj.r2) * wv)[m_].sum() / wv[m_].sum()),
                                 R2eff_minus_tab_energy_weighted_absmean=float((np.abs(zr - obj.r2) * wv)[m_].sum() / wv[m_].sum()))
            save()
        zf = OBJ0.zfit.to(torch.complex64)
        def Xz(ctx, sl):
            return X0(ctx, sl) * torch.exp(-zf[sl][None, :] * (ctx.tau - te).to(torch.complex64)[:, None])
        def Xz_real(ctx, sl):
            return X0(ctx, sl) * torch.exp(-zf[sl].real[None, :] * (ctx.tau - te).to(torch.float32)[:, None])
        run('T4', {'omission': lambda ctx, sl: Cfull(ctx, sl) * (Xfull(ctx, sl) - X0(ctx, sl)),
                   'table_residual': lambda ctx, sl: Cfull(ctx, sl) * (Xfull(ctx, sl) - Xt(ctx, sl)),
                   'eff_complex_residual': lambda ctx, sl: Cfull(ctx, sl) * (Xfull(ctx, sl) - Xz(ctx, sl)),
                   'eff_real_residual': lambda ctx, sl: Cfull(ctx, sl) * (Xfull(ctx, sl) - Xz_real(ctx, sl))})

    # ── T4: segments with complex rates ──
    if 't4seg' in a.parts:
        r2tab = obj.t_r2
        def dec_tab(t, sl):
            return torch.exp(-r2tab[sl][None, :] * torch.clamp(t - te, min=0).to(torch.float32)[:, None]).to(torch.complex64)
        zf = OBJ0.zfit.to(torch.complex64) if hasattr(OBJ0, 'zfit') else None
        def dec_eff(t, sl):
            return torch.exp(-zf[sl][None, :] * torch.clamp(t - te, min=0).to(torch.complex64)[:, None])
        g_dec = lambda t, sl: g_b0(t, sl, dec_tab)
        g_deff = lambda t, sl: g_b0(t, sl, dec_eff)
        rr = res.setdefault('T4seg', {})
        for key, g in [('nodecay', g_b0), ('table', g_dec), ('effective', g_deff)]:
            fam = family(key, g)
            rr[f'{key}_svdn_L_rule'] = rule_L(fam)
            rr[f'{key}_svdn_tail_pred'] = {L: fam.tail(L) * np.sqrt(wn.mean()) for L in range(4, 15)}
        # the same rule with time-averaged (decay-weighted) voxel weights: double counting check
        dmean = np.exp(-obj.r2[:, None] * (des.tau[None, ::40] - te)) @ (wn[::40] / wn[::40].sum())
        famd = seg.Family(obj, g_dec, tw_tau, wn, wv * dmean ** 2)
        rr['table_svdn_L_rule_decayweighted_voxels'] = rule_L(famd)
        del famd
        save()
        Lr = rr['nodecay_svdn_L_rule']
        for L in (sorted(set([Lr - 1, Lr, Lr + 1])) if VOX is None else [Lr]):
            for key, g in [('nodecay', g_b0), ('table', g_dec)]:
                b = family(key, g).basis(L)
                # each basis on its own target, and the no-decay basis applied to the decaying target
                run('T4seg', {f'{key}_svdn_L{L}': seg_resid(b, g, xtot_static)})
                del b
        for L in ([10, 12, 14] if VOX is None else []):
            for key, g in [('nodecay', g_b0), ('table', g_dec)]:
                b = seg.basis_fixed(obj, g, L, des.tau[0], des.tau[-1])
                run('T4seg', {f'{key}_fixed_hist_L{L}': seg_resid(b, g, xtot_static)})
                del b
        torch.cuda.empty_cache()

    # ── T9 voxel basis ──
    if 't9' in a.parts:
        dlt = obj.res
        def sincw(ctx):
            k = ctx.k * (dlt / 2)
            return torch.prod(torch.where(k.abs() < 1e-6, torch.ones_like(k), torch.sin(k) / k), 1).to(torch.float32)
        # absorbed: x' = IDFT[ prod sinc(k_g D/2) DFT(x) ] on the 2 mm grid (the image redefinition a Dirac model makes)
        nx, ny, nz = obj.dims
        gg = np.zeros(nx * ny * nz, np.complex64); gg[obj.idx] = obj.x_tot
        G = np.fft.fftn(gg.reshape((nx, ny, nz), order='F'))
        kk = [2 * np.pi * np.fft.fftfreq(m, dlt) * dlt / 2 for m in (nx, ny, nz)]
        sw = np.sinc(kk[0][:, None, None] / np.pi) * np.sinc(kk[1][None, :, None] / np.pi) * np.sinc(kk[2][None, None, :] / np.pi)
        xp = np.fft.ifftn(G * sw).ravel(order='F')[obj.idx].astype(np.complex64)
        t_xp = torch.tensor(xp, device=pb.DEV)
        run('T9', {'omission': lambda ctx, sl: Cfull(ctx, sl) * Xfull(ctx, sl) * (sincw(ctx)[:, None] - 1),
                   'absorbed_residual': lambda ctx, sl: Cfull(ctx, sl) * (t_xp[sl][None, :] - obj.t_x[sl][None, :] * sincw(ctx)[:, None])})

    # ── T10 GIRF (illustrative clinical chain, rotation-0 error rotated with the readout; 5 % residual) ──
    if 't10' in a.parts:
        run_dir = pb.DESIGNS[dname]
        ge = np.load(run_dir + '/girf_err.npy') * 2 * np.pi  # rad/m, 10 us raster from the ADC start
        tg = np.arange(len(ge)) * 1e-5
        dk = np.stack([np.interp(des.t, tg, ge[:, c]) for c in range(3)], 1)
        def girf(scale):
            def f(ctx, sl):
                d = torch.tensor(dk[ctx.n] @ ctx.R.T, dtype=torch.float64, device=pb.DEV) * scale
                return Cfull(ctx, sl) * Xfull(ctx, sl) * (pb.polar(d @ obj.t_pos[sl].T) - 1)
            return f
        run('T10', {'girf_omission': girf(1.0), 'girf_5pct_residual': girf(0.05)})
    save()


VOX = None


def region_vox(obj, name):
    """object voxel indices of a 2 mm region at the reference pose (truth frame 0), dilated by 1 voxel"""
    from scipy.ndimage import binary_dilation
    import flow
    nx, ny, nz = obj.dims
    if name == 'blood':
        m = obj.lumen > 0
    elif name in ('vessels', 'myocardium'):
        lab = obj.labels2(0)
        m = np.isin(lab, [24, 25, 26, 27]) if name == 'vessels' else lab == 18
    elif name == 'coronary':
        P, Rr = flow.coronary_points(0)
        from scipy.spatial import cKDTree
        c = np.array(flow.sim()['presc']['center'])
        d, j = cKDTree(P).query(obj.pos + c, distance_upper_bound=0.01)
        m = np.isfinite(d) & (d <= Rr[np.minimum(j, len(Rr) - 1)] + 0.001)
    g = np.zeros(nx * ny * nz, bool); g[obj.idx[m]] = True
    if name != 'myocardium':
        g = binary_dilation(g.reshape((nx, ny, nz), order='F'), iterations=1).ravel(order='F')
    return np.flatnonzero(g[obj.idx])


if __name__ == '__main__':
    torch.set_num_threads(6)
    OBJ0 = pb.Obj()
    for d in a.designs.split(','):
        main(d)
    pb.log('done')
