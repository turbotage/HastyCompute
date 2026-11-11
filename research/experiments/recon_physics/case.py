"""Validation case of the v2 reconstruction operator (research/recon_physics_wiring.md items 4 and 6).

    case.py prepare      -> <CASE>/case.json, x.npy, support.npy, z_{fit,wt,tab}.npy (rate maps), rate_stats.json
    case.py exact        -> <CASE>/y_exact.npy, ynav_exact.npy (direct sums from rec_truth.npy / recnav_truth.npy,
                            written by `opval_v2 records --chain resonance --tag truth`), energy.npy
    case.py eval [tags]  -> <CASE>/R.json (+ printed table): R of y_<tag>.npy against the exact data

Image and readouts as research/readout_physics_budget.md (pb.py imported read-only): the band-limited 2 mm truth image of
v2 (truth_image_lowmem.py, total = water + fat + blood), 32 whitened true coil maps, the budget's 48 readouts (encode,
frame, rotation; with their 10 previous TRs from girf_sensitivity's v2seq.scan_rotations) on v2's selected
reconstruction samples (all |k| < 150 rad/m, every 8th else; trapezoid weights) and every 2nd rewinder navigator sample.

Exact data (direct sum, f64 phases, GPU), readout r at its frame f:
  y_c(n) = e^{-i phi0_n} sum_v S_c(v) X_f(v, tau_n) e^{-i [k_n.r_v + wbar_v tau_n + w_n.m(v)]}
  X_f(v, tau) = x(v) [(1 - ff_f(v)) h_w(v, tau) + ff_f(v) c(tau) h_f(v, tau)]
with k, w (concomitant coefficients from the played int GG^T since the excitation), phi0 (B0-term phase), tau from the
records of the TRUE chain (girf.rs 'resonance', 10 previous TRs); h_w, h_f = the intravoxel effective signal of the
voxel's own tissue sources (pb.geff_tables: T2* and field spread of the 8 sources per voxel relative to the map field;
water with the blood fraction decaying at R2B = 31/s) normalised at the echo; ff_f = the truth fat fraction of frame f
(2 mm). Not in the truth: motion (static image), flow (T5/T6/T7/T12 of the budget), GNL. [CHECK: the static-image
convention of the budget.]
Rate maps for the model (z = R2* + i dw, s^-1, rad/s; dw relative to the map field):
  fit  per-voxel complex rate fitted (noise-time-weighted Gauss-Newton) to the frame-mean carrier over the window
  wt   table R2* + the signal-weighted source frequency of the voxel (what a data-estimated B0 map gives)
  tab  table R2* only (truth_r2s)
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'readout_physics'))
sys.path.insert(0, os.path.join(HERE, '..', 'girf_sensitivity'))
import pb  # noqa: E402

OUTD = os.path.join(pb.ROOT, 'recon_physics_2026-11-11')
CASE = os.environ.get('RP_CASE', os.path.join(OUTD, 'case_v2'))
R2B = 31.0
TR_V2 = 0.0141


def obj_truth():
    obj = pb.Obj(dil=1)
    pb.OUT = OUTD  # truth_x_v2.npz from truth_image_lowmem.py
    pb.use_truth(obj, 'v2')
    return obj


def carriers(obj, des):
    """h_w, h_f (nt x N, torch complex64, normalised at the echo) on the geff time grid, and the geff tables"""
    gt = pb.geff_tables(obj, tr=TR_V2)
    tH, Hw, Hf = gt['water'][0], gt['water'][1], gt['fat'][1]
    te_t = torch.tensor([des.te], dtype=torch.float64, device=pb.DEV)
    aw = torch.tensor(np.abs(obj.xw), dtype=torch.float32, device=pb.DEV)
    ab = torch.tensor(np.abs(obj.xb), dtype=torch.float32, device=pb.DEV)
    tt = torch.tensor(tH, dtype=torch.float32, device=pb.DEV)[:, None]
    hwb = (aw[None, :] * Hw + ab[None, :] * torch.exp(-R2B * tt)) / torch.clamp(aw + ab, min=1e-30)[None, :]
    # voxels without water / blood amplitude in the band-limited image: the water-source table (1 without sources)
    low = (aw + ab) <= 1e-6 * float((aw + ab).max())
    hwb[:, low] = Hw[:, low]
    del Hw
    def norm(H):
        h0 = pb.interp_t(tH, H, te_t)[0]
        a = torch.abs(h0)
        d = torch.where(a > 0.2, h0, h0 / torch.clamp(a, min=1e-30) * 0.2)
        return H / d[None, :], int((a <= 0.2).sum().item())
    hw, nw = norm(hwb)
    del hwb
    gt['water'] = (gt['water'][0], None) + tuple(gt['water'][2:])
    hasf = torch.tensor(gt['fat'][2] > 0, device=pb.DEV)
    hf, nf = norm(Hf)
    del Hf
    gt['fat'] = (gt['fat'][0], None) + tuple(gt['fat'][2:])
    hf = torch.where(hasf[None, :], hf, hw)
    torch.cuda.empty_cache()
    pb.log(f'carriers: {nw} water / {nf} fat voxels with |h(TE)| <= 0.2 (normalised by 0.2)')
    return tH, hw, hf, gt


def prepare():
    import v2seq
    os.makedirs(CASE, exist_ok=True)
    obj = obj_truth()
    des = pb.Design('v2')
    n, w = des.sample_select(kc=150.0, stride=8)
    na_all = 5395  # v2: 10.79 ms of ADC at 2 us
    nav = np.arange(des.na + 1, na_all, 2)
    rots = v2seq.scan_rotations(48, 7, 10)
    # pb synthesises v2 as R~ (R0 spoke) with R0 the design's own first rotation (the readout part of tr_grad[0]);
    # the records (opval_v2, as the simulator) play R spoke: compose R = R~ R0 to keep the budget's sample positions
    s_ = json.load(open(os.path.join(pb.DESIGNS['v2'], 'summary.json')))['best']
    gtr = np.load(os.path.join(pb.DESIGNS['v2'], 'tr_grad.npy'))[0][s_['i_ro']:s_['i_ro'] + s_['i_adc'] + 1]
    gsp = np.load(os.path.join(pb.DESIGNS['v2'], 'spoke_grad.npy'))[:s_['i_adc'] + 1]
    U, _, Vt = np.linalg.svd(gtr.T @ gsp)
    R0 = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    pb.log(f'design rotation R0: |R0 spoke - tr_grad readout| max {np.abs(gsp @ R0.T - gtr).max():.2e} T/m')
    rots = dict(rots, R=np.einsum('ijab,bc->ijac', rots['R'], R0))
    ros = []
    for i in range(rots['R'].shape[0]):
        ros.append(dict(idx=int(rots['idx'][i, -1]), enc=int(rots['enc'][i, -1]), frame=int(rots['frame'][i, -1]),
                        hist_enc=rots['enc'][i].tolist(), hist_rot=rots['R'][i].tolist(), n=n.tolist(), w=w.tolist(), nav=nav.tolist()))
    nx, ny, nz = obj.dims
    nv = nx * ny * nz
    full = lambda v, dt: (lambda a: (a.__setitem__(obj.idx, v), a)[1])(np.zeros(nv, dt))
    x = full(obj.x_tot.astype(np.complex64), np.complex64)
    np.save(os.path.join(CASE, 'x.npy'), x)
    np.save(os.path.join(CASE, 'support.npy'), full(np.ones(obj.N, np.float32), np.float32))
    # rate maps
    tH, hw, hf, gt = carriers(obj, des)
    ffm = torch.tensor(obj.ff.mean(1), device=pb.DEV)
    G = json.load(open(os.path.join(pb.ROOT, 'physics_budget_2026-11-10', 'global_v2.json')))
    n_sel = np.array(G['energy']['n']); e_sel = np.array(G['energy']['E'])
    import seg as segmod
    E_all = np.exp(np.interp(np.arange(des.na), n_sel, np.log(np.maximum(e_sel, 1e-30))))
    wn = segmod.noise_time_weights(E_all, des.kabs, obj.nc, obj.sigma)
    tt = torch.tensor(des.tau[::10], dtype=torch.float64, device=pb.DEV)
    ww = torch.tensor(wn[::10], dtype=torch.float64, device=pb.DEV)
    c = pb.fat_c(tt)[:, None].to(torch.complex128)
    dt_ = (tt - des.te)[:, None]
    z = torch.zeros(obj.N, dtype=torch.complex128, device=pb.DEV)
    r2tab = torch.tensor(obj.r2, dtype=torch.float64, device=pb.DEV)
    for c0 in range(0, obj.N, 20000):
        sl = torch.arange(c0, min(c0 + 20000, obj.N), device=pb.DEV)
        f = ffm[sl].to(torch.float64)[None, :]
        Y = ((1 - f) * pb.interp_t(tH, hw[:, sl], tt) + f * c * pb.interp_t(tH, hf[:, sl], tt)).to(torch.complex128)
        B = ((1 - f) + f * c)
        zz = r2tab[sl].to(torch.complex128)
        for it in range(12):
            model = B * torch.exp(-zz[None, :] * dt_)
            r = Y - model
            J = -dt_ * model
            num = (ww[:, None] * J.conj() * r).sum(0); den = (ww[:, None] * (J.abs() ** 2)).sum(0)
            zz = zz + num / torch.clamp(den, min=1e-30)
            zz = torch.complex(torch.clamp(zz.real, 0, 5000), torch.clamp(zz.imag, -2 * np.pi * 300, 2 * np.pi * 300))
        z[sl] = zz
    z = z.cpu().numpy()
    # signal-weighted source frequency (water and fat sources, blood at the map field) + table R2*
    Aw, Af = gt['water'][2], gt['fat'][2]
    dww = np.where(Aw > 0, gt['water'][3] - obj.om, 0.0); dwf = np.where(Af > 0, gt['fat'][3] - obj.om, 0.0)
    aw, af, ab = np.abs(obj.xw), np.abs(obj.xf), np.abs(obj.xb)
    dw = (aw * dww + af * dwf) / np.maximum(aw + af + ab, 1e-30)
    r2b = (aw * obj.r2 + af * obj.r2 + ab * R2B) / np.maximum(aw + af + ab, 1e-30)
    zwt = obj.r2 + 1j * dw
    ztab = obj.r2 + 0j
    for name, v in [('fit', z), ('wt', zwt), ('tab', ztab), ('tabb', r2b + 1j * dw)]:
        np.save(os.path.join(CASE, f'z_{name}.npy'), full(v.astype(np.complex64), np.complex64))
    m = obj.wE > 1e-3 * np.median(obj.wE[obj.wE > 0])
    stats = {k: dict(r2_p1_50_99=np.percentile(v.real[m], [1, 50, 99]).tolist(), df_hz_p1_50_99=np.percentile(v.imag[m] / 2 / np.pi, [1, 50, 99]).tolist())
             for k, v in [('fit', z), ('wt', zwt), ('tab', ztab)]}
    json.dump(stats, open(os.path.join(CASE, 'rate_stats.json'), 'w'), indent=1)
    case = dict(dims=list(obj.dims), vox=obj.res, nc=obj.nc, sigma=obj.sigma, b0=obj.b0, dwell=2e-6, na_win=int(des.na), na_all=na_all,
                design=pb.DESIGNS['v2'], scan=pb.SCAN, readouts=ros, te=des.te,
                note='research/experiments/recon_physics/case.py prepare (truth image: truth_image_lowmem.py v2)')
    json.dump(case, open(os.path.join(CASE, 'case.json'), 'w'))
    pb.log(f'case: {len(ros)} readouts, {len(n)} window + {len(nav)} navigator samples each; rate stats {stats}')


def exact(carrier='truth', rate='fit', recs='truth', kinds=('window', 'navigator')):
    """carrier 'truth' (the sources' intravoxel signal) or 'model' (the operator's own model x [(1 - f) + f c] e^{-z (tau - TE)}
    with the rate map z_<rate>: R against it isolates the operator's approximations from the physics floor)"""
    case = json.load(open(os.path.join(CASE, 'case.json')))
    obj = obj_truth()
    des = pb.Design('v2')
    if carrier == 'truth':
        tH, hw, hf, _ = carriers(obj, des)
    else:
        zf = np.load(os.path.join(CASE, f'z_{rate}.npy'))[obj.idx]
        tz = torch.tensor(zf, dtype=torch.complex64, device=pb.DEV)
    ffs = torch.tensor(obj.ff, device=pb.DEV)
    m4 = pb.monos(obj.t_pos)
    nc = obj.nc
    chunk = int(float(os.environ.get('PB_CHUNK_ELEMS', '3e7')))
    for kind, recf, key in [('window', f'rec_{recs}.npy', 'n'), ('navigator', f'recnav_{recs}.npy', 'nav')]:
        if kind not in kinds:
            continue
        rec = np.load(os.path.join(CASE, recf))
        out = []
        off = 0
        t0 = time.time()
        for r in case['readouts']:
            ns = len(r[key])
            rr = rec[off:off + ns].astype(np.float64); off += ns
            k = torch.tensor(rr[:, 0:3], device=pb.DEV)
            w4 = torch.tensor(rr[:, 9:13], device=pb.DEV)
            tau = torch.tensor(rr[:, 13], device=pb.DEV)
            ph0 = torch.tensor(rr[:, 15], device=pb.DEV)
            fr = r['frame']
            cfat = pb.fat_c(tau)[:, None]
            Y = torch.zeros((ns, nc), dtype=torch.complex64, device=pb.DEV)
            cv = max(2048, chunk // ns)
            for c0 in range(0, obj.N, cv):
                sl = torch.arange(c0, min(c0 + cv, obj.N), device=pb.DEV)
                f = ffs[sl, fr][None, :]
                if carrier == 'truth':
                    X = obj.t_x[sl][None, :] * ((1 - f) * pb.interp_t(tH, hw[:, sl], tau) + f * cfat * pb.interp_t(tH, hf[:, sl], tau))
                else:
                    X = obj.t_x[sl][None, :] * ((1 - f) + f * cfat) * torch.exp(-tz[sl][None, :] * (tau - des.te).to(torch.complex64)[:, None])
                ph = k @ obj.t_pos[sl].T + tau[:, None] * obj.t_om[sl][None, :] + w4 @ m4[sl].T
                Y += (pb.polar(ph) * X) @ obj.t_ST[sl]
            Y *= pb.polar(ph0)[:, None]
            out.append(Y.cpu().numpy())
        y = np.concatenate(out, 0)
        sfx = '' if (carrier == 'truth' and recs == 'truth') else f'_{carrier}{rate if carrier == "model" else ""}_{recs}'
        np.save(os.path.join(CASE, ('y_exact' if kind == 'window' else 'ynav_exact') + sfx + '.npy'), y.astype(np.complex64))
        if sfx:
            continue
        pb.log(f'exact {kind}: {y.shape} ({time.time() - t0:.0f}s)')
        if kind == 'window':
            nsel = np.array(case['readouts'][0]['n'])
            e = (np.abs(y.reshape(len(case['readouts']), len(nsel), nc)) ** 2).sum(2).mean(0)
            E_all = np.exp(np.interp(np.arange(case['na_win']), nsel, np.log(np.maximum(e, 1e-30))))
            np.save(os.path.join(CASE, 'energy.npy'), E_all.astype(np.float64))


def R_of(case, y, ye, nav=False):
    key, na = ('nav', case['na_all'] - case['na_win']) if nav else ('n', case['na_win'])
    nr = len(case['readouts'])
    ns = len(case['readouts'][0][key])
    w = np.full(ns, 2.0) if nav else np.array(case['readouts'][0]['w'])
    d = (np.abs(y - ye) ** 2).reshape(nr, ns, -1).sum(2).mean(0)
    nc = y.shape[1]
    return float(np.sqrt((w * d).sum() / (na * nc * case['sigma'] ** 2)))


def evaluate(tags):
    case = json.load(open(os.path.join(CASE, 'case.json')))
    ref = os.environ.get('RP_REF', 'y_exact')
    ye = np.load(os.path.join(CASE, ref + '.npy'))
    yn = np.load(os.path.join(CASE, 'ynav_exact.npy')) if os.path.exists(os.path.join(CASE, 'ynav_exact.npy')) else None
    des = pb.Design('v2')
    fn = os.path.join(CASE, 'R.json' if ref == 'y_exact' else f'R_{ref}.json')
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    if not tags:
        tags = sorted(f[2:-4] for f in os.listdir(CASE) if f.startswith('y_') and f.endswith('.npy') and f != 'y_exact.npy')
    n = np.array(case['readouts'][0]['n'])
    q = des.kabs[n] / des.kmax
    nr = len(case['readouts'])
    for t in tags:
        y = np.load(os.path.join(CASE, f'y_{t}.npy'))
        info = json.load(open(os.path.join(CASE, f'y_{t}.json'))) if os.path.exists(os.path.join(CASE, f'y_{t}.json')) else {}
        if info.get('nav'):
            if yn is None:
                continue
            res[t] = dict(R=R_of(case, y, yn[:, :y.shape[1]], nav=True), nav=True, info=info)
        else:
            nc = y.shape[1]
            yy = ye[:, :nc]
            d = (np.abs(y - yy) ** 2).reshape(nr, len(n), nc).sum(2).mean(0)
            w = np.array(case['readouts'][0]['w'])
            R = lambda msk: float(np.sqrt((w * d * msk).sum() / (case['na_win'] * nc * case['sigma'] ** 2)))
            res[t] = dict(R=R(1.0), **{f'R_k{a:.1f}-{b:.1f}': R(((q >= a) & (q < b)).astype(float)) for a, b in [(0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 9)]}, info=info)
        print(f'{t:>28}: R {res[t]["R"]:.4f}' + ('' if res[t].get('nav') else '  (|k| bands ' + ' '.join(f'{res[t][k]:.3f}' for k in res[t] if k.startswith('R_k')) + ')'))
    json.dump(res, open(fn, 'w'), indent=1)


if __name__ == '__main__':
    torch.set_num_threads(6)
    cmd = sys.argv[1]
    if cmd == 'prepare':
        prepare()
    elif cmd == 'exact':
        # exact [truth|model] [rate] [records tag] [window,navigator]
        a = sys.argv[2:] + [None] * 4
        exact(a[0] or 'truth', a[1] or 'fit', a[2] or 'truth', tuple((a[3] or 'window,navigator').split(',')))
    elif cmd == 'eval':
        evaluate(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
