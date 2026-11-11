"""T8: tissue motion during the readout (cardiac wall motion, respiration) on the truth tissue image (reference pose).

Lagrangian tissue velocity at the reference pose X: cardiac v_c = dU/dt (motion/cardiac_motion.hv, 48 periodic samples,
central differences, trilinear), respiratory v_r = rdot(t) sum_k B_k(X) (motion/resp_motion.hv; rdot of the scan's
breathing trace: its 99th percentile |rdot| = 'resp_peak'). Error image C (x_w H_w + x_f H_f c) (e^{-i m1.v} - 1),
regions all tissue / myocardium (truth label 18 at frame 0, 2 mm majority). Phases: systole (truth frame 3), diastole
(frame 12); readouts at that phase (48). Writes <OUT>/tissue_<design>.json.
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import pb, flow
sys.path.insert(0, os.path.join(pb.REPO, 'python/lib'))
from hastycompute import value_format as vf

OUT = pb.OUT
TRS = {'current': 0.01129, 'v2': 0.0141}
PH = {'systole': 3, 'diastole': 12}


def tri_grid(G, org, h, p):
    """G: (C, Z, Y, X) on origin org (x, y, z), spacing h; p (n, 3) subject coords -> (n, C), 0 outside"""
    Gt = np.moveaxis(G, 0, -1)  # Z, Y, X, C
    Gx = np.ascontiguousarray(np.transpose(Gt, (2, 1, 0, 3)))  # X, Y, Z, C
    return flow.trilinear(Gx.astype(np.float32), np.asarray(org), h, p)


if __name__ == '__main__':
    torch.set_num_threads(6)
    obj0 = pb.Obj()
    lab = obj0.labels2(0)
    myo = lab == 18
    sim = flow.sim()
    c = np.array(sim['presc']['center'])
    psub = obj0.pos + c
    cm = vf.open_lazy(os.path.join(pb.ROOT, 'motion/cardiac_motion.hv'))
    U = np.asarray(cm['U_mm'][:]).astype(np.float32) * 1e-3  # 48, 3, Z, Y, X (m)
    tU = np.asarray(cm['t'][:]); per = float(cm['period'])
    rm = vf.open_lazy(os.path.join(pb.ROOT, 'motion/resp_motion.hv'))
    B = np.asarray(rm['B'][:]).astype(np.float32).sum(0)  # 3, Z, Y, X
    vr_unit = tri_grid(B, rm['origin_m'], float(rm['voxel_m']), psub)
    rdot = np.asarray(sim['breathing']['rdot'][:])
    rpk = float(np.percentile(np.abs(rdot), 99))
    for dname in ['current', 'v2']:
        fn = os.path.join(OUT, f'tissue_{dname}.json')
        if os.path.exists(fn):
            continue
        obj = pb.use_truth(obj0, dname)
        des = pb.Design(dname)
        nsel = des.sample_select(kc=150.0, stride=8)
        gt = pb.geff_tables(obj, tr=TRS[dname])
        tH, Hw, Hf = gt['water'][0], gt['water'][1], gt['fat'][1]
        m4 = pb.monos(obj.t_pos)
        tissue = (np.abs(obj.xw) + np.abs(obj.xf)) > 0
        res = dict(resp_rdot_p99=rpk, resp_speed_unit_p99_mm_s=float(np.percentile(np.linalg.norm(vr_unit[tissue], axis=1), 99) * 1e3))
        for ph, f in PH.items():
            tc = f * per / 20
            # cardiac velocity: central difference of the periodic samples around tc (linear interpolation in t)
            dt = tU[1] - tU[0]
            i = int(np.floor((tc - tU[0]) / dt)) % len(tU)
            fr = (tc - tU[0]) / dt - np.floor((tc - tU[0]) / dt)
            def vel_at(j):
                return (U[(j + 1) % len(tU)] - U[(j - 1) % len(tU)]) / (2 * dt)
            Vg = (1 - fr) * vel_at(i) + fr * vel_at(i + 1)
            vc = tri_grid(Vg, cm['origin_m'], float(cm['voxel_m']), psub)
            variants = {'cardiac': vc, 'resp_peak': vr_unit * rpk, 'cardiac+resp_peak': vc + vr_unit * rpk}
            tv = {k: torch.tensor(np.where(tissue[:, None], v, 0.0), dtype=torch.float64, device=pb.DEV) for k, v in variants.items()}
            tmyo = torch.tensor(myo, device=pb.DEV)
            def X(ctx, sl):
                hw = pb.interp_t(tH, Hw[:, sl], ctx.tau); hf = pb.interp_t(tH, Hf[:, sl], ctx.tau)
                return obj.t_xw[sl][None, :] * hw + obj.t_xf[sl][None, :] * hf * pb.fat_c(ctx.tau)[:, None]
            def C(ctx, sl):
                return pb.polar(ctx.tau[:, None] * obj.t_om[sl][None, :] + ctx.w4 @ m4[sl].T)
            def mk(var, kind, reg):
                def fn_(ctx, sl):
                    v = tv[var][sl]
                    ph_ = ctx.m1 @ v.T if kind == 'm1' else (ctx.t1[:, None] * (ctx.k @ v.T) if kind == 'disp' else ctx.m1perp @ v.T)
                    z = C(ctx, sl) * X(ctx, sl) * (pb.polar(ph_) - 1)
                    return z if reg == 'all' else z * tmyo[sl][None, :].to(torch.complex64)
                return fn_
            ef = {f'{var}|{kind}|{reg}': mk(var, kind, reg) for var in variants for kind in ['m1', 'disp', 'perp'] for reg in ['all', 'myo']
                  if not (var != 'cardiac' and kind != 'm1')}
            ros = pb.scan_readouts(48, offset=13)
            E, ns = pb.eval_terms(obj, des, ros, ef, nsel=nsel, frame_override=f)
            out = {}
            nmyo = int(myo.sum()); nt = int(tissue.sum())
            for k, e in E.items():
                r = pb.R_split(obj, e, ns, des)
                reg = k.split('|')[2]
                r['R_loc'] = r['R'] * np.sqrt(obj.N / (nmyo if reg == 'myo' else nt))
                out[k] = r
            out['speed_pct_50_95_99_myo_mm_s'] = (np.percentile(np.linalg.norm(vc[myo], axis=1), [50, 95, 99]) * 1e3).tolist()
            out['n_myo'] = nmyo; out['n_tissue'] = nt
            res[ph] = out
            pb.log(f'{dname} {ph}: ' + ', '.join(f'{k} {v["R"]:.3g}/{v["R_loc"]:.3g}' for k, v in out.items() if isinstance(v, dict)))
        json.dump(res, open(fn, 'w'), indent=1)
    pb.log('done')
