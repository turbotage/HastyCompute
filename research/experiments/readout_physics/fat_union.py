"""T3 handling: per-frame fat maps with a noise-weighted SVD basis.

The joint water + fat per-frame-map operator applies, per segment node l, the fat factor (1 - f) + f c(tau_l) to an
interpolatory basis. Any basis in cardinal form is interpolatory, so the weighted SVD can serve it if its time span holds
both the water functions g(v, t) and the fat functions g(v, t) c(t) of every voxel ('union' Gram: water rows weighted
w_v (1 - f_min)^2, fat rows w_v f_max^2 over the frames), interpolators by least squares over the same union rows, then
QDEIM nodes. Compared (per-frame truth f of each readout's frame, static image, 48 readouts) with
  * fixed uniform nodes + the same union least-squares interpolators (the existing per-frame path, `--seg-weight energy`);
  * one folded map (time mean) with the svdn basis at its rule L (= run_global T3 joint_svdn) against the per-frame truth.
Writes <OUT>/fat_union_<design>.json.
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import pb, seg

OUT = pb.OUT


def union_family(obj, g, cfat, ts, q, w_water, w_fat, fit):
    """Gram of the union rows on the time grid; returns eigen-decomposition and the projection helper"""
    st = torch.tensor(np.sqrt(q), dtype=torch.float32, device=pb.DEV)
    tts = torch.tensor(ts, dtype=torch.float64, device=pb.DEV)
    c = cfat(tts)
    Gm = torch.zeros((len(ts), len(ts)), dtype=torch.complex128, device=pb.DEV)
    for c0 in range(0, len(fit), 262144):
        sl = fit[c0:c0 + 262144]
        rw = g(tts, sl).T * st[None, :]
        for wv, rows in [(w_water[sl], rw), (w_fat[sl], rw * c[None, :])]:
            r_ = rows * torch.sqrt(wv).to(torch.float32)[:, None]
            Gm += (r_.conj().T @ r_).to(torch.complex128)
    ev, V = torch.linalg.eigh(Gm)
    return torch.clamp(ev.flip(0).real, min=0).cpu().numpy(), V.flip(1), tts, st


def coeffs(g, tts, st, Vl, sl, fat=None):
    rows = g(tts, sl).T * st[None, :]
    if fat is not None:
        rows = rows * fat[None, :]
    return (rows @ Vl).T  # L x |sl|


def union_interp(g, cfat, D_w, D_f, fit, w_water, w_fat):
    """least-squares interpolators over the union rows: water g(v, t) ~ a D_w(v), fat g c(t) ~ a D_f(v)"""
    sw = torch.sqrt(w_water[fit]).to(torch.complex128); sf = torch.sqrt(w_fat[fit]).to(torch.complex128)
    B = torch.cat([(D_w[:, fit].to(torch.complex128) * sw[None, :]).T, (D_f[:, fit].to(torch.complex128) * sf[None, :]).T], 0)
    Gr = B.conj().T @ B
    Gr += 1e-10 * torch.trace(Gr).real / Gr.shape[0] * torch.eye(Gr.shape[0], dtype=torch.complex128, device=pb.DEV)
    Gi = torch.linalg.inv(Gr)

    def interp(tau):
        out = []
        for c0 in range(0, len(tau), 32):
            t = tau[c0:c0 + 32]
            yw = g(t, fit).to(torch.complex128)
            yf = yw * cfat(t).to(torch.complex128)[:, None]
            y = torch.cat([yw * sw[None, :], yf * sf[None, :]], 1)
            out.append(((y @ B.conj()) @ Gi.T).to(torch.complex64))
        return torch.cat(out, 0)
    return interp


if __name__ == '__main__':
    torch.set_num_threads(6)
    obj0 = pb.Obj()
    for dname in ['v2', 'current']:
        fn = os.path.join(OUT, f'fat_union_{dname}.json')
        res = json.load(open(fn)) if os.path.exists(fn) else {}
        obj = pb.use_truth(obj0, dname)
        des = pb.Design(dname)
        te = des.te
        nsel = des.sample_select(kc=150.0, stride=8)
        ros = pb.scan_readouts(48, offset=7)
        G = json.load(open(os.path.join(OUT, f'global_{dname}.json')))
        n_sel = np.array(G['energy']['n']); e_sel = np.array(G['energy']['E'])
        E_all = np.exp(np.interp(np.arange(des.na), n_sel, np.log(np.maximum(e_sel, 1e-30))))
        wn = seg.noise_time_weights(E_all, des.kabs, obj.nc, obj.sigma)
        rmean = pb.scan_readouts(400, offset=1)['R']
        wbar = np.mean([des.conc_w(R) for R in rmean], 0)
        t_wbar = torch.tensor(wbar, dtype=torch.float64, device=pb.DEV)
        m4 = pb.monos(obj.t_pos)
        nv = obj.N
        xs = seg.smooth_mag(obj, obj.x_tot)
        wv = xs.astype(np.float64) ** 2 * obj.cE; wv = wv / np.median(wv[wv > 0])
        ffs = torch.tensor(obj.ff, device=pb.DEV)
        fmin = obj.ff.min(1); fmax = obj.ff.max(1); fmean = obj.ff.mean(1)
        w_water = torch.tensor(wv * np.maximum(1 - fmin, 0.0) ** 2 + 1e-12, dtype=torch.float64, device=pb.DEV)
        w_fat = torch.tensor(wv * fmax ** 2 + 1e-12, dtype=torch.float64, device=pb.DEV)
        fit = torch.arange(0, nv, 3, device=pb.DEV)

        def g(t, sl):
            n = torch.clamp(torch.round((t - des.tau[0]) / (des.tau[1] - des.tau[0])).long(), 0, des.na - 1)
            return pb.polar(t[:, None] * obj.t_om[sl][None, :] + t_wbar[n] @ m4[sl].T)

        cfat = pb.fat_c
        ts, q = seg.seg_time_grid(des.tau, wn, 96)
        ev, V, tts, st = union_family(obj, g, cfat, ts, q, w_water, w_fat, fit)
        tail = lambda L: float(np.sqrt(ev[L:].sum() / ev.sum()) * np.sqrt(wn.mean()))
        Lr = next(L for L in range(2, 60) if 1.5 * tail(L) <= 0.2)
        res['union_rule_L'] = Lr
        res['union_tail_pred'] = {L: tail(L) for L in range(8, 22)}
        pb.log(f'{dname}: union rule L {Lr}')
        tg = torch.tensor(des.tau, dtype=torch.float64, device=pb.DEV)

        def model_err(Acard, Dcard, cnodes, fmap):
            """C_lin-free error: x [(1 - f) g + f g c] - sum_l a'_l D'_l [(1 - f) + f c_l] x, f = frame map or a fixed map"""
            def fn(ctx, sl):
                nidx = torch.tensor(ctx.n, device=pb.DEV)
                f = ffs[sl, ctx.frame] if fmap == 'frame' else torch.tensor(fmean, device=pb.DEV)[sl]
                gt_ = g(ctx.tau, sl)
                c = cfat(ctx.tau)[:, None]
                target = gt_ * ((1 - f)[None, :] + f[None, :] * c)
                A = Acard[nidx]
                Dl = Dcard[:, sl]
                if fmap == 'frame':
                    mdl = A @ (Dl * (1 - f)[None, :]) + (A * cnodes[None, :]) @ (Dl * f[None, :])
                else:
                    mdl = A @ Dl
                dq = (ctx.w4 - t_wbar[nidx]) @ m4[sl].T
                return pb.polar(dq) * (target - mdl) * obj.t_x[sl][None, :]
            return fn

        efs = {}
        info = {}
        for L in sorted(set([Lr - 2, Lr - 1, Lr, Lr + 1, 12, 14])):
            key = f'union_svdn_L{L}'
            if key in res:
                continue
            Vl = V[:, :L].to(torch.complex64)
            D_w = torch.zeros((L, nv), dtype=torch.complex64, device=pb.DEV)
            for c0 in range(0, nv, 262144):
                sl = torch.arange(c0, min(c0 + 262144, nv), device=pb.DEV)
                D_w[:, c0:c0 + len(sl)] = coeffs(g, tts, st, Vl, sl)
            c_ts = cfat(tts)
            D_f = torch.zeros_like(D_w)
            for c0 in range(0, nv, 262144):
                sl = torch.arange(c0, min(c0 + 262144, nv), device=pb.DEV)
                D_f[:, c0:c0 + len(sl)] = coeffs(g, tts, st, Vl, sl, fat=c_ts)
            interp = union_interp(g, cfat, D_w, D_f, fit, w_water, w_fat)
            b = seg.Basis(D_w, interp, 'union', L)
            nodes_t, cond, piv = seg.deim_nodes(b, tg)
            A = interp(tg)
            Ap = A[torch.tensor(piv, device=pb.DEV)]
            Api = torch.linalg.inv(Ap.to(torch.complex128)).to(torch.complex64)
            Acard = A @ Api; Dcard = Ap @ D_w
            cn = cfat(torch.tensor(np.array(nodes_t), dtype=torch.float64, device=pb.DEV))
            efs[key] = model_err(Acard, Dcard, cn, 'frame')
            info[key] = dict(nodes_ms=((np.array(nodes_t) - te) * 1e3).tolist(), cond=float(cond))
            del D_f
        # fixed uniform nodes with the union LS interpolators (the per-frame path today, energy weights)
        for L in [14, 15, 16]:
            key = f'fixed_union_L{L}'
            if key in res:
                continue
            tl = torch.tensor(np.linspace(des.tau[0], des.tau[-1], L), dtype=torch.float64, device=pb.DEV)
            D_w = torch.zeros((L, nv), dtype=torch.complex64, device=pb.DEV)
            for c0 in range(0, nv, 262144):
                sl = torch.arange(c0, min(c0 + 262144, nv), device=pb.DEV)
                D_w[:, c0:c0 + len(sl)] = g(tl, sl)
            D_f = D_w * cfat(tl)[:, None]
            interp = union_interp(g, cfat, D_w, D_f, fit, w_water, w_fat)
            efs[key] = model_err(interp(tg), D_w, cfat(tl), 'frame')
        t0 = time.time()
        if efs:
            E, ns = pb.eval_terms(obj, des, ros, efs, nsel=nsel)
            for k, e in E.items():
                res[k] = pb.R_split(obj, e, ns, des)
                res[k].update(info.get(k, {}))
        json.dump(res, open(fn, 'w'), indent=1)
        pb.log(f'{dname} ({time.time() - t0:.0f}s): ' + ', '.join(f'{k} {v["R"]:.3g}' for k, v in res.items() if isinstance(v, dict) and 'R' in v))
    pb.log('done')
