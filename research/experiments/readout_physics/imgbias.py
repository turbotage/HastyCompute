"""Image-space consequence of the unmodelled in-readout displacement: velocity bias in the vessels.

Ideal fully sampled least-squares reconstruction on the 2 mm grid of a box around the heart and great vessels: every
2 mm DFT frequency inside the sampled ball is observed with the readout displacement of the design at that |k|,
X_e(k) = sum_v x_v e^{i phi_e(v)} e^{-i k.(r_v + v_v t1(|k|))} (current design: the out and the return visit of |k|,
averaged with their sampling densities); x_e = IDFT. Encodes: reference + one per axis, phi_e = (pi / VENC) v_a with
VENC 3 m/s (no wraps; the bias is a mixture effect, linear in the encode phase for small phases). v_hat_a =
arg(x_a conj(x_0)) VENC / pi. Bias = v_hat(with displacement) - v_hat(static data): the partial-volume / band-limit
part common to both cancels. Blood: band-limited 2 mm image of blood frame 5 f with the source velocity per voxel.

    imgbias.py [--phases systole,diastole] [--designs current,v2]
"""
import json, os, sys, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import pb, flow, warp_eval as we
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

OUT = pb.OUT
VENC = 3.0


def t1_of_k(des, kq):
    """per |k| (rad/m): list of (t1, weight) for the visits of that radius over the recon window"""
    k = des.kabs; t1 = des.t1
    i_max = int(np.argmax(k))
    legs = [(slice(0, i_max + 1), 1)]
    if i_max < des.na - 10:
        legs.append((slice(i_max, des.na), -1))
    out = []
    for sl, sgn in legs:
        kk = k[sl]; tt = t1[sl]; tau = des.tau[sl]
        if sgn < 0:
            kk, tt, tau = kk[::-1], tt[::-1], tau[::-1]
        kk_m = np.maximum.accumulate(kk)  # monotone envelope
        t_of = np.interp(kq, kk_m, tt, right=np.nan)
        dtdk = np.abs(np.gradient(tau, kk_m + 1e-9 * np.arange(len(kk_m))))
        w = np.interp(kq, kk_m, dtdk, right=0)
        out.append((t_of, w))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phases', default='systole,diastole')
    ap.add_argument('--designs', default='current,v2')
    a = ap.parse_args()
    torch.set_num_threads(6)
    obj0 = pb.Obj()
    res = {}
    fn = os.path.join(OUT, 'imgbias.json')
    if os.path.exists(fn):
        res = json.load(open(fn))
    for ph in a.phases.split(','):
        f = we.PH[ph]
        for dname in a.designs.split(','):
            key = f'{dname}_{ph}'
            if key in res:
                continue
            obj = pb.use_truth(obj0, dname)
            des = pb.Design(dname)
            ps = flow.blood_frame(obj, 5 * f, flow.TRS[dname])
            mv = we.build_model_voxels(obj, ps, 5 * f)
            # regions of the model voxels (warp_eval convention)
            dims = obj.dims; nvox = dims[0] * dims[1] * dims[2]
            from scipy.ndimage import gaussian_filter
            ampn = ps.t_a.real.cpu().numpy()
            names = ['coronary', 'vessels', 'chambers']
            Am = np.stack([gaussian_filter(np.bincount(obj.idx[ps.vox[ps.region_np[r]]], weights=ampn[ps.region_np[r]], minlength=nvox).reshape(dims, order='F'), 0.7).ravel(order='F')[obj.idx[mv['sel']]] for r in names], 1)
            Am[:, 0] *= 3
            lab_all = Am.argmax(1); has_all = Am.max(1) > 0
            # box around the coronary voxels (the heart surface: coronaries, aortic root, chambers' walls), 8 voxel margin;
            # every blood model voxel inside the box takes part (its displaced PSF reaches the coronaries)
            ijk_all = obj.ijk[mv['sel']]
            cor = has_all & (lab_all == 0)
            lo = np.maximum(ijk_all[cor].min(0) - 8, 0); hi = np.minimum(ijk_all[cor].max(0) + 9, np.array(obj.dims))
            inbox = np.all((ijk_all >= lo) & (ijk_all < hi), axis=1)
            nb = hi - lo
            loc = ijk_all[inbox] - lo
            x = mv['x'][inbox]; v = mv['v'][inbox]
            lab = lab_all[inbox]; has = has_all[inbox]
            pos = (loc * obj.res).astype(np.float64)
            # 2 mm DFT frequencies of the box inside the ball
            kk = [2 * np.pi * np.fft.fftfreq(n, obj.res) for n in nb]
            KX, KY, KZ = np.meshgrid(*kk, indexing='ij')
            K = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], 1)
            kq = np.linalg.norm(K, axis=1)
            inb = kq <= des.kmax
            legs = t1_of_k(des, kq[inb])
            pb.log(f'{key}: box {nb.tolist()}, {len(x)} voxels, {inb.sum()} k in the ball, {len(legs)} visit(s)')
            tK = torch.tensor(K[inb], dtype=torch.float64, device=pb.DEV)
            tpos = torch.tensor(pos, dtype=torch.float64, device=pb.DEV)
            tx = torch.tensor(x, dtype=torch.complex64, device=pb.DEV)
            tv = torch.tensor(v, dtype=torch.float64, device=pb.DEV)
            enc = [np.zeros(3), np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])]
            imgs = {}
            for mode in ['static', 'disp']:
                for e, ev in enumerate(enc):
                    phi = torch.tensor((np.pi / VENC) * (v @ ev), dtype=torch.float32, device=pb.DEV)
                    amp = tx * torch.polar(torch.ones_like(phi), phi)
                    Xk = torch.zeros(len(tK), dtype=torch.complex64, device=pb.DEV)
                    wsum = torch.zeros(len(tK), dtype=torch.float64, device=pb.DEV)
                    for (t1k, wk) in (legs if mode == 'disp' else [(np.zeros(inb.sum()), np.ones(inb.sum()))]):
                        okk = np.isfinite(t1k) & (wk > 0)
                        t1t = torch.tensor(np.nan_to_num(t1k), dtype=torch.float64, device=pb.DEV)
                        wt = torch.tensor(np.where(okk, wk, 0), dtype=torch.float64, device=pb.DEV)
                        acc = torch.zeros(len(tK), dtype=torch.complex64, device=pb.DEV)
                        for c0 in range(0, len(tK), 512):
                            kc = tK[c0:c0 + 512]
                            ph_ = kc @ tpos.T + t1t[c0:c0 + 512, None] * (kc @ tv.T)
                            acc[c0:c0 + 512] = pb.polar(ph_) @ amp
                        Xk += (acc * wt.to(torch.float32))
                        wsum += wt
                    Xk = Xk / torch.clamp(wsum, min=1e-30).to(torch.float32)
                    full = np.zeros(len(K), np.complex64); full[inb] = Xk.cpu().numpy()
                    img = np.fft.ifftn(full.reshape(tuple(nb)))
                    imgs[(mode, e)] = img
            # velocities at the model voxels
            def vh(mode):
                r = imgs[(mode, 0)]
                return np.stack([np.angle(imgs[(mode, a_)] * np.conj(r)) * VENC / np.pi for a_ in (1, 2, 3)], -1)
            vs, vd = vh('static'), vh('disp')
            mag = np.abs(imgs[('static', 0)])
            sel_l = tuple(loc.T)
            bias = (vd - vs)[sel_l]
            vtrue = v
            out = {}
            for i, r in enumerate(names):
                m = has & (lab == i)
                mm = m & (np.abs(x) > 0.2 * np.percentile(np.abs(x[m]), 95)) if m.any() else m
                if mm.sum() < 5:
                    continue
                b = np.linalg.norm(bias[mm], axis=1); sp = np.linalg.norm(vtrue[mm], axis=1)
                bpar = (bias[mm] * vtrue[mm]).sum(1) / np.maximum(sp, 1e-6)
                out[r] = dict(n=int(mm.sum()), speed_mm_s_50_95=(np.percentile(sp, [50, 95]) * 1e3).tolist(), bias_mm_s_50_95_99=(np.percentile(b, [50, 95, 99]) * 1e3).tolist(),
                              bias_along_v_mean_mm_s=float(bpar.mean() * 1e3), rel_bias_along_v_median=float(np.median(bpar / np.maximum(sp, 0.05))),
                              bias_rms_over_speed_rms=float(np.sqrt((b ** 2).mean() / (sp ** 2).mean())))
            res[key] = out
            pb.log(f'{key}: {out}')
            json.dump(res, open(fn, 'w'), indent=1)
            # figure: slice through the coronary voxels with the most coronary signal
            cm = has & (lab == 0)
            if cm.any():
                zc = int(np.median(loc[cm, 2]))
                fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
                sp_img = np.full(tuple(nb), np.nan); sp_img[sel_l] = np.linalg.norm(vtrue, axis=1)
                bi = np.full(tuple(nb), np.nan); bi[sel_l] = np.linalg.norm(bias, axis=1)
                ax[0].imshow(mag[:, :, zc].T, origin='lower', cmap='gray'); ax[0].set_title(f'{key}: |x| static, z = {zc + lo[2]}')
                im = ax[1].imshow(sp_img[:, :, zc].T * 1e3, origin='lower', cmap='viridis'); plt.colorbar(im, ax=ax[1]); ax[1].set_title('true speed (mm/s)')
                im = ax[2].imshow(bi[:, :, zc].T * 1e3, origin='lower', cmap='magma', vmax=np.nanpercentile(bi, 99) * 1e3); plt.colorbar(im, ax=ax[2]); ax[2].set_title('|velocity bias| from the unmodelled displacement (mm/s)')
                cz = cm & (loc[:, 2] == zc)
                for axx in ax:
                    axx.plot(loc[cz, 0], loc[cz, 1], 'c.', ms=2)
                fig.tight_layout(); fig.savefig(os.path.join(OUT, f'figs/imgbias_{key}.png'), dpi=100); plt.close(fig)
    pb.log('done')
