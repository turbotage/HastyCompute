"""Truth-based image (truth_image.py: band-limited static tissue + blood frame 0) and the per-voxel effective decay
of its sources, against the measured noise-free data of the simulated current-design scan.

Model: y_c(j) = sum_v S_c(v) e^{-i k_j.r_v} e^{-i wbar_v tau_j} e^{-i w_ro(tau_j).m(r_v)} [x_w H_w(v, tau) + x_f H_f(v, tau) c(tau)
         + x_b e^{-R2*_b tau}],  tau since excitation, H from the sources (exact intravoxel T2* and field spread).
Writes calib_truth.json, figs/calib_truth.png. The complex scale s (|k| < 30 rad/m fit) should be ~1 if the image,
the coil maps and the steady-state weights are consistent with the simulator.
"""
import json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import pb

OUT = pb.OUT


def model_truth(obj, gt, conc=True, r2b=31.0, blood=True):
    m = pb.monos(obj.t_pos)
    tw, Hw = gt['water'][0], gt['water'][1]
    Hf = gt['fat'][1]

    def f(ctx, sl):
        ph = ctx.tau[:, None] * obj.t_om[sl][None, :]
        if conc:
            ph = ph + ctx.w4 @ m[sl].T
        z = pb.polar(ph)
        hw = pb.interp_t(tw, Hw[:, sl], ctx.tau)
        hf = pb.interp_t(tw, Hf[:, sl], ctx.tau)
        c = pb.fat_c(ctx.tau)[:, None]
        img = obj.t_xw[sl][None, :] * hw + obj.t_xf[sl][None, :] * hf * c
        if blood:
            img = img + obj.t_xb[sl][None, :] * torch.exp(-r2b * ctx.tau.to(torch.float32))[:, None]
        return z * img
    return f


def model_data(obj, des, ros, nsel, f):
    n, w = nsel
    Ym = []
    tv = torch.arange(obj.N, device=pb.DEV)
    for r in range(len(ros['idx'])):
        ctx = pb.make_ctx(des, ros['R'][r], n, w, frame=ros['frame'][r], enc=ros['enc'][r])
        Y = torch.zeros((len(n), obj.nc), dtype=torch.complex64, device=pb.DEV)
        ch = int(1.2e8 // len(n))
        for c0 in range(0, obj.N, ch):
            sl = tv[c0:c0 + ch]
            Y += (pb.polar(ctx.k @ obj.t_pos[sl].T) * f(ctx, sl)) @ obj.t_ST[sl]
        Ym.append(Y.cpu().numpy())
    return np.array(Ym)


if __name__ == '__main__':
    torch.set_num_threads(6)
    obj = pb.use_truth(pb.Obj(), 'current')
    gt = pb.geff_tables(obj, tr=0.01129)
    des = pb.Design('current')
    ros = pb.scan_readouts(48, offset=3)
    nsel = des.sample_select(kc=150.0, stride=4)
    n, w = nsel
    dcl = np.load(os.path.join(pb.SCAN, 'data_clean.npy'), mmap_mode='r')
    Yd = np.array([np.asarray(dcl[i])[n].astype(np.complex128) @ obj.Wh.T for i in ros['idx']])
    half = np.arange(len(n)) < np.argmax(des.kabs[n])
    kq = des.kabs[n] / des.kmax
    c = des.kabs[n] < 30
    rows, prof = [], {}
    for name, f in [('truth', model_truth(obj, gt)), ('truth_noblood', model_truth(obj, gt, blood=False))]:
        Ym = model_data(obj, des, ros, nsel, f)
        s = (np.conj(Ym[:, c]) * Yd[:, c]).sum() / (np.abs(Ym[:, c]) ** 2).sum()
        em1 = (np.abs(Ym) ** 2).sum(2).mean(0)
        em = em1 * abs(s) ** 2
        ed = (np.abs(Yd) ** 2).sum(2).mean(0)
        res = (np.abs(Yd - s * Ym) ** 2).sum(2).mean(0)
        res1 = (np.abs(Yd - Ym) ** 2).sum(2).mean(0)
        prof[name] = (em1, ed, res)
        pb.log(f'{name}: scale {s:.4f} (|s| {abs(s):.4f})')
        for leg, msk in [('out', half), ('return', ~half)]:
            for a, b in [(0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]:
                mm = msk & (kq >= a) & (kq < b)
                rows.append(dict(model=name, scale=[float(s.real), float(s.imag)], leg=leg, band=f'{a}-{b}',
                                 energy_ratio_unscaled=float((w[mm] * em1[mm]).sum() / (w[mm] * ed[mm]).sum()),
                                 nres_scaled=float(np.sqrt((w[mm] * res[mm]).sum() / (w[mm] * ed[mm]).sum())),
                                 nres_unscaled=float(np.sqrt((w[mm] * res1[mm]).sum() / (w[mm] * ed[mm]).sum())),
                                 meas_over_noise=float((w[mm] * ed[mm]).sum() / w[mm].sum() / (obj.nc * obj.sigma ** 2))))
                print(rows[-1])
    json.dump(dict(note=__doc__, bands=rows), open(OUT + '/calib_truth.json', 'w'), indent=1)
    em1, ed, res = prof['truth']
    np.savez(OUT + '/calib_truth_profile.npz', n=n, w=w, tau=des.tau[n], kq=kq, em=em1, ed=ed, res=res)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].semilogy(des.tau[n] * 1e3, ed / (obj.nc * obj.sigma ** 2), '.', ms=2, label='measured (data_clean, whitened)')
    ax[0].semilogy(des.tau[n] * 1e3, em1 / (obj.nc * obj.sigma ** 2), '.', ms=2, label='truth-image model (unscaled)')
    ax[0].semilogy(des.tau[n] * 1e3, res / (obj.nc * obj.sigma ** 2), '.', ms=2, label='|meas - s model|^2')
    ax[0].set_xlabel('tau (ms)'); ax[0].set_ylabel('energy / (C sigma^2)'); ax[0].legend(fontsize=8); ax[0].set_title('current design, 48 readouts')
    ax[1].semilogy(kq[half], (em1 / ed)[half], '.', ms=2, label='out'); ax[1].semilogy(kq[~half], (em1 / ed)[~half], '.', ms=2, label='return')
    ax[1].set_xlabel('|k| / kmax'); ax[1].set_ylabel('model / measured energy (unscaled)'); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT + '/figs/calib_truth.png', dpi=110)
