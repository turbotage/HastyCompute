"""Calibrate the image x (warm start, encode 0) to data units and check the voxel-level full model against the
measured (noise-free) data of the simulated current-design scan.

Model: y_c(j) = sum_v S_c(v) x_v e^{-i k_j.r_v} e^{-i w tau_j} e^{-i w_ro(tau_j).m(r_v)} e^{-R2*(tau_j - TE)} [(1 - f) + f c(tau_j)]
(B0, readout concomitant part, R2* table, six-peak fat with the readout's frame map). Complex scale s fitted at
|k| < 30 rad/m. Writes calib.json (s, energy ratio per |k| band, normalised residual per band) and figures.
"""
import json, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import pb

OUT = os.path.join(pb.ROOT, 'physics_budget_2026-11-10')
os.makedirs(OUT + '/figs', exist_ok=True)


def model_full(obj, ff_frame=True, r2s=True, conc=True):
    m = pb.monos(obj.t_pos)
    ffs = torch.tensor(obj.ff, device=pb.DEV)

    def f(ctx, sl):
        ph = ctx.tau[:, None] * obj.t_om[sl][None, :]
        if conc:
            ph = ph + ctx.w4 @ m[sl].T
        z = pb.polar(ph)
        if r2s:
            z = z * torch.exp(-obj.t_r2[sl][None, :] * (ctx.tau - ctx.des.te).to(torch.float32)[:, None])
        fr = ctx.frame if ff_frame else 0
        ff = ffs[sl, fr][None, :]
        c = pb.fat_c(ctx.tau)[:, None]
        z = z * ((1 - ff) + ff * c)
        return z * obj.t_x[sl][None, :]
    return f


def model_data(obj, des, ros, nsel, f):
    n, w = nsel
    Ym = []
    tv = torch.arange(obj.N, device=pb.DEV)
    for r in range(len(ros['idx'])):
        ctx = pb.make_ctx(des, ros['R'][r], n, w, frame=ros['frame'][r], enc=ros['enc'][r])
        Y = torch.zeros((len(n), obj.nc), dtype=torch.complex64, device=pb.DEV)
        ch = int(1.6e8 // len(n))
        for c0 in range(0, obj.N, ch):
            sl = tv[c0:c0 + ch]
            Y += (pb.polar(ctx.k @ obj.t_pos[sl].T) * f(ctx, sl)) @ obj.t_ST[sl]
        Ym.append(Y.cpu().numpy())
    return np.array(Ym)


def radial_filter(obj, H_k, H_v):
    """x -> IFFT(H(|k|) FFT(x)) on the full 2 mm grid"""
    nx, ny, nz = obj.dims
    g = np.zeros(nx * ny * nz, np.complex64); g[obj.idx] = obj.x
    G = np.fft.fftn(g.reshape((nx, ny, nz), order='F'))
    kk = [2 * np.pi * np.fft.fftfreq(m, obj.res) for m in (nx, ny, nz)]
    K = np.sqrt(kk[0][:, None, None] ** 2 + kk[1][None, :, None] ** 2 + kk[2][None, None, :] ** 2)
    G *= np.interp(K, H_k, H_v).astype(np.float32)
    return np.fft.ifftn(G).ravel(order='F')[obj.idx].astype(np.complex64)


def profile(obj, des, ros, nsel, Ym, Yd, s):
    n, w = nsel
    kq = des.kabs[n] / des.kmax
    em = (np.abs(Ym) ** 2).sum(2).mean(0) * abs(s) ** 2
    ed = (np.abs(Yd) ** 2).sum(2).mean(0)
    res = (np.abs(Yd - s * Ym) ** 2).sum(2).mean(0)
    return kq, em, ed, res


if __name__ == '__main__':
    torch.set_num_threads(6)
    obj = pb.Obj()
    des = pb.Design('current')
    ros = pb.scan_readouts(48, offset=3)
    nsel = des.sample_select(kc=150.0, stride=4)
    n, w = nsel
    dcl = np.load(os.path.join(pb.SCAN, 'data_clean.npy'), mmap_mode='r')
    Yd = np.array([np.asarray(dcl[i])[n].astype(np.complex128) @ obj.Wh.T for i in ros['idx']])
    f = model_full(obj)
    half = np.arange(len(n)) < np.argmax(des.kabs[n])
    c = des.kabs[n] < 30
    hist = []
    for it in range(3):
        Ym = model_data(obj, des, ros, nsel, f)
        s = (np.conj(Ym[:, c]) * Yd[:, c]).sum() / (np.abs(Ym[:, c]) ** 2).sum()
        kq, em, ed, res = profile(obj, des, ros, nsel, Ym, Yd, s)
        hist.append(dict(it=it, kq=kq, em=em, ed=ed, res=res, s=s))
        pb.log(f'iteration {it}: scale {s:.4f}')
        if it == 2:
            break
        # out-leg energy ratio in |k| bins -> radial amplitude filter of the image (smoothed in log)
        edges = np.concatenate([[0, 5, 10, 20, 30, 45, 60, 80], np.geomspace(100, des.kmax * 1.001, 24)])
        kc_, rat = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m = half & (des.kabs[n] >= a) & (des.kabs[n] < b)
            if m.sum() >= 2:
                kc_.append(0.5 * (a + b)); rat.append((w[m] * ed[m]).sum() / (w[m] * em[m]).sum())
        kc_ = np.array(kc_); lr = np.log(np.array(rat))
        lr = np.convolve(np.pad(lr, 1, mode='edge'), np.ones(3) / 3, mode='valid')
        Hv = np.exp(0.5 * lr)
        Hk = np.concatenate([[0], kc_, [10 * des.kmax]]); Hv = np.concatenate([[Hv[0]], Hv, [Hv[-1]]])
        Hv = Hv / Hv[0]
        obj.set_x(radial_filter(obj, Hk, Hv) * (s / abs(s)) ** 0)
        hist[-1]['Hk'] = Hk; hist[-1]['Hv'] = Hv
        pb.log('filter', np.round(Hv, 3))
    obj.set_x(obj.x * s)
    np.save(OUT + '/x_cal.npy', obj.x)
    bands = [(0, 0.02), (0.02, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]
    rows = []
    for h in [hist[0], hist[-1]]:
        kq, em, ed, res = h['kq'], h['em'], h['ed'], h['res']
        for leg, msk in [('out', half), ('return', ~half)]:
            for a, b in bands:
                m = msk & (kq >= a) & (kq < b)
                rows.append(dict(x='warm' if h is hist[0] else 'filtered', leg=leg, band=f'{a}-{b}', energy_ratio_model_over_meas=float((w[m] * em[m]).sum() / (w[m] * ed[m]).sum()),
                                 nres=float(np.sqrt((w[m] * res[m]).sum() / (w[m] * ed[m]).sum())), meas_energy_over_noise=float((w[m] * ed[m]).sum() / w[m].sum() / (obj.nc * obj.sigma ** 2))))
    out = dict(note='x_cal = warm_full encode 0 (dilated truth support), radial amplitude filter H(|k|) from the out-leg model/measured energy ratio (2 passes), times the complex scale fitted at |k| < 30 rad/m',
               scale_first=[float(hist[0]['s'].real), float(hist[0]['s'].imag)], scale_final=[float(s.real), float(s.imag)], sigma=obj.sigma,
               filter_k=[list(map(float, h['Hk'])) for h in hist[:-1]], filter_v=[list(map(float, h['Hv'])) for h in hist[:-1]], bands=rows)
    json.dump(out, open(OUT + '/calib.json', 'w'), indent=1)
    np.savez(OUT + '/calib_profile.npz', n=n, w=w, tau=des.tau[n], **{f'{k}{i}': h[k] for i, h in enumerate(hist) for k in ('kq', 'em', 'ed', 'res')})
    for r_ in rows:
        print(r_)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, h in enumerate([hist[0], hist[-1]]):
        ax[i].semilogy(des.tau[n] * 1e3, h['ed'] / (obj.nc * obj.sigma ** 2), '.', ms=2, label='measured (data_clean, whitened)')
        ax[i].semilogy(des.tau[n] * 1e3, h['em'] / (obj.nc * obj.sigma ** 2), '.', ms=2, label='model')
        ax[i].semilogy(des.tau[n] * 1e3, h['res'] / (obj.nc * obj.sigma ** 2), '.', ms=2, label='|meas - model|^2')
        ax[i].set_xlabel('tau (ms)'); ax[i].set_ylabel('energy / (C sigma^2)'); ax[i].legend(fontsize=8)
        ax[i].set_title(['warm start x', 'radially filtered x (x_cal)'][i] + ', 48 readouts')
    for i, h in enumerate([hist[0], hist[-1]]):
        ax[2].semilogy(h['kq'][half], (h['em'] / h['ed'])[half], '.', ms=2, label=['warm', 'filtered'][i] + ' out')
        ax[2].semilogy(h['kq'][~half], (h['em'] / h['ed'])[~half], '.', ms=2, label=['warm', 'filtered'][i] + ' return')
    ax[2].set_xlabel('|k| / kmax'); ax[2].set_ylabel('model / measured energy'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT + '/figs/calib.png', dpi=110)
