"""Noise-referenced data error R of a trajectory / k0-phase error, linearised: per readout r and selected ADC sample n
the 4x4 Gram G_rn = Re(D^H D) of the data derivatives D_a(n, c) = sum_v S_c(v) X(v, tau_n) C(v, tau_n) (-i r_a)
e^{-i k_n.r_v} (a = x, y, z) and D_0 = the same with (-i) (a global phase). For a residual q_rn = (dk_x, dk_y, dk_z,
dphi) (rad/m, rad):
    R^2 = sum_n w_n mean_r q_rn^T G_rn q_rn / (na C sigma^2)       (pb.R_of, the budget's metric)
Object, image, coils, noise, readouts, sample selection: as research/experiments/readout_physics (pb.Obj /
pb.use_truth('v2') / pb.scan_readouts(48, offset=7) / sample_select(kc=150, stride=8)), imported, not modified.
Carrier (simplified, see check_t10): X = (x_w + x_f c_fat(tau)) e^{-R2*_table tau} + x_b e^{-R2_blood tau},
C = e^{-i omega_v tau} (B0 map); no concomitant phase, no intravoxel H table (needs ~2 GB VRAM).
Also the exact (non-linearised) error energy of the budget's T10 cases (girf_err.npy rotated) for the calibration.

    rgram.py [--device cuda] [--chunk 10000]   -> <OUT>/gram_v2.npz
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('HASTY_REPO', '/home/turbotage/Documents/GitHub/HastyCompute')
OUT = os.path.join(REPO, 'data/hasty/seq/studies/girf_sensitivity_2026-11-11')
sys.path.insert(0, os.path.join(REPO, 'research/experiments/readout_physics'))
sys.path.insert(0, HERE)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def select(v, kc=150.0, stride=8):
    """pb.Design.sample_select on the reconstruction window, the same rule on the navigator window"""
    def one(i0, i1):
        m = i1 - i0
        sel = np.zeros(m, bool); sel[::stride] = True; sel[-1] = True
        sel |= v.kabs[i0:i1] < kc
        n = np.flatnonzero(sel)
        w = np.zeros(len(n)); d = np.diff(n).astype(float)
        w[:-1] += 0.5 * d; w[1:] += 0.5 * d; w[0] += 0.5; w[-1] += 0.5
        return n + i0, w
    n1, w1 = one(0, v.na)
    n2, w2 = one(v.na, len(v.t))
    return np.r_[n1, n2], np.r_[w1, w2], np.r_[np.ones(len(n1), bool), np.zeros(len(n2), bool)]


def region_vox(obj, name):
    """copy of run_global.region_vox (vessels / myocardium / blood), dilated by 1 voxel"""
    from scipy.ndimage import binary_dilation
    nx, ny, nz = obj.dims
    if name == 'blood':
        m = obj.lumen > 0
    else:
        lab = obj.labels2(0)
        m = np.isin(lab, [24, 25, 26, 27]) if name == 'vessels' else lab == 18
    g = np.zeros(nx * ny * nz, bool); g[obj.idx[m]] = True
    if name != 'myocardium':
        g = binary_dilation(g.reshape((nx, ny, nz), order='F'), iterations=1).ravel(order='F')
    return np.flatnonzero(g[obj.idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--chunk', type=int, default=10000)
    ap.add_argument('--nr', type=int, default=48)
    ap.add_argument('--region', default='vessels')
    a = ap.parse_args()
    torch.set_num_threads(6)
    import pb, v2seq, gmodel as gm
    dev = torch.device(a.device)
    pb.DEV = torch.device('cpu')  # keep pb's object tensors on the CPU; chunks go to `dev`
    fn = os.path.join(OUT, 'gram_v2.npz')
    obj = pb.use_truth(pb.Obj(), 'v2')
    v = v2seq.V2()
    rots = v2seq.scan_rotations(a.nr, 7, 2)
    Rs = rots['R'][:, -1]
    n, w, isrec = select(v)
    ns = len(n)
    tau = v.te + v.t[n]
    log(f'{obj.N} voxels, {obj.nc} coils, sigma {obj.sigma:.2f}; {ns} samples ({isrec.sum()} recon), {a.nr} readouts')
    # budget T10 cases: girf_err.npy (rotation of the seqdesign search set) rotated with each readout, as run_global
    ge = np.load(v2seq.RUN + '/girf_err.npy') * 2 * np.pi
    tg = np.arange(len(ge)) * 1e-5
    dk_t10 = np.stack([np.interp(v.t[n], tg, ge[:, c]) for c in range(3)], 1)
    vox_reg = region_vox(obj, a.region) if a.region else np.zeros(0, int)
    log(f'region {a.region}: {len(vox_reg)} voxels')
    in_reg = np.zeros(obj.N, bool); in_reg[vox_reg] = True
    # per-readout sample tensors
    kk = torch.tensor(np.stack([v.k_base[n] @ R.T for R in Rs]), dtype=torch.float64, device=dev)  # nr x ns x 3
    dkt = torch.tensor(np.stack([dk_t10 @ R.T for R in Rs]), dtype=torch.float64, device=dev)
    t_tau = torch.tensor(tau, dtype=torch.float64, device=dev)
    cf = pb.fat_c(t_tau.cpu()) if False else None
    om_f = 2 * np.pi * pb.GBAR * 3.0 * pb.FAT_PPM * 1e-6
    cfat = torch.tensor((pb.FAT_AMP[None, :] * np.exp(-1j * tau[:, None] * om_f[None, :])).sum(1), dtype=torch.complex64, device=dev)
    eb = torch.exp(-31.0 * t_tau).to(torch.float32)
    nr, C = a.nr, obj.nc
    D = torch.zeros((nr, ns, 4, C), dtype=torch.complex64, device=dev)
    Dr = torch.zeros((nr, ns, 4, C), dtype=torch.complex64, device=dev)
    Ex = torch.zeros((2, nr, ns, C), dtype=torch.complex64, device=dev)
    t0 = time.time()
    for c0 in range(0, obj.N, a.chunk):
        sl = slice(c0, min(c0 + a.chunk, obj.N))
        pos = torch.tensor(obj.pos[sl], dtype=torch.float64, device=dev)
        S = torch.tensor(obj.S[:, sl].T.copy(), dtype=torch.complex64, device=dev)
        Sr = S * torch.tensor(in_reg[sl], dtype=torch.float32, device=dev)[:, None]
        xw = torch.tensor(obj.xw[sl], device=dev); xf = torch.tensor(obj.xf[sl], device=dev); xb = torch.tensor(obj.xb[sl], device=dev)
        om = torch.tensor(obj.om[sl], dtype=torch.float64, device=dev)
        r2 = torch.tensor(obj.r2[sl], dtype=torch.float32, device=dev)
        dec = torch.exp(-r2[None, :] * t_tau.to(torch.float32)[:, None])  # ns x V
        X = dec * (xw[None, :] + xf[None, :] * cfat[:, None]) + xb[None, :] * eb[:, None]
        posf = pos.to(torch.float32)
        anyreg = bool(in_reg[sl].any())
        for r in range(nr):
            ph = kk[r] @ pos.T + t_tau[:, None] * om[None, :]
            B = pb.polar(ph) * X  # ns x V
            D[r, :, 3] += -1j * (B @ S)
            for q in range(3):
                Bq = B * posf[:, q][None, :]
                D[r, :, q] += -1j * (Bq @ S)
                if anyreg:
                    Dr[r, :, q] += -1j * (Bq @ Sr)
            if anyreg:
                Dr[r, :, 3] += -1j * (B @ Sr)
            for j, s in enumerate((1.0, 0.05)):
                e = pb.polar((dkt[r] * s) @ pos.T) - 1
                Ex[j, r] += (B * e) @ S
        if (c0 // a.chunk) % 20 == 0:
            el = time.time() - t0
            log(f'  voxels {sl.stop}/{obj.N} ({el:.0f}s, eta {el / sl.stop * (obj.N - sl.stop):.0f}s)')
    G = torch.einsum('rnac,rnbc->rnab', D.conj(), D).real.cpu().numpy()
    Gr = torch.einsum('rnac,rnbc->rnab', Dr.conj(), Dr).real.cpu().numpy()
    Eex = (Ex.abs() ** 2).sum(-1).cpu().numpy()  # 2 x nr x ns
    np.savez(fn, n=n, w=w, isrec=isrec, G=G, Gr=Gr, Eex=Eex, na=v.na, nc=C, sigma=obj.sigma, N=obj.N, N_reg=len(vox_reg), region=a.region,
             idx=rots['idx'][:, -1])
    # calibration against the budget's exact T10 (global_v2.json)
    rec = isrec
    Rex = [float(np.sqrt((w[rec] * Eex[j][:, rec].mean(0)).sum() / (v.na * C * obj.sigma ** 2))) for j in range(2)]
    q = np.concatenate([np.stack([dk_t10 @ R.T for R in Rs]), np.zeros((nr, ns, 1))], -1) * 0.05
    Rlin = float(np.sqrt((w[rec] * np.einsum('rna,rnab,rnb->rn', q, G, q)[:, rec].mean(0)).sum() / (v.na * C * obj.sigma ** 2)))
    ref = json.load(open(os.path.join(pb.OUT, 'global_v2.json')))['T10']
    res = dict(R_exact_simplified_carrier=dict(omission=Rex[0], residual_5pct=Rex[1]), R_linearised_5pct=Rlin,
               R_budget=dict(omission=ref['girf_omission']['R'], residual_5pct=ref['girf_5pct_residual']['R']), seconds=time.time() - t0)
    json.dump(res, open(os.path.join(OUT, 'gram_v2_check.json'), 'w'), indent=1)
    log(json.dumps(res))


if __name__ == '__main__':
    main()
