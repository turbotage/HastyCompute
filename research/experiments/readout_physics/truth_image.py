"""Truth-based images on the 2 mm reconstruction grid from the simulator's sources (static tissue, reference pose;
one Eulerian blood frame), band-limited: x = IDFT_2mm[ T(k_g) ], T(k) = sum_s a_s e^{-i k.(r_s - c)} at the 2 mm DFT
frequencies (the least-squares Dirac-voxel image of fully sampled data in the band). Sources lie on a 0.5 mm lattice
offset (0, 0, 0.245 mm) from the prescription centre: deposited on that lattice, FFT, central crop, offset phase,
IFFT (GPU).

a_s = PD x steady state sin a (1 - E1) / (1 - cos a E1) (12 deg, TR of the design, rect slab as ss_maps.py) for tissue
(water / fat by shift_ppm); blood: PD x steady state at its T1 x inflow factor (blood_inflow_gain x pd_scale of the
sim, = 1: the spin-history amplitude is not reproduced, [CHECK]).

    truth_image.py            -> <OUT>/truth_x_{current,v2}.npz (water, fat, blood; full 2 mm grid, complex64)
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import pb
sys.path.insert(0, os.path.join(pb.REPO, 'python/lib'))
from hastycompute import value_format as vf

OUT = os.path.join(pb.ROOT, 'physics_budget_2026-11-10')
TRS = {'current': 0.01129, 'v2': 0.0141}


def ss(t1, tr, flip=np.deg2rad(12.0)):
    e1 = np.exp(-tr / np.maximum(t1, 1e-4))
    return np.sin(flip) * (1 - e1) / (1 - np.cos(flip) * e1)


def bandlimit(lat_idx, amp, dims, off_z):
    """sources at 0.5 mm lattice indices (relative to the 2 mm grid's index frame, x4) -> 2 mm band-limited image"""
    nx, ny, nz = dims
    F = (4 * nx, 4 * ny, 4 * nz)
    g = torch.zeros(F[0] * F[1] * F[2], dtype=torch.float32, device=pb.DEV)
    q = torch.tensor(lat_idx[:, 0] + F[0] * (lat_idx[:, 1] + F[1] * lat_idx[:, 2]), device=pb.DEV)
    g.index_add_(0, q, torch.tensor(amp, dtype=torch.float32, device=pb.DEV))
    g = g.reshape(F[2], F[1], F[0])  # z, y, x (C order of the flat index)
    G = torch.fft.fftn(g.to(torch.complex64), dim=(0, 1, 2))
    del g
    # central crop: frequencies -n/2 .. n/2 - 1 of the 2 mm grid
    def crop(G, ax, n):
        Nf = G.shape[ax]
        ii = torch.tensor(np.r_[0:n // 2, Nf - n // 2:Nf], device=pb.DEV)
        return torch.index_select(G, ax, ii)
    G = crop(G, 0, nz); G = crop(G, 1, ny); G = crop(G, 2, nx)
    # lattice index 4 i <-> 2 mm voxel centre i; the sources sit at +off_z (lattice units) in z: phase e^{-i kz off}
    kz = 2 * np.pi * np.fft.fftfreq(nz)  # rad per 2 mm voxel
    G = G * torch.exp(-1j * torch.tensor(kz * off_z / 4.0, device=pb.DEV)).to(torch.complex64)[:, None, None]
    # DFT_2mm(x)(p) = DFT_fine(a)(p) e^{-2 pi i p off / (4 n)}: x ~ the 2 mm voxel sum of the sources (smooth object)
    x = torch.fft.ifftn(G, dim=(0, 1, 2))
    return x.permute(2, 1, 0).cpu().numpy()  # x, y, z


if __name__ == '__main__':
    t0 = time.time()
    v = vf.open_lazy(os.path.join(pb.ROOT, 'sim_flow_10min/sim_input.hv'))
    meta = json.load(open(os.path.join(pb.SCAN, 'meta.json')))
    dims = tuple(meta['recon_dims'])
    presc = v['presc']
    c = np.array(presc['center']); res = float(presc['res'])
    slab = presc['slab']
    t = v['tissue']
    pos = t['pos'][:].astype(np.float64)
    pd = t['pd'][:].astype(np.float64); t1 = t['t1'][:].astype(np.float64); sh = t['shift_ppm'][:]
    t2s = t['t2s'][:].astype(np.float64); fppm = t['field_ppm'][:].astype(np.float64)
    zrel = (pos - np.array(slab['center'])) @ np.array(slab['normal'])
    inslab = np.abs(zrel) <= 0.5 * slab['thk']
    # 0.5 mm lattice relative to the 2 mm grid: index 4 i + 0 <-> voxel centre i (x, y), z offset 0.245 mm
    lat = (pos - c) / 0.0005 + 4 * np.array([d // 2 for d in dims])
    off = np.median(lat - np.rint(lat), axis=0)
    pb.log('lattice offsets (0.5 mm units)', off)
    li = np.rint(lat - off).astype(np.int64)
    ok = np.all((li >= 0) & (li < 4 * np.array(dims)), axis=1) & inslab
    fat = sh != 0
    pb.log(f'{len(pos)} tissue sources, {ok.sum()} in the FOV and slab, fat {fat[ok].sum()}')
    out = {}
    for des, tr in TRS.items():
        a = pd * ss(t1, tr)
        xw = bandlimit(li[ok & ~fat], a[ok & ~fat], dims, off[2])
        xf = bandlimit(li[ok & fat], a[ok & fat], dims, off[2])
        out[des] = (xw, xf)
        pb.log(f'{des}: water / fat images done ({time.time() - t0:.0f}s)')
    # blood frame 0 (Eulerian), steady state at its T1 (inflow not modelled)
    b = v['blood'][0]
    bp = np.asarray(b['pos'][:]).astype(np.float64); bpd = np.asarray(b['pd'][:]).astype(np.float64); bt1 = np.asarray(b['t1'][:]).astype(np.float64)
    blat = (bp - c) / 0.0005 + 4 * np.array([d // 2 for d in dims])
    boff = np.median(blat - np.rint(blat), axis=0)
    bli = np.rint(blat - boff).astype(np.int64)
    bz = np.abs((bp - np.array(slab['center'])) @ np.array(slab['normal'])) <= 0.5 * slab['thk']
    bok = np.all((bli >= 0) & (bli < 4 * np.array(dims)), axis=1) & bz
    pb.log(f'blood frame 0: {len(bp)} sources, {bok.sum()} in FOV and slab, lattice offset {boff}')
    for des, tr in TRS.items():
        xb = bandlimit(bli[bok], (bpd * ss(bt1, tr))[bok], dims, boff[2])
        xw, xf = out[des]
        np.savez(os.path.join(OUT, f'truth_x_{des}.npz'), water=xw.astype(np.complex64), fat=xf.astype(np.complex64), blood=xb.astype(np.complex64))
        pb.log(f'{des}: saved (water {np.abs(xw).sum():.3e}, fat {np.abs(xf).sum():.3e}, blood {np.abs(xb).sum():.3e})')
    pb.log('done')
